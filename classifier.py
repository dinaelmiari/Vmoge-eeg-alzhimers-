# -*- coding: utf-8 -*-
"""
Standalone Classifier Ablation — Three Conditions + Youden Threshold
=====================================================================
EEG-Based Alzheimer's Detection | Amsterdam UMC Masters Thesis

PURPOSE
-------
This script uses the ALREADY SAVED synthetic epochs from vmoge_v7_plots.py
to run three classifier conditions WITHOUT any VMoGE retraining:

  [BL]   Train: real HC + real AD              (baseline)
  [AUG]  Train: real HC + real AD + synth AD   (augmentation)
  [TSTR] Train: real HC + synth AD only         (train-synthetic-test-real)

Each condition is evaluated at:
  • threshold = 0.5         (standard, matches v7 results)
  • threshold = t* (Youden) (calibrated on validation set)

Both epoch-level and patient-level metrics are reported.

TOTAL COMPUTE: ~45 minutes on Hinton V100 (15 EEGNet trainings)
NO VMoGE retraining — uses saved synthetic epochs directly.

HOW TO RUN
----------
  python classifier_ablation.py
  # or submit as Slurm job — see bottom of file

REFERENCE
---------
  TSTR paradigm: Esteban et al. (2017) Real-valued Medical Time Series
                 Generation with Recurrent Conditional GANs. arXiv:1706.02633
  Youden index:  Youden, W.J. (1950) Index for rating diagnostic tests.
                 Cancer, 3(1), 32–35.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, json, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                              roc_curve, auc as sklearn_auc)
from scipy.stats import wilcoxon

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ★ edit paths only ★
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR    = '/scratch/delmiari/thesisproject/processed'
SYNTH_DIR   = '/scratch/delmiari/thesisproject/pipeline_results/synthetic_epochs'
RESULTS_DIR = '/scratch/delmiari/thesisproject/pipeline_results'
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'ablation_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Signal — must match vmoge_v7_plots.py exactly
N_CHANNELS   = 19
N_SAMPLES    = 1000
SFREQ        = 500
RANDOM_STATE = 42
NUM_FOLDS    = 5

# EEGNet training
EEGNET_EPOCHS    = 150
EEGNET_LR        = 1e-3
EEGNET_BATCH     = 32
EEGNET_PATIENCE  = 10
EEGNET_DROPOUT   = 0.5
USE_FOCAL_LOSS   = True
FOCAL_GAMMA      = 2.0
AD_WEIGHT_MULT   = 3.0

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEM = (DEVICE.type == 'cuda')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("Ablation")

print(f"Device: {DEVICE}")
print(f"SAVE_DIR:  {SAVE_DIR}")
print(f"SYNTH_DIR: {SYNTH_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# EEGNET MODEL (identical to vmoge_v7_plots.py)
# ─────────────────────────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_ch=N_CHANNELS, n_t=N_SAMPLES, dropout=EEGNET_DROPOUT):
        super().__init__()
        F1=8; D=2; F2=16
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        with torch.no_grad():
            self._flat = self.block2(self.block1(torch.zeros(1,1,n_ch,n_t))).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(self._flat, 2))
    def forward(self, x):
        return self.fc(self.block2(self.block1(x)))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=FOCAL_GAMMA):
        super().__init__(); self.weight=weight; self.gamma=gamma
    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt  = torch.exp(-ce)
        return (((1-pt)**self.gamma)*ce).mean()


def make_loader(X, y, batch_size, shuffle=True):
    Xt = torch.FloatTensor(X.astype(np.float32)).unsqueeze(1)
    yt = torch.LongTensor(y.astype(np.int64))
    if shuffle:
        w       = 1.0 / np.bincount(y)[y]
        sampler = WeightedRandomSampler(torch.FloatTensor(w), len(w), True)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, sampler=sampler)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)


def train_eegnet(X_tr, y_tr, X_val, y_val, seed=42, label=""):
    torch.manual_seed(seed)
    model  = EEGNet().to(DEVICE)
    counts = np.bincount(y_tr)
    wb     = len(y_tr) / (len(counts) * counts)
    wb[1] *= AD_WEIGHT_MULT
    cw     = torch.FloatTensor(wb).to(DEVICE)
    log.info(f"  [{label}] HC:{cw[0]:.3f} AD:{cw[1]:.3f} | "
             f"n_HC={int((y_tr==0).sum())} n_AD={int((y_tr==1).sum())}")
    crit  = FocalLoss(weight=cw) if USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=cw)
    opt   = optim.Adam(model.parameters(), lr=EEGNET_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EEGNET_EPOCHS, 1e-5)
    ltr   = make_loader(X_tr,  y_tr,  EEGNET_BATCH, shuffle=True)
    lval  = make_loader(X_val, y_val, EEGNET_BATCH, shuffle=False)
    best_val, best_state, no_imp = float('inf'), None, 0
    for epoch in range(EEGNET_EPOCHS):
        model.train()
        for Xb, yb in ltr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(Xb), yb).backward(); opt.step()
        sched.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for Xb, yb in lval:
                vl += F.cross_entropy(model(Xb.to(DEVICE)), yb.to(DEVICE), weight=cw).item()
        vl /= len(lval)
        if vl < best_val - 1e-4:
            best_val, no_imp = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= EEGNET_PATIENCE:
                log.info(f"  [{label}] Early stop ep {epoch+1}"); break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def get_probs(model, X, batch_size=256):
    model.eval()
    all_probs = []
    Xt = torch.FloatTensor(X.astype(np.float32)).unsqueeze(1)
    for i in range(0, len(Xt), batch_size):
        xb     = Xt[i:i+batch_size].to(DEVICE)
        logits = model(xb)
        all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.array(all_probs)


def youden_threshold(y_val, val_probs):
    """
    Find optimal classification threshold using Youden index.
    t* = argmax(sensitivity + specificity - 1)
    Computed on VALIDATION set, applied to TEST set.
    Reference: Youden (1950) Cancer 3(1):32-35
    """
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    youden  = tpr - fpr                    # = sensitivity + specificity - 1
    best_idx = np.argmax(youden)
    t_star   = float(thresholds[best_idx])
    log.info(f"  Youden t* = {t_star:.4f}  "
             f"(sens={tpr[best_idx]*100:.1f}% spec={(1-fpr[best_idx])*100:.1f}% "
             f"at this threshold on val set)")
    return t_star


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    try:    auc = float(roc_auc_score(y_true, probs) * 100)
    except: auc = float('nan')
    return {
        'auc':         auc,
        'sensitivity': float(tp/(tp+fn+1e-10)*100),
        'specificity': float(tn/(tn+fp+1e-10)*100),
        'f1_macro':    float(f1_score(y_true, preds, average='macro')*100),
        'threshold':   float(threshold),
        'tp':int(tp), 'tn':int(tn), 'fp':int(fp), 'fn':int(fn),
    }


def patient_soft_vote(probs, y_te, sub_te, threshold=0.5):
    """Mean AD probability per patient → classify if mean >= threshold."""
    unique_subs = np.unique(sub_te)
    pat_probs, pat_preds, pat_labels = [], [], []
    for sub in unique_subs:
        mask = (sub_te == sub)
        mean_p   = probs[mask].mean()
        true_lab = y_te[mask][0]
        pat_probs.append(float(mean_p))
        pat_preds.append(int(mean_p >= threshold))
        pat_labels.append(int(true_lab))
    return compute_metrics(np.array(pat_labels), np.array(pat_probs), threshold)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ABLATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation():
    log.info("Loading data...")

    def _load(fname):
        path = os.path.join(SAVE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        return np.load(path, allow_pickle=True)

    X_all  = np.concatenate([_load('X_train.npy'), _load('X_test.npy')]).astype(np.float32)
    y_all  = np.concatenate([_load('y_train.npy'), _load('y_test.npy')]).astype(np.int64)
    sub_raw= np.concatenate([_load('train_subject_ids.npy'), _load('test_subject_ids.npy')])
    sub_all= np.array([int(str(s).replace('sub-','')) for s in sub_raw], dtype=np.int64) \
             if sub_raw.dtype.kind in {'U','S','O'} else sub_raw.astype(np.int64)

    log.info(f"Data: {X_all.shape} | HC={(y_all==0).sum()} | AD={(y_all==1).sum()}")

    # Reconstruct IDENTICAL fold splits (same seed = same splits as v7)
    unique_subs = np.unique(sub_all)
    sub_labels  = np.array([y_all[sub_all==s][0] for s in unique_subs])
    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    conditions  = ['BL', 'AUG', 'TSTR']
    all_results = []
    all_roc_data= {c: [] for c in conditions}

    for fold_idx, (tv_idx, te_idx) in enumerate(skf.split(unique_subs, sub_labels)):
        fold = fold_idx + 1
        log.info(f"\n{'='*65}")
        log.info(f"  FOLD {fold} / {NUM_FOLDS}")
        log.info(f"{'='*65}")

        # ── Reconstruct splits (deterministic — identical to v7) ──────────────
        tv_subs  = unique_subs[tv_idx]; te_subs = unique_subs[te_idx]
        rng      = np.random.default_rng(fold_idx*7 + RANDOM_STATE)
        perm     = rng.permutation(len(tv_subs))
        n_val    = max(2, int(len(tv_subs)*0.20))
        val_subs = tv_subs[perm[:n_val]]; tr_subs = tv_subs[perm[n_val:]]

        def get(subs):
            mask = np.isin(sub_all, subs)
            return X_all[mask], y_all[mask], sub_all[mask]

        X_tr,  y_tr,  sub_tr  = get(tr_subs)
        X_val, y_val, sub_val = get(val_subs)
        X_te,  y_te,  sub_te  = get(te_subs)

        log.info(f"  Train: {len(X_tr)} ({(y_tr==0).sum()} HC, {(y_tr==1).sum()} AD) | "
                 f"Val: {len(X_val)} | Test: {len(X_te)}")

        # ── Load saved synthetic epochs for this fold ─────────────────────────
        synth_X_path = os.path.join(SYNTH_DIR, f'synth_fold{fold}_X.npy')
        synth_y_path = os.path.join(SYNTH_DIR, f'synth_fold{fold}_y.npy')

        if not os.path.exists(synth_X_path):
            log.error(f"  Synthetic file not found: {synth_X_path}")
            log.error("  Run vmoge_v7_plots.py first to generate and save synthetic epochs.")
            continue

        synth_X = np.load(synth_X_path).astype(np.float32)
        synth_y = np.load(synth_y_path).astype(np.int64)
        log.info(f"  Loaded synthetic: {synth_X.shape} | all AD={( synth_y==1).all()}")

        # ── Build three training sets ─────────────────────────────────────────
        ad_mask   = (y_tr == 1)
        X_tr_ad   = X_tr[ad_mask];    y_tr_ad = y_tr[ad_mask]
        X_tr_hc   = X_tr[~ad_mask];   y_tr_hc = y_tr[~ad_mask]

        training_sets = {
            'BL':   (np.concatenate([X_tr_hc, X_tr_ad]),
                     np.concatenate([y_tr_hc,  y_tr_ad])),
            'AUG':  (np.concatenate([X_tr_hc, X_tr_ad, synth_X]),
                     np.concatenate([y_tr_hc,  y_tr_ad, synth_y])),
            'TSTR': (np.concatenate([X_tr_hc, synth_X]),
                     np.concatenate([y_tr_hc,  synth_y])),
        }

        fold_results = {'fold': fold}

        for cond in conditions:
            log.info(f"\n  ── {cond} ────────────────────────────────────────")
            X_c, y_c = training_sets[cond]

            # ── Train EEGNet ──────────────────────────────────────────────────
            model = train_eegnet(X_c, y_c, X_val, y_val,
                                  seed=fold_idx*31 + conditions.index(cond)*7 + 42,
                                  label=cond)

            # ── Get probabilities on val and test ─────────────────────────────
            val_probs  = get_probs(model, X_val)
            test_probs = get_probs(model, X_te)

            # ── Youden threshold from validation set ──────────────────────────
            t_star = youden_threshold(y_val, val_probs)

            # ── Evaluate at both thresholds ───────────────────────────────────
            for thr_name, thr in [('fixed_0.5', 0.5), ('youden', t_star)]:

                ep_m  = compute_metrics(y_te,  test_probs, thr)
                pat_m = patient_soft_vote(test_probs, y_te, sub_te, thr)

                key = f"{cond}_{thr_name}"
                fold_results[f"{key}_epoch"]   = ep_m
                fold_results[f"{key}_patient"]  = pat_m

                log.info(f"  [{cond} thr={thr:.3f}] "
                         f"Epoch:   AUC={ep_m['auc']:.1f}% "
                         f"Sens={ep_m['sensitivity']:.1f}% "
                         f"Spec={ep_m['specificity']:.1f}%")
                log.info(f"  [{cond} thr={thr:.3f}] "
                         f"Patient: AUC={pat_m['auc']:.1f}% "
                         f"Sens={pat_m['sensitivity']:.1f}% "
                         f"Spec={pat_m['specificity']:.1f}%")

            # Store ROC data for plotting
            all_roc_data[cond].append({
                'fold': fold,
                'y_true': y_te.tolist(),
                'y_prob': test_probs.tolist(),
            })

        all_results.append(fold_results)

        # Save after every fold
        out = os.path.join(RESULTS_DIR, 'ablation_results.json')
        with open(out, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"\n  Saved → {out}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  ABLATION SUMMARY — 5-FOLD MEAN ± STD")
    log.info(f"{'='*65}")

    summary_rows = []
    header = f"{'Condition':<10} {'Threshold':<12} {'Level':<8} " \
             f"{'AUC':>10} {'Sensitivity':>13} {'Specificity':>13} {'F1':>10}"
    log.info("  " + header)
    log.info("  " + "-"*70)

    for cond in conditions:
        for thr_name in ['fixed_0.5', 'youden']:
            for level in ['epoch', 'patient']:
                key = f"{cond}_{thr_name}_{level}"
                vals = {m: [r[f"{cond}_{thr_name}_{level}"][m]
                            for r in all_results]
                        for m in ['auc','sensitivity','specificity','f1_macro']}
                row = {
                    'condition': cond, 'threshold': thr_name,
                    'level': level, **{m: float(np.nanmean(v)) for m, v in vals.items()},
                    **{f"{m}_std": float(np.nanstd(v)) for m, v in vals.items()},
                }
                summary_rows.append(row)
                log.info(f"  {cond:<10} {thr_name:<12} {level:<8} "
                         f"{row['auc']:>6.1f}±{row['auc_std']:.1f}  "
                         f"{row['sensitivity']:>6.1f}±{row['sensitivity_std']:.1f}  "
                         f"{row['specificity']:>6.1f}±{row['specificity_std']:.1f}  "
                         f"{row['f1_macro']:>6.1f}±{row['f1_macro_std']:.1f}")

    # ── Statistical tests (Wilcoxon signed-rank, N=5) ─────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  STATISTICAL TESTS (Wilcoxon signed-rank, N=5 folds)")
    log.info(f"{'='*65}")
    for level in ['epoch', 'patient']:
        for thr_name in ['youden']:
            bl_aucs  = [r[f"BL_{thr_name}_{level}"]['auc']   for r in all_results]
            aug_aucs = [r[f"AUG_{thr_name}_{level}"]['auc']  for r in all_results]
            tstr_aucs= [r[f"TSTR_{thr_name}_{level}"]['auc'] for r in all_results]
            try:
                _, p_bl_aug  = wilcoxon(bl_aucs, aug_aucs)
                _, p_bl_tstr = wilcoxon(bl_aucs, tstr_aucs)
                _, p_aug_tstr= wilcoxon(aug_aucs, tstr_aucs)
            except Exception:
                p_bl_aug = p_bl_tstr = p_aug_tstr = float('nan')
            log.info(f"  [{level} Youden] BL vs AUG:   p={p_bl_aug:.4f} "
                     f"{'*' if p_bl_aug<0.05 else 'ns'}")
            log.info(f"  [{level} Youden] BL vs TSTR:  p={p_bl_tstr:.4f} "
                     f"{'*' if p_bl_tstr<0.05 else 'ns'}")
            log.info(f"  [{level} Youden] AUG vs TSTR: p={p_aug_tstr:.4f} "
                     f"{'*' if p_aug_tstr<0.05 else 'ns'}")

    # ── Save full summary ─────────────────────────────────────────────────────
    out_summary = os.path.join(RESULTS_DIR, 'ablation_summary.json')
    with open(out_summary, 'w') as f:
        json.dump({'fold_results': all_results, 'summary': summary_rows}, f,
                  indent=2, default=str)
    log.info(f"\n  Summary saved → {out_summary}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _make_ablation_plots(all_results, summary_rows, all_roc_data)
    return all_results, summary_rows


def _make_ablation_plots(all_results, summary_rows, all_roc_data):
    folds  = list(range(1, NUM_FOLDS+1))
    colors = {'BL':'steelblue', 'AUG':'mediumseagreen', 'TSTR':'mediumpurple'}

    # ── 1. AUC comparison — all conditions × both thresholds × epoch + patient
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (level, thr_name), title in zip(
        axes.flat,
        [('epoch','fixed_0.5'), ('epoch','youden'),
         ('patient','fixed_0.5'), ('patient','youden')],
        ['Epoch  |  threshold=0.5',  'Epoch  |  Youden threshold',
         'Patient | threshold=0.5', 'Patient | Youden threshold'],
    ):
        x = np.arange(NUM_FOLDS); w = 0.25
        for k, cond in enumerate(('BL','AUG','TSTR')):
            aucs = [r[f"{cond}_{thr_name}_{level}"]['auc'] for r in all_results]
            bars = ax.bar(x + (k-1)*w, aucs, w, label=cond,
                          color=colors[cond], alpha=0.85)
            for bar, v in zip(bars, aucs):
                ax.text(bar.get_x()+w/2, bar.get_height()+0.5, f'{v:.0f}',
                        ha='center', fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([f'F{f}' for f in folds])
        ax.set_ylim(0, 115); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('AUC (%)')
    plt.suptitle('AUC-ROC: BL vs AUG vs TSTR\n'
                 '(top row = epoch level, bottom = patient level)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ablation_auc_comparison.png')
    plt.savefig(path, dpi=150); plt.close(); log.info(f"  Plot: {path}")

    # ── 2. Sensitivity × Specificity grid (the trade-off chart) ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, thr_name in zip(axes, ['fixed_0.5', 'youden']):
        for cond in ('BL','AUG','TSTR'):
            sens = [r[f"{cond}_{thr_name}_epoch"]['sensitivity'] for r in all_results]
            spec = [r[f"{cond}_{thr_name}_epoch"]['specificity'] for r in all_results]
            ax.scatter(spec, sens, color=colors[cond], s=80, alpha=0.8,
                       label=f"{cond} (mean sens={np.mean(sens):.0f}%)",
                       zorder=3)
            ax.plot(np.mean(spec), np.mean(sens), marker='*',
                    markersize=16, color=colors[cond], zorder=4)
        ax.set_xlabel('Specificity (%)', fontsize=11)
        ax.set_ylabel('Sensitivity (%)', fontsize=11)
        thr_label = 'threshold=0.5' if thr_name=='fixed_0.5' else 'Youden threshold'
        ax.set_title(f'Sensitivity–Specificity Trade-off\n({thr_label})', fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_xlim(-5, 110); ax.set_ylim(-5, 110)
        ax.plot([0,100],[100,0],'--',color='grey',lw=0.8,alpha=0.5)  # iso-accuracy line
    plt.suptitle('BL vs AUG vs TSTR — Epoch Level\n(★ = fold mean)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ablation_sens_spec.png')
    plt.savefig(path, dpi=150); plt.close(); log.info(f"  Plot: {path}")

    # ── 3. Summary bar — mean ± std, Youden threshold, both levels ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics   = ['auc', 'sensitivity', 'specificity', 'f1_macro']
    labels    = ['AUC', 'Sensitivity', 'Specificity', 'F1']
    x = np.arange(len(metrics)); w = 0.25

    for ax, level in zip(axes, ['epoch', 'patient']):
        for k, cond in enumerate(('BL','AUG','TSTR')):
            means = [np.nanmean([r[f"{cond}_youden_{level}"][m]
                                 for r in all_results]) for m in metrics]
            stds  = [np.nanstd([r[f"{cond}_youden_{level}"][m]
                                for r in all_results]) for m in metrics]
            bars  = ax.bar(x + (k-1)*w, means, w, yerr=stds, capsize=4,
                           label=cond, color=colors[cond], alpha=0.85)
            for bar, v in zip(bars, means):
                ax.text(bar.get_x()+w/2, bar.get_height()+1.5, f'{v:.0f}',
                        ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 120); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Score (%)')
        ax.set_title(f'Mean ± Std across 5 Folds\n({level.capitalize()} level, Youden threshold)')
    plt.suptitle('BL vs AUG vs TSTR — All Metrics (Youden threshold)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ablation_summary_bar.png')
    plt.savefig(path, dpi=150); plt.close(); log.info(f"  Plot: {path}")

    # ── 4. ROC curves — all three conditions on one plot ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    mean_fpr  = np.linspace(0, 1, 100)
    for ax, cond in zip(axes, ('BL','AUG','TSTR')):
        interp_tprs = []
        for fd in all_roc_data[cond]:
            fpr, tpr, _ = roc_curve(fd['y_true'], fd['y_prob'])
            auc_v       = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, alpha=0.4, lw=1.2,
                    label=f"Fold {fd['fold']} (AUC={auc_v:.3f})",
                    color=colors[cond])
            interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
        if interp_tprs:
            mt  = np.mean(interp_tprs, 0); st = np.std(interp_tprs, 0)
            ax.plot(mean_fpr, mt, 'k--', lw=2,
                    label=f'Mean AUC={sklearn_auc(mean_fpr,mt):.3f}')
            ax.fill_between(mean_fpr, mt-st, mt+st, alpha=0.12, color='grey')
        ax.plot([0,1],[0,1],'--',color='grey',lw=0.8)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f'ROC — {cond}', color=colors[cond], fontweight='bold')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle('ROC Curves: BL vs AUG vs TSTR (epoch level)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ablation_roc.png')
    plt.savefig(path, dpi=150); plt.close(); log.info(f"  Plot: {path}")

    # ── 5. Threshold effect: 0.5 vs Youden, per condition ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, cond in zip(axes, ('BL','AUG','TSTR')):
        for metric, marker, ls in [('sensitivity','o','-'),('specificity','s','--')]:
            v_fixed  = [r[f"{cond}_fixed_0.5_epoch"][metric] for r in all_results]
            v_youden = [r[f"{cond}_youden_epoch"][metric]    for r in all_results]
            ax.plot(folds, v_fixed,  marker=marker, ls=ls,  color='tomato',
                    label=f'{metric} (0.5)', alpha=0.8)
            ax.plot(folds, v_youden, marker=marker, ls=ls, color='steelblue',
                    label=f'{metric} (Youden)', alpha=0.8)
        ax.set_title(f'{cond} — threshold effect', fontweight='bold',
                     color=colors[cond])
        ax.set_xlabel('Fold'); ax.set_ylabel('%')
        ax.set_ylim(0, 110); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('Effect of Youden Threshold Calibration per Condition', fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ablation_threshold_effect.png')
    plt.savefig(path, dpi=150); plt.close(); log.info(f"  Plot: {path}")

    log.info(f"\n  All ablation plots saved to: {PLOTS_DIR}")
