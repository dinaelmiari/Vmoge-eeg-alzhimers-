# -*- coding: utf-8 -*-
"""

# =============================================================================
# CELL 1 — IMPORTS
# =============================================================================
import os, sys, json, logging, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.linalg import sqrtm
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                              roc_curve, auc as sklearn_auc)

# =============================================================================
# CELL 2 — CONFIGURATION  
# =============================================================================
IS_COLAB = False

if IS_COLAB:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    SAVE_DIR    = '/content/drive/MyDrive/thesisproject/processed'
    RESULTS_DIR = '/content/drive/MyDrive/thesisproject/pipeline_results'
else:
    SAVE_DIR    = '/scratch/delmiari/thesisproject/processed'
    RESULTS_DIR = '/scratch/delmiari/thesisproject/pipeline_results'

SYNTH_DIR = os.path.join(RESULTS_DIR, 'synthetic_epochs')   # [S2]
PLOTS_DIR = os.path.join(RESULTS_DIR, 'thesis_plots')        # [S5]
for d in [RESULTS_DIR, SYNTH_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Signal
N_CHANNELS  = 19
N_SAMPLES   = 1000
SFREQ       = 500
N_RFFT      = N_SAMPLES // 2 + 1
ALL_FREQS   = np.fft.rfftfreq(N_SAMPLES, d=1.0/SFREQ)
FREQ_MASK   = (ALL_FREQS >= 0.5) & (ALL_FREQS <= 40.0)
N_FREQ_CLIN = int(FREQ_MASK.sum())

# CV
NUM_FOLDS    = 5
RANDOM_STATE = 42

# EEGNet
EEGNET_EPOCHS   = 150
EEGNET_LR       = 1e-3
EEGNET_BATCH    = 32
EEGNET_PATIENCE = 10
EEGNET_DROPOUT  = 0.5

# VMoGE
LATENT_DIM    = 128
DDPM_T        = 1000
BETA_START    = 1e-4
BETA_END      = 0.02
WARMUP_EPOCHS = 80
JOINT_EPOCHS  = 250
VMOGE_LR      = 1e-4
VMOGE_BATCH   = 16
LAMBDA_REC    = 0.2

# Generation — [S1]: no fixed cap, generate to match HC
DDIM_STEPS              = 50
LATENT_NOISE            = 0.15
APPLY_SPECTRAL_CORRECTION = True
MAX_SYNTH_PER_BATCH     = 50    # DDIM batch size (memory limit, not total cap)

# Classifier
USE_FOCAL_LOSS       = True
FOCAL_GAMMA          = 2.0
AD_WEIGHT_MULTIPLIER = 3.0

# Quality gate
FD_THRESHOLD = 5.0

# Channel asymmetry
LEFT_CH  = [0, 2, 4, 6, 11, 13, 15]
RIGHT_CH = [1, 3, 5, 7, 12, 14, 16]

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEM = (DEVICE.type == 'cuda')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("VMoGE_v7")

print(f"Device: {DEVICE}")
print(f"SFREQ={SFREQ} | N_SAMPLES={N_SAMPLES} | N_CLINICAL_BINS={N_FREQ_CLIN}")
print(f"NUM_FOLDS={NUM_FOLDS} | LATENT_DIM={LATENT_DIM}")
print(f"[S1] Generate to match HC  [S2] Save synth  "
      f"[S3] Patient-level eval  [S4] Global baseline")
assert N_FREQ_CLIN == 80, f"Expected 80 bins, got {N_FREQ_CLIN}. Check SFREQ."

# =============================================================================
# CELL 3 — DIFFUSION SCHEDULE
# =============================================================================
def make_schedule(device=DEVICE):
    betas = torch.linspace(BETA_START, BETA_END, DDPM_T).to(device)
    ab    = torch.cumprod(1 - betas, 0)
    return {'betas': betas, 'alpha_bars': ab,
            'sqrt_ab': torch.sqrt(ab), 'sqrt_1mab': torch.sqrt(1 - ab)}

SCHEDULE = make_schedule()
print(f"DDPM schedule ready (T={DDPM_T})")

# =============================================================================
# CELL 4 — FFT REPRESENTATION
# =============================================================================
HANNING = np.hanning(N_SAMPLES)

def signed_log(x):   return np.sign(x) * np.log1p(np.abs(x))
def inv_signed_log(x): return np.sign(x) * np.expm1(np.abs(x))

def eeg_to_complex_spectrum(X):
    N        = len(X)
    clin_out = np.zeros((N, 2, N_CHANNELS, N_FREQ_CLIN), dtype=np.float32)
    full_out = np.zeros((N, 2, N_CHANNELS, N_RFFT),      dtype=np.float32)
    for i, epoch in enumerate(X):
        fft_c          = np.fft.rfft(epoch * HANNING, axis=-1)
        rl             = signed_log(fft_c.real)
        im             = signed_log(fft_c.imag)
        full_out[i,0]  = rl;  full_out[i,1]  = im
        clin_out[i,0]  = rl[:, FREQ_MASK]
        clin_out[i,1]  = im[:, FREQ_MASK]
    return clin_out, full_out

def complex_spectrum_to_eeg(spec_full_norm, spec_mean, spec_std):
    spec = spec_full_norm * spec_std + spec_mean
    eeg  = np.fft.irfft(inv_signed_log(spec[:,0]) + 1j*inv_signed_log(spec[:,1]),
                         n=N_SAMPLES, axis=-1)
    for i in range(len(eeg)):
        s = eeg[i].std() + 1e-10
        eeg[i] = (eeg[i] - eeg[i].mean()) / s
    return eeg.astype(np.float32)

# =============================================================================
# CELL 5 — ADJACENCY MATRIX
# =============================================================================
def build_adjacency(device=DEVICE):
    names = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
             'T3','C3','Cz','C4','T4',
             'T5','P3','Pz','P4','T6','O1','O2']
    idx = {n: i for i, n in enumerate(names)}
    n   = len(names)
    edges = [
        ('Fp1',['Fp2','F7','F3']),   ('Fp2',['Fp1','F8','F4']),
        ('F7', ['Fp1','F3','T3']),   ('F3', ['Fp1','F7','Fz','C3']),
        ('Fz', ['F3','F4','Cz']),    ('F4', ['Fp2','Fz','F8','C4']),
        ('F8', ['Fp2','F4','T4']),   ('T3', ['F7','C3','T5']),
        ('C3', ['F3','T3','Cz','P3']),('Cz',['Fz','C3','C4','Pz']),
        ('C4', ['F4','Cz','T4','P4']),('T4',['F8','C4','T6']),
        ('T5', ['T3','P3','O1']),    ('P3', ['C3','T5','Pz','O1']),
        ('Pz', ['Cz','P3','P4']),    ('P4', ['C4','Pz','T6','O2']),
        ('T6', ['T4','P4','O2']),    ('O1', ['T5','P3','Pz','O2']),
        ('O2', ['T6','P4','Pz','O1']),
    ]
    A = np.zeros((n, n))
    for ch, nbs in edges:
        for nb in nbs:
            A[idx[ch], idx[nb]] = A[idx[nb], idx[ch]] = 1.0
    At    = A + np.eye(n)
    D_inv = np.diag(1.0 / np.sqrt(At.sum(1) + 1e-10))
    return torch.FloatTensor(D_inv @ At @ D_inv).to(device)

ADJ = build_adjacency()

# =============================================================================
# CELL 6 — MODEL ARCHITECTURES 
# =============================================================================

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


class GraphConvLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=True)
    def forward(self, H, A):
        return F.relu(self.W(torch.bmm(A.unsqueeze(0).expand(H.size(0),-1,-1), H)))


class ThreeLayerGCN(nn.Module):
    def __init__(self, in_f, hidden=32, out_f=32):
        super().__init__()
        self.l1 = GraphConvLayer(in_f,   hidden)
        self.l2 = GraphConvLayer(hidden, hidden)
        self.l3 = GraphConvLayer(hidden, out_f)
        self.r1 = nn.Linear(in_f, hidden, bias=False) if in_f!=hidden else nn.Identity()
        self.r3 = nn.Linear(hidden, out_f, bias=False) if hidden!=out_f else nn.Identity()
    def forward(self, H, A):
        H1 = self.l1(H, A)  + self.r1(H)
        H2 = self.l2(H1, A) + H1
        return  self.l3(H2, A) + self.r3(H2)


class SpectralFreqExpert(nn.Module):
    def __init__(self, kernel, out_f=32):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=kernel, padding=kernel//2, bias=False),
            nn.BatchNorm1d(16), nn.ELU(), nn.AdaptiveAvgPool1d(16))
        self.gcn = ThreeLayerGCN(16*16, 32, out_f)
    def forward(self, spec, adj):
        B, C, Nch, Nf = spec.shape
        x = spec.permute(0,2,1,3).reshape(B*Nch, C, Nf)
        h = self.freq_conv(x).view(B, Nch, -1)
        return self.gcn(h, adj)


class GatingNetwork(nn.Module):
    def __init__(self, n_exp=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, n_exp), nn.Softmax(dim=-1))
    def forward(self, spec): return self.net(spec)


class VMoGE_Encoder(nn.Module):
    KERNELS = [32, 16, 8, 4]
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.experts = nn.ModuleList([SpectralFreqExpert(k) for k in self.KERNELS])
        self.gate    = GatingNetwork(4)
        self.fc_z    = nn.Sequential(
            nn.Linear(N_CHANNELS*32, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(256, latent_dim), nn.Tanh())
    def forward(self, spec, adj):
        w    = self.gate(spec)
        outs = [e(spec, adj) for e in self.experts]
        comb = sum(w[:, i].view(-1,1,1) * outs[i] for i in range(4))
        return self.fc_z(comb.flatten(1))


class ComplexRecDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 2*N_CHANNELS*N_FREQ_CLIN))
    def forward(self, z): return self.net(z).view(-1, 2, N_CHANNELS, N_FREQ_CLIN)


class SinEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-np.log(10000)*torch.arange(half, device=t.device)/half)
        ang   = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], 1)


class SpectralResBlock(nn.Module):
    def __init__(self, ch, t_dim=128, z_dim=LATENT_DIM):
        super().__init__()
        self.c1 = nn.Conv2d(ch,ch,3,padding=1); self.c2 = nn.Conv2d(ch,ch,3,padding=1)
        self.n1 = nn.GroupNorm(min(8,ch),ch);   self.n2 = nn.GroupNorm(min(8,ch),ch)
        self.act= nn.SiLU()
        self.tp = nn.Linear(t_dim, ch); self.zp = nn.Linear(z_dim, ch)
    def forward(self, x, te, z):
        h = self.act(self.n1(self.c1(x)))
        h = h + self.tp(te).unsqueeze(-1).unsqueeze(-1)
        h = h + self.zp(z).unsqueeze(-1).unsqueeze(-1)
        return x + self.act(self.n2(self.c2(h)))


class SpectralDiffDecoder(nn.Module):
    def __init__(self, in_ch=2, t_dim=128, z_dim=LATENT_DIM):
        super().__init__()
        self.te  = SinEmbed(t_dim)
        self.tmp = nn.Sequential(nn.Linear(t_dim,t_dim*2),nn.SiLU(),nn.Linear(t_dim*2,t_dim))
        self.inc = nn.Conv2d(in_ch,32,3,padding=1)
        self.e1  = SpectralResBlock(32);  self.d1 = nn.Conv2d(32,64,(1,4),stride=(1,2),padding=(0,1))
        self.e2  = SpectralResBlock(64);  self.d2 = nn.Conv2d(64,128,(1,4),stride=(1,2),padding=(0,1))
        self.mid = SpectralResBlock(128)
        self.u2  = nn.ConvTranspose2d(128,64,(1,4),stride=(1,2),padding=(0,1));  self.de2 = SpectralResBlock(128)
        self.u1  = nn.ConvTranspose2d(128,32,(1,4),stride=(1,2),padding=(0,1));  self.de1 = SpectralResBlock(64)
        self.out = nn.Sequential(nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,in_ch,1))
    def forward(self, x, t, z):
        te = self.tmp(self.te(t)); x = self.inc(x)
        e1 = self.e1(x,te,z);  x = self.d1(e1)
        e2 = self.e2(x,te,z);  x = self.d2(e2)
        x  = self.mid(x,te,z); x = self.u2(x)
        h,w = min(x.shape[2],e2.shape[2]),min(x.shape[3],e2.shape[3])
        x  = self.de2(torch.cat([x[:,:,:h,:w],e2[:,:,:h,:w]],1),te,z); x = self.u1(x)
        h,w = min(x.shape[2],e1.shape[2]),min(x.shape[3],e1.shape[3])
        x  = self.de1(torch.cat([x[:,:,:h,:w],e1[:,:,:h,:w]],1),te,z)
        x  = F.interpolate(x,size=(N_CHANNELS,N_RFFT),mode='bilinear',align_corners=False)
        return self.out(x)

# =============================================================================
# CELL 7 — TRAINING UTILITIES
# =============================================================================
def add_noise_4d(x0, t, sch):
    noise = torch.randn_like(x0)
    sab   = sch['sqrt_ab'][t].view(-1,1,1,1)
    s1m   = sch['sqrt_1mab'][t].view(-1,1,1,1)
    return sab*x0 + s1m*noise, noise

def make_weighted_loader(X, y, batch_size, shuffle=True):
    Xt = torch.FloatTensor(X.astype(np.float32)).unsqueeze(1)   # explicit cast — safe against object arrays
    yt = torch.LongTensor(y.astype(np.int64))
    if shuffle:
        w       = 1.0 / np.bincount(y)[y]
        sampler = WeightedRandomSampler(torch.FloatTensor(w), len(w), True)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, sampler=sampler)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)

def compute_epoch_metrics(y_true, y_pred, y_prob):
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    try:    auc = float(roc_auc_score(y_true, y_prob)*100)
    except: auc = float('nan')
    return {
        'auc':         auc,
        'sensitivity': float(tp/(tp+fn+1e-10)*100),
        'specificity': float(tn/(tn+fp+1e-10)*100),
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro')*100),
        'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn),
    }

@torch.no_grad()
def predict_probs(model, X_np, batch_size=256):
    """Return (preds, probs) — batched to avoid OOM."""
    model.eval()
    all_probs, all_preds = [], []
    Xt = torch.FloatTensor(X_np).unsqueeze(1)
    for i in range(0, len(Xt), batch_size):
        xb = Xt[i:i+batch_size].to(DEVICE)
        logits = model(xb)
        all_probs.extend(torch.softmax(logits,1)[:,1].cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(all_preds), np.array(all_probs)

# =============================================================================
# CELL 8 — [S3] PATIENT-LEVEL SOFT VOTING
# Soft voting: mean AD probability across all epochs of one patient.
# Reference: CRCC arxiv 2602.19138 Eq.(6); LEAD arxiv 2502.01678 Sec.2.2
# =============================================================================
def patient_level_evaluation(model, X_te, y_te, sub_te, label=""):
    """
    Aggregate epoch-level probabilities per patient via soft voting.
    Returns patient-level metrics AND a per-patient probability dict.
    """
    _, ep_probs = predict_probs(model, X_te)
    unique_subs = np.unique(sub_te)

    pat_probs, pat_preds, pat_labels = [], [], []
    for sub in unique_subs:
        mask     = (sub_te == sub)
        mean_prob = ep_probs[mask].mean()   # soft vote = mean AD probability
        true_lab  = y_te[mask][0]
        pat_probs.append(float(mean_prob))
        pat_preds.append(int(mean_prob >= 0.5))
        pat_labels.append(int(true_lab))

    pat_probs  = np.array(pat_probs)
    pat_preds  = np.array(pat_preds)
    pat_labels = np.array(pat_labels)

    tn,fp,fn,tp = confusion_matrix(pat_labels, pat_preds, labels=[0,1]).ravel()
    try:    p_auc = float(roc_auc_score(pat_labels, pat_probs)*100)
    except: p_auc = float('nan')

    metrics = {
        'auc':         p_auc,
        'sensitivity': float(tp/(tp+fn+1e-10)*100),
        'specificity': float(tn/(tn+fp+1e-10)*100),
        'f1_macro':    float(f1_score(pat_labels, pat_preds, average='macro')*100),
        'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn),
        'n_patients':  len(unique_subs),
        'n_hc':        int((pat_labels==0).sum()),
        'n_ad':        int((pat_labels==1).sum()),
    }

    if label:
        log.info(f"  [{label}] Patient-level | "
                 f"AUC={p_auc:.1f}%  Sens={metrics['sensitivity']:.1f}%  "
                 f"Spec={metrics['specificity']:.1f}%  "
                 f"TP={tp} TN={tn} FP={fp} FN={fn}")

    return metrics, dict(zip([int(s) for s in unique_subs], zip(pat_probs, pat_labels)))

# =============================================================================
# CELL 9 — FOCAL LOSS + EEGNET TRAINING
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=FOCAL_GAMMA):
        super().__init__(); self.weight=weight; self.gamma=gamma
    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt  = torch.exp(-ce)
        return (((1-pt)**self.gamma)*ce).mean()

def train_eegnet(X_tr, y_tr, X_val, y_val, seed=42):
    torch.manual_seed(seed)
    model  = EEGNet().to(DEVICE)
    counts = np.bincount(y_tr)
    wb     = len(y_tr)/(len(counts)*counts)
    wb[1] *= AD_WEIGHT_MULTIPLIER
    cw     = torch.FloatTensor(wb).to(DEVICE)
    log.info(f"  EEGNet weights HC:{cw[0]:.3f} AD:{cw[1]:.3f} "
             f"(focal={'on' if USE_FOCAL_LOSS else 'off'} γ={FOCAL_GAMMA})")
    crit  = FocalLoss(weight=cw) if USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=cw)
    opt   = optim.Adam(model.parameters(), lr=EEGNET_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EEGNET_EPOCHS, 1e-5)
    ltr   = make_weighted_loader(X_tr, y_tr, EEGNET_BATCH, shuffle=True)
    lval  = make_weighted_loader(X_val,y_val,EEGNET_BATCH, shuffle=False)
    best_val, best_state, no_imp = float('inf'), None, 0
    for epoch in range(EEGNET_EPOCHS):
        model.train()
        for Xb,yb in ltr:
            Xb,yb = Xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); crit(model(Xb),yb).backward(); opt.step()
        sched.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for Xb,yb in lval:
                vl += F.cross_entropy(model(Xb.to(DEVICE)),yb.to(DEVICE),weight=cw).item()
        vl /= len(lval)
        if vl < best_val-1e-4: best_val,no_imp=vl,0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= EEGNET_PATIENCE: log.info(f"  EEGNet early stop ep {epoch+1}"); break
    model.load_state_dict(best_state)
    return model

# =============================================================================
# CELL 10 — VMoGE TRAINING
# =============================================================================
def train_vmoge(X_ad_fold, X_all_fold, y_all_fold, seed=42):
    torch.manual_seed(seed)
    log.info("    Computing spectral representations...")
    clin_all,full_all = eeg_to_complex_spectrum(X_all_fold)
    clin_ad, full_ad  = eeg_to_complex_spectrum(X_ad_fold)
    clin_mean=clin_all.mean(); clin_std=clin_all.std()+1e-10
    full_mean=full_all.mean(); full_std=full_all.std()+1e-10
    clin_all_norm=(clin_all-clin_mean)/clin_std
    clin_ad_norm =(clin_ad -clin_mean)/clin_std
    full_ad_norm =(full_ad -full_mean)/full_std

    encoder=VMoGE_Encoder().to(DEVICE); rec_dec=ComplexRecDecoder().to(DEVICE); diff_dec=SpectralDiffDecoder().to(DEVICE)
    warmup_history    = []
    joint_diff_history= []

    # Phase 1 — warm-up on ALL subjects
    log.info(f"    Phase 1: Warm-up ({WARMUP_EPOCHS} epochs, ALL subjects)...")
    w       = 1.0/np.bincount(y_all_fold)[y_all_fold]
    sampler = WeightedRandomSampler(torch.FloatTensor(w),len(w),True)
    clin_t  = torch.FloatTensor(clin_all_norm)
    lw = DataLoader(TensorDataset(clin_t,clin_t),VMOGE_BATCH,sampler=sampler,drop_last=True)
    opt_w=optim.AdamW(list(encoder.parameters())+list(rec_dec.parameters()),lr=VMOGE_LR,weight_decay=1e-4)
    sch_w=optim.lr_scheduler.CosineAnnealingLR(opt_w,WARMUP_EPOCHS,1e-6)
    for epoch in range(WARMUP_EPOCHS):
        encoder.train(); rec_dec.train()
        ep_loss = 0.0
        for sb,tb in lw:
            sb,tb=sb.to(DEVICE),tb.to(DEVICE)
            z=encoder(sb,ADJ); loss=F.mse_loss(rec_dec(z),tb)
            opt_w.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters())+list(rec_dec.parameters()),1.0)
            opt_w.step()
            ep_loss += loss.item()
        sch_w.step()
        warmup_history.append(ep_loss / len(lw))
        if (epoch+1)%20==0: log.info(f"      Warm-up {epoch+1}/{WARMUP_EPOCHS} | loss={warmup_history[-1]:.5f}")

    # Phase 2 — joint training on AD only
    log.info(f"    Phase 2: Joint training ({JOINT_EPOCHS} epochs, AD only)...")
    lad=DataLoader(TensorDataset(torch.FloatTensor(clin_ad_norm),torch.FloatTensor(full_ad_norm)),VMOGE_BATCH,shuffle=True,drop_last=True)
    opt_j=optim.AdamW(list(encoder.parameters())+list(diff_dec.parameters()),lr=VMOGE_LR,weight_decay=1e-4)
    sch_j=optim.lr_scheduler.CosineAnnealingLR(opt_j,JOINT_EPOCHS,1e-6)
    rec_dec.eval()
    for epoch in range(JOINT_EPOCHS):
        encoder.train(); diff_dec.train()
        for cb,fb in lad:
            cb,fb=cb.to(DEVICE),fb.to(DEVICE)
            z=encoder(cb,ADJ); t_idx=torch.randint(0,DDPM_T,(cb.size(0),),device=DEVICE)
            xn,eps=add_noise_4d(fb,t_idx,SCHEDULE)
            ld=F.mse_loss(diff_dec(xn,t_idx,z),eps)
            with torch.no_grad(): lr_=F.mse_loss(rec_dec(z.detach()),cb)*LAMBDA_REC
            loss=ld+lr_
            opt_j.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters())+list(diff_dec.parameters()),1.0)
            opt_j.step()
        sch_j.step()
        joint_diff_history.append(ld.item())
        if (epoch+1)%50==0: log.info(f"      Joint {epoch+1}/{JOINT_EPOCHS} | diff={ld.item():.5f} | rec={lr_.item():.5f}")

    encoder.eval(); all_z=[]
    with torch.no_grad():
        for i in range(0,len(clin_ad_norm),64):
            b=torch.FloatTensor(clin_ad_norm[i:i+64]).to(DEVICE)
            all_z.append(encoder(b,ADJ).cpu().numpy())
    Z_ad=np.concatenate(all_z); sigma_ad=Z_ad.std(0)
    log.info(f"    Z_ad: {Z_ad.shape} | sigma_ad mean: {sigma_ad.mean():.4f}")
    return encoder,diff_dec,full_mean,full_std,Z_ad,sigma_ad,warmup_history,joint_diff_history

# =============================================================================
# CELL 11 — [S1] GENERATION: MATCH HC COUNT 
# =============================================================================
@torch.no_grad()
def generate_synthetic_ad(encoder, diff_dec, full_mean, full_std, Z_ad, sigma_ad, n_samples):
    """Generate n_samples synthetic AD epochs. No fixed cap — caller decides n."""
    encoder.eval(); diff_dec.eval()
    step_idx = torch.linspace(0, DDPM_T-1, DDIM_STEPS+1).long().flip(0)
    all_spec = []
    for i in range(0, n_samples, MAX_SYNTH_PER_BATCH):
        n_this = min(MAX_SYNTH_PER_BATCH, n_samples-i)
        idx    = np.random.randint(0, len(Z_ad), n_this)
        z_samp = np.clip(Z_ad[idx] + LATENT_NOISE*sigma_ad*np.random.randn(n_this,LATENT_DIM), -1., 1.)
        z_t    = torch.FloatTensor(z_samp).to(DEVICE)
        x      = torch.randn(n_this, 2, N_CHANNELS, N_RFFT, device=DEVICE)
        for j in range(DDIM_STEPS):
            t_now=step_idx[j].to(DEVICE).expand(n_this); t_next=step_idx[j+1]
            np_=diff_dec(x,t_now,z_t)
            ab_now=SCHEDULE['alpha_bars'][step_idx[j]]
            ab_nxt=SCHEDULE['alpha_bars'][t_next] if t_next>=0 else torch.tensor(1.).to(DEVICE)
            x0p=((x-(1-ab_now).sqrt()*np_)/ab_now.sqrt()).clamp(-4,4)
            x=ab_nxt.sqrt()*x0p+(1-ab_nxt).sqrt()*np_
        all_spec.append(x.cpu().numpy())
    return complex_spectrum_to_eeg(np.concatenate(all_spec), full_mean, full_std)

# =============================================================================
# CELL 12 — SPECTRAL BIAS CORRECTION
# =============================================================================
def spectral_bias_correction(X_synth, X_real_ad):
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1./SFREQ)
    BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,40)}
    def mlbp(X,lo,hi,n=300):
        mask=(freqs>=lo)&(freqs<hi)
        return float(np.mean([np.log1p((np.abs(np.fft.rfft(ep,axis=-1))**2)[:,mask].mean()) for ep in X[:n]]))
    log.info("  Spectral bias correction:")
    corrections={}
    for band,(lo,hi) in BANDS.items():
        pr=mlbp(X_real_ad,lo,hi); ps=mlbp(X_synth,lo,hi)
        cf=np.exp(pr-ps); corrections[band]=(lo,hi,float(cf))
        log.info(f"    {band:5s}: err={100*(ps-pr)/(abs(pr)+1e-8):+.1f}%  factor={cf:.4f}")
    X_out=np.zeros_like(X_synth)
    for i,ep in enumerate(X_synth):
        fe=np.fft.rfft(ep,axis=-1)
        for band,(lo,hi,cf) in corrections.items():
            fe[:,(freqs>=lo)&(freqs<hi)] *= np.sqrt(cf)
        c=np.fft.irfft(fe,n=N_SAMPLES,axis=-1); s=c.std()+1e-10
        X_out[i]=(c-c.mean())/s
    return X_out.astype(np.float32), corrections

# =============================================================================
# CELL 13 — EVALUATION + FN PROFILING
# =============================================================================
def compute_spectral_metrics(X_real_ad, X_synth):
    freqs=np.fft.rfftfreq(N_SAMPLES,d=1./SFREQ)
    BANDS={'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,40)}
    def bp(X,lo,hi,n=200):
        mask=(freqs>=lo)&(freqs<hi)
        return np.array([np.mean([np.log1p((np.abs(np.fft.rfft(ep[c]))**2)[mask].mean()) for c in range(N_CHANNELS)]) for ep in X[:n]])
    def tar(X,n=200):
        ratios=[]
        for ep in X[:n]:
            r=[]
            for c in range(N_CHANNELS):
                psd=np.abs(np.fft.rfft(ep[c]))**2
                r.append(psd[(freqs>=4)&(freqs<8)].mean()/(psd[(freqs>=8)&(freqs<13)].mean()+1e-10))
            ratios.append(np.mean(r))
        return np.array(ratios)
    def feat(X,n=150):
        return np.array([[np.log1p(bp(X,lo,hi,n).mean()) for _,(lo,hi) in BANDS.items()] for _ in [None]*min(n,len(X))],dtype=np.float32)[:min(n,len(X))]
    def frechet(fr,fs):
        mr,ms=fr.mean(0),fs.mean(0); reg=np.eye(5)*1e-4
        cr=np.cov(fr,rowvar=False)+reg; cs=np.cov(fs,rowvar=False)+reg
        cm=sqrtm(cr@cs)
        if np.iscomplexobj(cm): cm=cm.real
        return max(float(np.real((mr-ms)@(mr-ms)+np.trace(cr+cs-2*cm))),0.)
    tar_r,tar_s=tar(X_real_ad),tar(X_synth)
    fr,fs=feat(X_real_ad),feat(X_synth)
    fd=frechet(fr,fs) if len(fr)>5 and len(fs)>5 else float('inf')
    log.info("  PSD comparison (real AD vs synthetic):")
    for band,(lo,hi) in BANDS.items():
        r=bp(X_real_ad,lo,hi).mean(); s=bp(X_synth,lo,hi).mean()
        err=100*(s-r)/(abs(r)+1e-8); flag="  ← HIGH" if abs(err)>20 else ""
        log.info(f"    {band:5s}: real={r:.3f}  synth={s:.3f}  err={err:+.1f}%{flag}")
    return {'frechet':fd,'tar_bias':float(100*(tar_s.mean()-tar_r.mean())/(abs(tar_r.mean())+1e-8)),'ad_direction':bool(tar_s.mean()>tar_r.mean())}

def evaluate_with_fn_profile(model, X_te, y_te):
    y_pred,y_prob=predict_probs(model,X_te)
    m=compute_epoch_metrics(y_te,y_pred,y_prob)
    if m['fn']>0:
        fn_idx=np.where((y_pred==0)&(y_te==1))[0]; tp_idx=np.where((y_pred==1)&(y_te==1))[0]
        pct=100*m['fn']/max(m['fn']+m['tp'],1)
        log.warning(f"  [FN] {m['fn']} AD epochs misclassified ({pct:.1f}%)")
        freqs=np.fft.rfftfreq(N_SAMPLES,d=1./SFREQ)
        def bpw(idx,lo,hi):
            mask=(freqs>=lo)&(freqs<hi)
            vals=[np.log1p((np.abs(np.fft.rfft(X_te[i],axis=-1)).mean(0)**2)[mask].mean()) for i in idx[:100]]
            return float(np.mean(vals)) if vals else float('nan')
        fnt=bpw(fn_idx,4,8); fna=bpw(fn_idx,8,13); fnd=bpw(fn_idx,.5,4); fn_ta=fnt/(fna+1e-8)
        log.info(f"   ├─ Missed: delta={fnd:.3f} theta={fnt:.3f} alpha={fna:.3f} θ/α={fn_ta:.3f}")
        if len(tp_idx)>0:
            tpt=bpw(tp_idx,4,8); tpa=bpw(tp_idx,8,13); tpd=bpw(tp_idx,.5,4); tp_ta=tpt/(tpa+1e-8)
            log.info(f"   ├─ Caught: delta={tpd:.3f} theta={tpt:.3f} alpha={tpa:.3f} θ/α={tp_ta:.3f}")
            if fn_ta<tp_ta*0.85: log.info("   └─ WHY: Weaker EEG slowing (mild/early-stage AD)")
            elif fnd<tpd*0.85:   log.info("   └─ WHY: Low delta (atypical subtype or medication)")
            else:                 log.info("   └─ WHY: No dominant spectral difference — spatial failure")
    return m, y_pred, y_prob

# =============================================================================
# CELL 14 — THESIS PLOTS
# =============================================================================
def save_all_plots(fold_results, all_roc_data, global_baseline_results=None):
    """Generate all thesis-quality figures and save to PLOTS_DIR."""

    folds  = [r['fold'] for r in fold_results]
    n_folds= len(fold_results)

    # ── 1. Per-fold metric bar chart ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics   = ['auc','sensitivity','specificity','f1_macro']
    titles    = ['AUC-ROC (%)','Sensitivity (%)','Specificity (%)','F1 Macro (%)']
    x         = np.arange(n_folds); w=0.35
    for ax, m, t in zip(axes.flat, metrics, titles):
        bl  = [r['baseline_epoch'][m]  for r in fold_results]
        aug = [r['augmented_epoch'][m] for r in fold_results]
        b1  = ax.bar(x-w/2, bl,  w, label='Baseline',  color='steelblue',     alpha=0.82)
        b2  = ax.bar(x+w/2, aug, w, label='Augmented', color='mediumseagreen', alpha=0.82)
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+w/2, bar.get_height()+0.5, f'{bar.get_height():.1f}',
                    ha='center', fontsize=8)
        ax.set_title(t); ax.set_xticks(x); ax.set_xticklabels([f'F{f}' for f in folds])
        ax.set_ylim(0,115); ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
    plt.suptitle('Epoch-Level Metrics — Baseline vs Augmented per Fold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'fold_metrics_epoch.png'), dpi=150)
    plt.close(); log.info("  Plot: fold_metrics_epoch.png")

    # ── 2. Patient-level metrics bar chart ────────────────────────────────────
    if 'baseline_patient' in fold_results[0]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, m, t in zip(axes.flat, metrics, titles):
            bl  = [r['baseline_patient'][m]  for r in fold_results]
            aug = [r['augmented_patient'][m] for r in fold_results]
            b1  = ax.bar(x-w/2, bl,  w, label='Baseline',  color='steelblue',     alpha=0.82)
            b2  = ax.bar(x+w/2, aug, w, label='Augmented', color='mediumseagreen', alpha=0.82)
            for bar in list(b1)+list(b2):
                ax.text(bar.get_x()+w/2, bar.get_height()+0.5, f'{bar.get_height():.1f}',
                        ha='center', fontsize=8)
            ax.set_title(t); ax.set_xticks(x); ax.set_xticklabels([f'F{f}' for f in folds])
            ax.set_ylim(0,115); ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
        plt.suptitle('PATIENT-Level Metrics (Soft Voting) — Baseline vs Augmented', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,'fold_metrics_patient.png'), dpi=150)
        plt.close(); log.info("  Plot: fold_metrics_patient.png")

    # ── 3. ΔAUC per fold ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    deltas  = [r['delta_auc_epoch'] for r in fold_results]
    colors  = ['mediumseagreen' if d>=0 else 'tomato' for d in deltas]
    ax.bar(folds, deltas, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(np.mean(deltas), color='steelblue', linestyle='--', linewidth=1.5,
               label=f'Mean ΔAUC = {np.mean(deltas):+.2f}pp')
    for f,d in zip(folds,deltas):
        ax.text(f, d+np.sign(d)*0.3, f'{d:+.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Fold'); ax.set_ylabel('ΔAUC (pp)'); ax.legend()
    ax.set_title('ΔAUC per Fold (Augmented − Baseline)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'delta_auc_per_fold.png'), dpi=150)
    plt.close(); log.info("  Plot: delta_auc_per_fold.png")

    # ── 4. ROC curves (all folds + mean) ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mean_fpr  = np.linspace(0, 1, 100)
    for ax, key, title in zip(axes, ['baseline','augmented'], ['Baseline','Augmented']):
        interp_tprs = []
        for fd_data in all_roc_data:
            if key not in fd_data: continue
            fpr,tpr,_ = roc_curve(fd_data[key]['y_true'], fd_data[key]['y_prob'])
            auc_v     = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, alpha=0.4, linewidth=1.2,
                    label=f"Fold {fd_data['fold']} (AUC={auc_v:.3f})")
            interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
        if interp_tprs:
            mean_tpr = np.mean(interp_tprs, axis=0)
            std_tpr  = np.std(interp_tprs, axis=0)
            ax.plot(mean_fpr, mean_tpr, 'k--', lw=2,
                    label=f'Mean AUC={sklearn_auc(mean_fpr,mean_tpr):.3f}')
            ax.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr,
                            alpha=0.15, color='grey', label='±1 std')
        ax.plot([0,1],[0,1],'--',color='grey',lw=0.8)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves — {title}'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle('Receiver Operating Characteristic — Epoch Level', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'roc_curves.png'), dpi=150)
    plt.close(); log.info("  Plot: roc_curves.png")

    # ── 5. Patient-level confusion matrix (aggregated across folds) ───────────
    if 'baseline_patient' in fold_results[0]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, title in zip(axes, ['baseline_patient','augmented_patient'],
                                   ['Baseline (patient)','Augmented (patient)']):
            tp_=sum(r[key]['tp'] for r in fold_results)
            tn_=sum(r[key]['tn'] for r in fold_results)
            fp_=sum(r[key]['fp'] for r in fold_results)
            fn_=sum(r[key]['fn'] for r in fold_results)
            cm_mat = np.array([[tn_,fp_],[fn_,tp_]])
            im=ax.imshow(cm_mat, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm_mat[i,j]),ha='center',va='center',
                            fontsize=16,fontweight='bold',
                            color='white' if cm_mat[i,j]>cm_mat.max()/2 else 'black')
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(['Pred HC','Pred AD'])
            ax.set_yticklabels(['True HC','True AD'])
            ax.set_title(f'{title}\n(aggregated, N patients per fold)')
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,'confusion_matrix_patient.png'), dpi=150)
        plt.close(); log.info("  Plot: confusion_matrix_patient.png")

    # ── 6. Generation quality across folds ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    gen_fds    = [r['generation']['frechet']    for r in fold_results]
    gen_biases = [abs(r['generation']['tar_bias_after']) for r in fold_results]
    n_synths   = [r['n_synth']                for r in fold_results]

    axes[0].bar(folds, gen_fds,    color='steelblue', alpha=0.8)
    axes[0].axhline(FD_THRESHOLD,  color='red', linestyle='--', label=f'Gate={FD_THRESHOLD}')
    axes[0].set_title('Fréchet Distance per Fold'); axes[0].set_xlabel('Fold')
    axes[0].legend(); axes[0].grid(axis='y',alpha=0.3)

    axes[1].bar(folds, gen_biases, color='mediumseagreen', alpha=0.8)
    axes[1].axhline(20, color='orange', linestyle='--', label='20% flag')
    axes[1].set_title('|θ/α Bias| After Correction'); axes[1].set_xlabel('Fold')
    axes[1].legend(); axes[1].grid(axis='y',alpha=0.3)

    axes[2].bar(folds, n_synths,   color='mediumpurple', alpha=0.8)
    n_hcs = [r['n_hc_train'] for r in fold_results]
    axes[2].plot(folds, n_hcs, 'r--o', label='HC train count')
    axes[2].set_title('Synthetic AD epochs vs HC count'); axes[2].set_xlabel('Fold')
    axes[2].legend(); axes[2].grid(axis='y',alpha=0.3)

    plt.suptitle('Generation Quality per Fold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'generation_quality.png'), dpi=150)
    plt.close(); log.info("  Plot: generation_quality.png")

    # ── 7. Summary bar (mean ± std across folds) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    keys_order = ['auc','sensitivity','specificity','f1_macro']
    labels_ord = ['AUC-ROC','Sensitivity','Specificity','F1 Macro']
    x   = np.arange(len(keys_order)); w = 0.35
    bl_means  = [np.nanmean([r['baseline_epoch'][k]  for r in fold_results]) for k in keys_order]
    aug_means = [np.nanmean([r['augmented_epoch'][k] for r in fold_results]) for k in keys_order]
    bl_stds   = [np.nanstd([r['baseline_epoch'][k]   for r in fold_results]) for k in keys_order]
    aug_stds  = [np.nanstd([r['augmented_epoch'][k]  for r in fold_results]) for k in keys_order]
    b1 = ax.bar(x-w/2, bl_means,  w, yerr=bl_stds,  capsize=4, label='Baseline',  color='steelblue',     alpha=0.85)
    b2 = ax.bar(x+w/2, aug_means, w, yerr=aug_stds, capsize=4, label='Augmented', color='mediumseagreen', alpha=0.85)
    for bar,mean in zip(list(b1)+list(b2), bl_means+aug_means):
        ax.text(bar.get_x()+w/2, bar.get_height()+1.5, f'{mean:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels_ord); ax.set_ylim(0,115)
    ax.set_ylabel('Score (%)'); ax.set_title('Mean ± Std across 5 Folds (Epoch Level)')
    ax.legend(); ax.grid(axis='y',alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'summary_mean_std.png'), dpi=150)
    plt.close(); log.info("  Plot: summary_mean_std.png")

    # ── 8. 5-Fold averaged summary — the KEY table for your thesis ────────────
    # This is the "overall result" your supervisor wants to see.
    # Shows mean ± std for every metric across all 5 folds at both
    # epoch level and patient level, with ΔAUC and significance.
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis('off')

    rows = []
    col_labels = ['Metric', 'Level',
                  'Baseline\nMean ± Std', 'Augmented\nMean ± Std',
                  'Δ (pp)', 'Best Fold', 'Worst Fold']

    for m, label in zip(['auc','sensitivity','specificity','f1_macro'],
                        ['AUC-ROC (%)','Sensitivity (%)','Specificity (%)','F1 Macro (%)']):
        for level, key_bl, key_aug in [
            ('Epoch',   'baseline_epoch',   'augmented_epoch'),
            ('Patient', 'baseline_patient', 'augmented_patient'),
        ]:
            bl_v  = [r[key_bl][m]  for r in fold_results]
            aug_v = [r[key_aug][m] for r in fold_results]
            delta = np.nanmean(aug_v) - np.nanmean(bl_v)
            delta_per_fold = [a-b for a,b in zip(aug_v, bl_v)]
            best_fold  = folds[int(np.argmax(delta_per_fold))]
            worst_fold = folds[int(np.argmin(delta_per_fold))]
            rows.append([
                label if level=='Epoch' else '',
                level,
                f"{np.nanmean(bl_v):.1f} ± {np.nanstd(bl_v):.1f}",
                f"{np.nanmean(aug_v):.1f} ± {np.nanstd(aug_v):.1f}",
                f"{delta:+.1f}",
                f"F{best_fold}",
                f"F{worst_fold}",
            ])

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Colour Δ column: green if positive, red if negative
    for row_idx, row in enumerate(rows):
        delta_val = float(row[4])
        colour    = '#d4edda' if delta_val >= 0 else '#f8d7da'
        tbl[row_idx+1, 4].set_facecolor(colour)

    # Header row styling
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor('#2c3e50')
        tbl[0, col].set_text_props(color='white', fontweight='bold')

    ax.set_title('5-Fold Cross-Validation Averaged Results (Mean ± Std)',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'summary_table_5fold.png'), dpi=160, bbox_inches='tight')
    plt.close(); log.info("  Plot: summary_table_5fold.png")

    # ── 9. Training loss curves — warmup and joint per fold ──────────────────
    # These are collected from fold_results['warmup_history'] and ['joint_history']
    if 'warmup_history' in fold_results[0] and fold_results[0]['warmup_history']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax_wu, ax_jt = axes

        for r in fold_results:
            f = r['fold']
            if r.get('warmup_history'):
                ax_wu.plot(r['warmup_history'], alpha=0.7, linewidth=1.4, label=f'Fold {f}')
            if r.get('joint_diff_history'):
                ax_jt.plot(r['joint_diff_history'], alpha=0.7, linewidth=1.4, label=f'Fold {f}')

        ax_wu.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Target <0.05')
        ax_wu.set_title('Encoder Warm-up Loss per Fold')
        ax_wu.set_xlabel('Epoch'); ax_wu.set_ylabel('MSE Reconstruction Loss')
        ax_wu.legend(fontsize=8); ax_wu.grid(alpha=0.3)

        ax_jt.set_title('Joint Diffusion Loss per Fold (diff component)')
        ax_jt.set_xlabel('Epoch'); ax_jt.set_ylabel('DDPM Noise Prediction Loss')
        ax_jt.legend(fontsize=8); ax_jt.grid(alpha=0.3)

        plt.suptitle('VMoGE Training Convergence', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,'training_loss_curves.png'), dpi=150)
        plt.close(); log.info("  Plot: training_loss_curves.png")
    else:
        log.info("  (training_loss_curves skipped — history not stored in this run)")

    # ── 10. PSD comparison real AD vs synthetic — averaged across folds ───────
    if any('psd_bands_real' in r.get('generation',{}) for r in fold_results):
        bands = ['delta','theta','alpha','beta','gamma']
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(bands)); w = 0.35

        real_means = []
        synth_means= []
        real_stds  = []
        synth_stds = []
        for band in bands:
            rv = [r['generation']['psd_bands_real'].get(band, float('nan')) for r in fold_results]
            sv = [r['generation']['psd_bands_synth'].get(band, float('nan')) for r in fold_results]
            real_means.append(np.nanmean(rv));  real_stds.append(np.nanstd(rv))
            synth_means.append(np.nanmean(sv)); synth_stds.append(np.nanstd(sv))

        b1 = ax.bar(x-w/2, real_means,  w, yerr=real_stds,  capsize=4,
                    label='Real AD',   color='steelblue',     alpha=0.85)
        b2 = ax.bar(x+w/2, synth_means, w, yerr=synth_stds, capsize=4,
                    label='Synthetic', color='mediumseagreen', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([b.capitalize() for b in bands])
        ax.set_ylabel('Mean Log Band Power'); ax.set_title('PSD Comparison: Real AD vs Synthetic (Mean ± Std, 5 Folds)')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,'psd_comparison.png'), dpi=150)
        plt.close(); log.info("  Plot: psd_comparison.png")
    else:
        log.info("  (psd_comparison skipped — band-level PSD not stored per fold)")

    # ── 11. Radar / spider chart — mean metrics at a glance ──────────────────
    metrics_radar  = ['auc','sensitivity','specificity','f1_macro']
    labels_radar   = ['AUC-ROC','Sensitivity','Specificity','F1 Macro']
    bl_vals_radar  = [np.nanmean([r['baseline_epoch'][m]  for r in fold_results]) for m in metrics_radar]
    aug_vals_radar = [np.nanmean([r['augmented_epoch'][m] for r in fold_results]) for m in metrics_radar]

    N    = len(metrics_radar)
    angles = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]
    bl_vals_radar  += bl_vals_radar[:1]
    aug_vals_radar += aug_vals_radar[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_radar, fontsize=11)
    ax.set_ylim(0,100)
    ax.plot(angles, bl_vals_radar,  color='steelblue',     linewidth=2, label='Baseline')
    ax.fill(angles, bl_vals_radar,  color='steelblue',     alpha=0.15)
    ax.plot(angles, aug_vals_radar, color='mediumseagreen', linewidth=2, label='Augmented')
    ax.fill(angles, aug_vals_radar, color='mediumseagreen', alpha=0.15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15))
    ax.set_title('Mean Performance Radar\n(5-Fold Average)', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'radar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close(); log.info("  Plot: radar_chart.png")

    # ── 12. Specificity–Sensitivity trade-off scatter per fold ────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in fold_results:
        f = r['fold']
        ax.scatter(r['baseline_epoch']['specificity'],
                   r['baseline_epoch']['sensitivity'],
                   marker='o', s=90, color='steelblue', alpha=0.8, zorder=3)
        ax.scatter(r['augmented_epoch']['specificity'],
                   r['augmented_epoch']['sensitivity'],
                   marker='*', s=130, color='mediumseagreen', alpha=0.8, zorder=3)
        ax.annotate(f'F{f}',
                    xy=(r['augmented_epoch']['specificity'],r['augmented_epoch']['sensitivity']),
                    xytext=(3,3), textcoords='offset points', fontsize=9)
        ax.plot([r['baseline_epoch']['specificity'],  r['augmented_epoch']['specificity']],
                [r['baseline_epoch']['sensitivity'],   r['augmented_epoch']['sensitivity']],
                color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0],marker='o',color='w',markerfacecolor='steelblue',    markersize=9,label='Baseline'),
        Line2D([0],[0],marker='*',color='w',markerfacecolor='mediumseagreen',markersize=12,label='Augmented'),
    ]
    ax.legend(handles=legend_els)
    ax.set_xlabel('Specificity (%)', fontsize=11)
    ax.set_ylabel('Sensitivity (%)', fontsize=11)
    ax.set_title('Sensitivity–Specificity Trade-off per Fold\n(arrows show baseline → augmented)', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,'sensitivity_specificity_tradeoff.png'), dpi=150)
    plt.close(); log.info("  Plot: sensitivity_specificity_tradeoff.png")

    # ── Print the 5-fold averaged table to log ───────────────────────────────
    log.info("\n" + "="*70)
    log.info("  5-FOLD AVERAGED RESULTS SUMMARY")
    log.info("="*70)
    log.info(f"  {'Metric':<18} {'Level':<8} {'Baseline':>16} {'Augmented':>16} {'Δ':>8}")
    log.info("  " + "-"*66)
    for m, label in zip(['auc','sensitivity','specificity','f1_macro'],
                        ['AUC-ROC','Sensitivity','Specificity','F1 Macro']):
        for level, key_bl, key_aug in [('Epoch',   'baseline_epoch',   'augmented_epoch'),
                                        ('Patient', 'baseline_patient', 'augmented_patient')]:
            bl_v  = [r[key_bl][m]  for r in fold_results]
            aug_v = [r[key_aug][m] for r in fold_results]
            delta = np.nanmean(aug_v) - np.nanmean(bl_v)
            log.info(f"  {label:<18} {level:<8} "
                     f"{np.nanmean(bl_v):>6.2f}±{np.nanstd(bl_v):.2f}%"
                     f"   {np.nanmean(aug_v):>6.2f}±{np.nanstd(aug_v):.2f}%"
                     f"   {delta:>+6.2f}pp")
    log.info("="*70)

    # Save averaged results as JSON (easy to load later for analysis)
    avg_results = {}
    for m in ['auc','sensitivity','specificity','f1_macro']:
        avg_results[m] = {
            'baseline_epoch_mean':    float(np.nanmean([r['baseline_epoch'][m]   for r in fold_results])),
            'baseline_epoch_std':     float(np.nanstd([r['baseline_epoch'][m]    for r in fold_results])),
            'augmented_epoch_mean':   float(np.nanmean([r['augmented_epoch'][m]  for r in fold_results])),
            'augmented_epoch_std':    float(np.nanstd([r['augmented_epoch'][m]   for r in fold_results])),
            'baseline_patient_mean':  float(np.nanmean([r['baseline_patient'][m] for r in fold_results])),
            'baseline_patient_std':   float(np.nanstd([r['baseline_patient'][m]  for r in fold_results])),
            'augmented_patient_mean': float(np.nanmean([r['augmented_patient'][m]for r in fold_results])),
            'augmented_patient_std':  float(np.nanstd([r['augmented_patient'][m] for r in fold_results])),
        }
    avg_results['delta_auc_epoch_mean']   = float(np.nanmean([r['delta_auc_epoch']   for r in fold_results]))
    avg_results['delta_auc_patient_mean'] = float(np.nanmean([r['delta_auc_patient'] for r in fold_results]))
    with open(os.path.join(RESULTS_DIR, 'averaged_5fold_results.json'), 'w') as f:
        json.dump(avg_results, f, indent=2)
    log.info(f"  Averaged results saved → {os.path.join(RESULTS_DIR,'averaged_5fold_results.json')}")

    log.info(f"\n  All plots saved to: {PLOTS_DIR}")

# =============================================================================
# CELL 15 —  GLOBAL BASELINE (trained once on all subjects)
# =============================================================================
def train_global_baseline(X_all, y_all, sub_all):
    """
    [S4] Train one EEGNet on ALL subjects using a random 80/20 subject split.
    This gives a stable, fold-independent baseline.
    It is methodologically separate from the CV evaluation —
    we report it alongside but do not use it as the CV comparison point.
    """
    log.info("\n[GLOBAL BASELINE] Training one EEGNet on all subjects...")
    unique_subs = np.unique(sub_all)
    sub_labels  = np.array([y_all[sub_all==s][0] for s in unique_subs])
    rng         = np.random.default_rng(RANDOM_STATE)
    perm        = rng.permutation(len(unique_subs))
    n_val       = max(2, int(len(unique_subs)*0.20))
    val_subs    = unique_subs[perm[:n_val]]
    tr_subs     = unique_subs[perm[n_val:]]
    def get(subs): mask=np.isin(sub_all,subs); return X_all[mask], y_all[mask], sub_all[mask]
    X_tr,y_tr,_ = get(tr_subs)
    X_va,y_va,_ = get(val_subs)
    model = train_eegnet(X_tr, y_tr, X_va, y_va, seed=RANDOM_STATE)
    log.info("[GLOBAL BASELINE] Trained on all subjects — use for reference only.")
    return model

# =============================================================================
# CELL 16 — MAIN K-FOLD PIPELINE
# =============================================================================
def run_pipeline():
    log.info(" Loading pre-saved numpy partitions...")
    def _load(fname):
        path=os.path.join(SAVE_DIR,fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}\nAvailable: {sorted(os.listdir(SAVE_DIR))}")
        return np.load(path, allow_pickle=True)

    X_all  = np.concatenate([_load('X_train.npy'), _load('X_test.npy')]).astype(np.float32)
    y_all  = np.concatenate([_load('y_train.npy'), _load('y_test.npy')]).astype(np.int64)
    sub_raw= np.concatenate([_load('train_subject_ids.npy'), _load('test_subject_ids.npy')])
    sub_all= np.array([int(str(s).replace('sub-','')) for s in sub_raw], dtype=np.int64) \
             if sub_raw.dtype.kind in {'U','S','O'} else sub_raw.astype(np.int64)

    log.info(f"Data: {X_all.shape} | HC={(y_all==0).sum()} | AD={(y_all==1).sum()} | Subjects={len(np.unique(sub_all))}")

    # [S4] Global baseline
    global_model = train_global_baseline(X_all, y_all, sub_all)

    unique_subs = np.unique(sub_all)
    sub_labels  = np.array([y_all[sub_all==s][0] for s in unique_subs])
    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    all_roc_data = []

    for fold_idx, (tv_idx, te_idx) in enumerate(skf.split(unique_subs, sub_labels)):
        if fold_idx >= NUM_FOLDS: break
        fold = fold_idx+1
        log.info(f"\n{'='*70}")
        log.info(f"  FOLD {fold} / {NUM_FOLDS}")
        log.info(f"{'='*70}")

        tv_subs  = unique_subs[tv_idx]; te_subs = unique_subs[te_idx]
        rng      = np.random.default_rng(fold_idx*7+RANDOM_STATE)
        perm     = rng.permutation(len(tv_subs))
        n_val    = max(2, int(len(tv_subs)*0.20))
        val_subs = tv_subs[perm[:n_val]]; tr_subs = tv_subs[perm[n_val:]]

        def get(subs):
            mask=np.isin(sub_all,subs); return X_all[mask],y_all[mask],sub_all[mask]
        X_tr,y_tr,sub_tr   = get(tr_subs)
        X_val,y_val,_       = get(val_subs)
        X_te,y_te,sub_te   = get(te_subs)
        X_ad_tr             = X_tr[y_tr==1]
        n_hc_tr             = int((y_tr==0).sum())
        n_ad_tr             = int((y_tr==1).sum())

        log.info(f"  Train: {len(X_tr)} ({n_hc_tr} HC, {n_ad_tr} AD) | Val: {len(X_val)} | Test: {len(X_te)}")

        # ── Stage 1a: Baseline EEGNet ─────────────────────────────────────────
        log.info("\n  [1a] Baseline EEGNet...")
        model_bl = train_eegnet(X_tr, y_tr, X_val, y_val, seed=fold_idx*13+42)
        m_bl_ep, y_pred_bl, y_prob_bl = evaluate_with_fn_profile(model_bl, X_te, y_te)
        m_bl_pt, _ = patient_level_evaluation(model_bl, X_te, y_te, sub_te, "Baseline")
        log.info(f"  Baseline epoch → AUC={m_bl_ep['auc']:.2f}% Sens={m_bl_ep['sensitivity']:.2f}% Spec={m_bl_ep['specificity']:.2f}%")

        # ── Stage 2a: Train VMoGE ─────────────────────────────────────────────
        log.info("\n  [2a] Training VMoGE...")
        (encoder,diff_dec,full_mean,full_std,
         Z_ad,sigma_ad,
         warmup_hist,joint_hist) = train_vmoge(
            X_ad_fold=X_ad_tr, X_all_fold=X_tr, y_all_fold=y_tr, seed=fold_idx*17+42)

        # ── Stage 3a: Quality gate ────────────────────────────────────────────
        log.info("\n  [3a] Quality gate...")
        n_probe  = min(60, n_ad_tr)
        X_probe  = generate_synthetic_ad(encoder,diff_dec,full_mean,full_std,Z_ad,sigma_ad,n_probe)
        gen_q    = compute_spectral_metrics(X_ad_tr[:n_probe], X_probe)
        fd_val   = gen_q['frechet']
        log.info(f"  FD={fd_val:.4f} (gate={FD_THRESHOLD}) θ/α={gen_q['tar_bias']:+.1f}%")

        m_aug_ep = dict(m_bl_ep); m_aug_pt = dict(m_bl_pt)
        skip=None; n_synth=0
        gen_q_post = gen_q

        if fd_val > FD_THRESHOLD or math.isnan(fd_val):
            skip=f"FD={fd_val:.3f}>{FD_THRESHOLD}"
            log.warning(f"  [GATE FAILED] {skip}")
        else:
            # ── [S1] Generate to match HC count ──────────────────────────────
            n_synth = max(0, n_hc_tr - n_ad_tr)
            log.info(f"\n  [3b] Generating {n_synth} synthetic AD epochs "
                     f"(to match {n_hc_tr} HC | was capped at 800 before)...")
            X_synth = generate_synthetic_ad(encoder,diff_dec,full_mean,full_std,Z_ad,sigma_ad,n_synth)

            if APPLY_SPECTRAL_CORRECTION:
                log.info("  [Spectral correction]...")
                X_synth, _ = spectral_bias_correction(X_synth, X_ad_tr)
                gen_q_post  = compute_spectral_metrics(X_ad_tr[:100], X_synth[:100])
                log.info(f"  θ/α after correction: {gen_q_post['tar_bias']:+.1f}% (was {gen_q['tar_bias']:+.1f}%)")
                gen_q_post['frechet'] = fd_val   # FD measured before correction

            # ── [S2] Save synthetic epochs ────────────────────────────────────
            synth_x_path = os.path.join(SYNTH_DIR, f'synth_fold{fold}_X.npy')
            synth_y_path = os.path.join(SYNTH_DIR, f'synth_fold{fold}_y.npy')
            np.save(synth_x_path, X_synth)
            np.save(synth_y_path, np.ones(len(X_synth), dtype=np.int64))
            log.info(f"  [S2] Saved synthetic epochs → {synth_x_path}")

            # ── Stage 1b: Augmented EEGNet ────────────────────────────────────
            X_aug = np.concatenate([X_tr, X_synth])
            y_aug = np.concatenate([y_tr, np.ones(n_synth, dtype=np.int64)])
            log.info(f"\n  [1b] Augmented EEGNet (HC={( y_aug==0).sum()} AD={(y_aug==1).sum()})...")
            model_aug = train_eegnet(X_aug, y_aug, X_val, y_val, seed=fold_idx*23+42)
            m_aug_ep, y_pred_aug, y_prob_aug = evaluate_with_fn_profile(model_aug, X_te, y_te)
            m_aug_pt, _ = patient_level_evaluation(model_aug, X_te, y_te, sub_te, "Augmented")
            log.info(f"  Augmented epoch → AUC={m_aug_ep['auc']:.2f}% Sens={m_aug_ep['sensitivity']:.2f}% Spec={m_aug_ep['specificity']:.2f}%")

        # Store ROC data for plotting
        roc_entry = {'fold': fold}
        roc_entry['baseline']  = {'y_true': y_te, 'y_prob': y_prob_bl}
        if not skip: roc_entry['augmented'] = {'y_true': y_te, 'y_prob': y_prob_aug}
        all_roc_data.append(roc_entry)

        delta_ep = m_aug_ep['auc'] - m_bl_ep['auc']
        delta_pt = m_aug_pt['auc'] - m_bl_pt['auc']
        log.info(f"\n  Fold {fold} | ΔAUC epoch={delta_ep:+.2f}pp patient={delta_pt:+.2f}pp | FD={fd_val:.4f}")

        fold_result = {
            'fold':              fold,
            'fd':                float(fd_val),
            'n_hc_train':        n_hc_tr,
            'n_ad_train':        n_ad_tr,
            'n_synth':           n_synth,
            'baseline_epoch':    m_bl_ep,
            'augmented_epoch':   m_aug_ep,
            'baseline_patient':  m_bl_pt,
            'augmented_patient': m_aug_pt,
            'delta_auc_epoch':   float(delta_ep),
            'delta_auc_patient': float(delta_pt),
            'generation':        {**gen_q_post, 'tar_bias_after': gen_q_post.get('tar_bias', 0)},
            'skip_reason':       skip,
            'warmup_history':    warmup_hist,
            'joint_diff_history':joint_hist,
        }
        fold_results.append(fold_result)

        out_path = os.path.join(RESULTS_DIR, 'fold_results_v7.json')
        with open(out_path, 'w') as fp:
            json.dump(fold_results, fp, indent=2, default=str)
        log.info(f"  Saved → {out_path}")

        del encoder, diff_dec, model_bl
        if not skip: del model_aug
        if DEVICE.type=='cuda': torch.cuda.empty_cache()

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info(f"  FINAL SUMMARY ({len(fold_results)} folds)")
    log.info(f"{'='*70}")
    log.info("  EPOCH LEVEL:")
    for m in ['auc','sensitivity','specificity','f1_macro']:
        bl  = [r['baseline_epoch'][m]  for r in fold_results]
        aug = [r['augmented_epoch'][m] for r in fold_results]
        log.info(f"    {m.upper():12s} | BL={np.nanmean(bl):.2f}±{np.nanstd(bl):.2f} "
                 f"| AUG={np.nanmean(aug):.2f}±{np.nanstd(aug):.2f} "
                 f"| Δ={np.nanmean(aug)-np.nanmean(bl):+.2f}pp")
    log.info("  PATIENT LEVEL (soft voting):")
    for m in ['auc','sensitivity','specificity','f1_macro']:
        bl  = [r['baseline_patient'][m]  for r in fold_results]
        aug = [r['augmented_patient'][m] for r in fold_results]
        log.info(f"    {m.upper():12s} | BL={np.nanmean(bl):.2f}±{np.nanstd(bl):.2f} "
                 f"| AUG={np.nanmean(aug):.2f}±{np.nanstd(aug):.2f} "
                 f"| Δ={np.nanmean(aug)-np.nanmean(bl):+.2f}pp")

    # ── Save all thesis plots ─────────────────────────────────────────────────
    log.info("\nGenerating thesis plots...")
    save_all_plots(fold_results, all_roc_data)

    return fold_results

# =============================================================================
# CELL 17 — RUN
# =============================================================================
if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    results = run_pipeline()
    log.info("Pipeline complete.")
