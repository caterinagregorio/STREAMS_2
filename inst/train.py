import argparse
import math
import os

import numpy as np
import pyarrow.feather as feather
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True

# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class VAEDataset(Dataset):
    """
    Feather must have:
      a, b, onset_soft (NaN if unlabeled), onset_prior, patient_id, covariates
    """

    def __init__(self, path, covariate_cols):
        df = feather.read_feather(path)

        for c in ["a", "b", "onset_soft", "onset_prior", "patient_id"]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {path}")

        self.a = torch.tensor(df["a"].values, dtype=torch.float32).unsqueeze(1)
        self.b = torch.tensor(df["b"].values, dtype=torch.float32).unsqueeze(1)

        y = df["onset_soft"].values
        lab = ~np.isnan(y)
        self.lab = torch.tensor(lab.astype(np.float32)).unsqueeze(1)
        y_fill = np.where(lab, y.astype("float32"), 0.0)
        self.y = torch.tensor(y_fill, dtype=torch.float32).unsqueeze(1)

        prior = (
            df["onset_prior"]
            .fillna(0.5)
            .clip(1e-6, 1 - 1e-6)
            .astype("float32")
            .values
        )
        self.x_prior = torch.tensor(prior, dtype=torch.float32).unsqueeze(1)

        if covariate_cols:
            cond = df[covariate_cols].astype("float32").values
        else:
            cond = np.zeros((len(df), 0), dtype="float32")
        self.cond = torch.tensor(cond, dtype=torch.float32)

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, i):
        return (
            self.x_prior[i],
            self.cond[i],
            self.y[i],
            self.a[i],
            self.b[i],
            self.lab[i],
        )


# -----------------------------
# Model
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_lv = nn.Linear(64, latent_dim)

    def forward(self, x_prior, cond):
        h = torch.cat([x_prior, cond], dim=1)
        h = self.net(h)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.fc_p = nn.Linear(64, 1)
        self.fc_mu = nn.Linear(64, 1)
        self.fc_sd = nn.Linear(64, 1)

    def forward(self, z, cond, a, b):
        h = torch.cat([z, cond], dim=1)
        h = self.net(h)
        p_onset = torch.sigmoid(self.fc_p(h))
        age_mu = a + (b - a) * torch.sigmoid(self.fc_mu(h))
        age_sd = (b - a) * F.softplus(self.fc_sd(h)) + 1e-6
        return p_onset, age_mu, age_sd


class CVAE(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim):
        super().__init__()
        self.enc = Encoder(x_dim, cond_dim, latent_dim)
        self.dec = Decoder(latent_dim, cond_dim)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_prior, cond, a, b):
        mu, lv = self.enc(x_prior, cond)
        z = self.reparam(mu, lv)
        p, mu_age, sd_age = self.dec(z, cond, a, b)
        return p, mu_age, sd_age, mu, lv


# -----------------------------
# Prior augmentation (shared)
# -----------------------------
def prior_sigma_from_p(p, base=0.01, extra=0.06):
    # Uncertainty-shaped noise; max at p=0.5
    return base + extra * (4.0 * p * (1.0 - p))


def aug_prior_gauss(x, base=0.02, extra=0.08):
    sigma = prior_sigma_from_p(x, base=base, extra=extra)
    return (x + sigma * torch.randn_like(x)).clamp(1e-6, 1 - 1e-6)


def aug_prior_dropout(x, p=0.1):
    if p <= 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return (mask * x + (1 - mask) * 0.5).clamp(1e-6, 1 - 1e-6)


def aug_prior_beta(x, kappa=30.0):
    # Treat prior as mean of Beta(α=κp, β=κ(1-p))
    p = x.clamp(1e-6, 1 - 1e-6)
    a = (kappa * p).clamp_min(1e-3)
    b = (kappa * (1 - p)).clamp_min(1e-3)
    g1 = torch.distributions.Gamma(a, torch.ones_like(a)).sample()
    g2 = torch.distributions.Gamma(b, torch.ones_like(b)).sample()
    return (g1 / (g1 + g2)).clamp(1e-6, 1 - 1e-6)


def apply_prior_aug(x, mode, **kw):
    if mode == "none":
        return x
    if mode == "gauss":
        return aug_prior_gauss(
            x,
            base=kw.get("prior_sigma_base", 0.02),
            extra=kw.get("prior_sigma_extra", 0.08),
        )
    if mode == "dropout":
        return aug_prior_dropout(x, p=kw.get("prior_dropout_p", 0.1))
    if mode == "beta":
        return aug_prior_beta(x, kappa=kw.get("prior_kappa", 30.0))
    raise ValueError(f"Unknown prior_aug mode: {mode}")


# -----------------------------
# EMA
# -----------------------------
@torch.no_grad()
def update_ema(student, teacher, alpha=0.999):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(alpha).add_(ps.data, alpha=1 - alpha)


# -----------------------------
# Losses
# -----------------------------
def bce_masked_weighted(pred, target, mask, pos_weight=1.0, neg_weight=1.0, eps=1e-6):
    p = pred.clamp(eps, 1 - eps).to(dtype=target.dtype)
    loss = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    w_class = torch.where(
        target > 0.5,
        torch.as_tensor(pos_weight, dtype=loss.dtype, device=loss.device),
        torch.as_tensor(neg_weight, dtype=loss.dtype, device=loss.device),
    )
    w = w_class * mask
    denom = w.sum().clamp_min(1.0)
    return (loss * w).sum() / denom


def interval_gaussian_nll(mu, sd, a, b, mask):
    sd = sd.clamp_min(1e-6)
    alpha = (a - mu) / sd
    beta = (b - mu) / sd
    cdfa = torch.special.ndtr(alpha)
    cdfb = torch.special.ndtr(beta)
    prob = (cdfb - cdfa).clamp_min(1e-12)
    nll = -torch.log(prob)
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom


def kld(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def pairwise_separation_loss(
    mu,
    y,
    mask,
    margin=1.0,
    lambda_pull=0.01,
    lambda_push=0.05,
    subsample_max=256,
):
    device = mu.device
    lab = (mask > 0.5).squeeze(1)
    if lab.sum() < 2:
        return torch.tensor(0.0, device=device)

    idx = torch.nonzero(lab, as_tuple=False).squeeze(1)
    if idx.numel() > subsample_max:
        idx = idx[torch.randperm(idx.numel(), device=device)[:subsample_max]]

    e = F.normalize(mu[idx], dim=1)
    yy = (y[idx] > 0.5).squeeze(1)

    dots = e @ e.t()
    d2 = (2.0 - 2.0 * dots).clamp_min(0.0)
    d = torch.sqrt(d2 + 1e-9)

    M = torch.ones_like(d, dtype=torch.bool)
    M.fill_diagonal_(False)

    same = (yy.unsqueeze(1) == yy.unsqueeze(0)) & M
    diff = (~same) & M

    pull = d2[same].mean() if same.any() else torch.tensor(0.0, device=device)
    push = (
        F.relu(margin - d[diff]).mean()
        if diff.any()
        else torch.tensor(0.0, device=device)
    )
    return lambda_pull * pull + lambda_push * push


@torch.no_grad()
def teacher_stats_over_aug(teacher, x, cond, a, b, K, prior_aug, prior_kw):
    """Returns mean and variance of probs over K prior augmentations."""
    teacher.eval()
    ps = []
    for _ in range(K):
        xa = apply_prior_aug(x, prior_aug, **prior_kw)
        p, _, _, _, _ = teacher(xa, cond, a, b)
        ps.append(p)
    P = torch.stack(ps, dim=0)
    mean = P.mean(dim=0)
    var = P.var(dim=0, unbiased=False)
    return mean, var


# -----------------------------
# Validation (EMA teacher)
# -----------------------------
@torch.no_grad()
def evaluate(model_t, loader, device, lambda_sd):
    model_t.eval()
    agg = {k: 0.0 for k in ["total", "bce", "age", "sdreg"]}
    n = 0

    for x, cond, y, a, b, lab in loader:
        x, cond, y, a, b, lab = (
            x.to(device),
            cond.to(device),
            y.to(device),
            a.to(device),
            b.to(device),
            lab.to(device),
        )

        # Un-augmented prior at validation time, teacher model for stability
        p, mu_age, sd_age, _, _ = model_t(x, cond, a, b)

        L_bce = bce_masked_weighted(p, y, lab, pos_weight=1.0, neg_weight=1.0)
        mask_pos = lab * (y > 0.5).float()
        L_age = interval_gaussian_nll(mu_age, sd_age, a, b, mask_pos)

        sd_scaled = sd_age / (b - a + 1e-6)
        L_sdreg = (sd_scaled * mask_pos).sum() / (mask_pos.sum().clamp_min(1.0)) * lambda_sd

        total = L_bce + L_age + L_sdreg

        for k, v in zip(["total", "bce", "age", "sdreg"], [total, L_bce, L_age, L_sdreg]):
            agg[k] += float(v.item())

        n += 1

    n = max(n, 1)
    return {k: v / n for k, v in agg.items()}


# -----------------------------
# Training
# -----------------------------
def train_all(
    model_path,
    input_path,
    train_path,
    val_path,
    covariate_cols,
    latent_dim,
    max_epochs,
    batch_size=256,
    lr=2e-3,
    weight_decay=1e-4,
    warmup_epochs=50,
    max_beta=0.5,
    lambda_sd=0.01,
    lambda_pair_pull=0.1,
    lambda_pair_push=0.1,
    pair_margin=1.0,
    # New/missing args
    lambda_age_sup=1.0,       # weight for labeled age loss
    lambda_age_u=0.5,         # weight for unlabeled age loss
    age_u_ramp_epochs=50,     # ramp for unlabeled/consistency age losses
    lambda_age_cons=0.2,      # weight for EMA age consistency
    # FixMatch
    tau_pos=0.95,
    tau_neg=0.95,
    r_all=0.25,
    ema_alpha=0.99,
    lambda_fix_max=1.0,
    fix_ramp_epochs=40,
    # Prior augmentation
    prior_aug="gauss",
    prior_sigma_base=0.01,
    prior_sigma_extra=0.06,
    prior_dropout_p=0.10,
    prior_kappa=30.0,
    # Uncertainty gate
    K_aug=4,
    v_max=1e-3,
    # Early stop
    early_stop_patience=12,
    early_stop_warmup=50,
    seed=42,
):
    set_seed(seed)

    # Split by patient
    df = feather.read_feather(input_path)
    ids = df["patient_id"].unique()
    tr, va = train_test_split(ids, test_size=0.2, random_state=seed)

    df[df["patient_id"].isin(tr)].reset_index(drop=True).to_feather(train_path)
    df[df["patient_id"].isin(va)].reset_index(drop=True).to_feather(val_path)

    tr_ds = VAEDataset(train_path, covariate_cols)
    va_ds = VAEDataset(val_path, covariate_cols)

    dl_tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    x_dim, cond_dim = 1, len(covariate_cols)
    student = CVAE(x_dim, cond_dim, latent_dim)
    teacher = CVAE(x_dim, cond_dim, latent_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher.to(device)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6)

    best, bad = float("inf"), 0

    for epoch in range(1, max_epochs + 1):
        student.train()
        beta = min(max_beta, epoch / max(1, warmup_epochs))

        logs = {k: 0.0 for k in ["sup_bce", "age", "sd", "pair", "fix", "kl", "age_u", "age_cons", "total"]}
        sel_num = 0.0
        sel_pos_num = 0.0
        sel_neg_num = 0.0
        unl_num = 0.0

        for x, cond, y, a, b, lab in dl_tr:
            x, cond, y, a, b, lab = (
                x.to(device),
                cond.to(device),
                y.to(device),
                a.to(device),
                b.to(device),
                lab.to(device),
            )

            # ---- Shared prior augmentation (same x_aug for all branches) ----
            x_aug = apply_prior_aug(
                x,
                prior_aug,
                prior_sigma_base=prior_sigma_base,
                prior_sigma_extra=prior_sigma_extra,
                prior_dropout_p=prior_dropout_p,
                prior_kappa=prior_kappa,
            )

            # ---- Supervised (student) ----
            p_s, mu_age_s, sd_age_s, mu, lv = student(x_aug, cond, a, b)
            L_sup = bce_masked_weighted(p_s, y, lab, pos_weight=1.0, neg_weight=1.0)

            mask_pos = lab * (y > 0.5).float()
            L_age = interval_gaussian_nll(mu_age_s, sd_age_s, a, b, mask_pos) * lambda_age_sup

            sd_scaled = sd_age_s / (b - a + 1e-6)
            L_sd = (sd_scaled * mask_pos).sum() / (mask_pos.sum().clamp_min(1.0)) * lambda_sd

            L_pair = pairwise_separation_loss(
                mu,
                y,
                lab,
                margin=pair_margin,
                lambda_pull=lambda_pair_pull,
                lambda_push=lambda_pair_push,
            )

            # ---- Unlabeled gating (FixMatch) ----
            prior_kw = dict(
                prior_sigma_base=prior_sigma_base,
                prior_sigma_extra=prior_sigma_extra,
                prior_dropout_p=prior_dropout_p,
                prior_kappa=prior_kappa,
            )

            with torch.no_grad():
                teacher.eval()
                # teacher stats over K prior augmentations (for confidence + thresholds)
                pt_mean, pt_var = teacher_stats_over_aug(
                    teacher,
                    x,
                    cond,
                    a,
                    b,
                    K=K_aug,
                    prior_aug=prior_aug,
                    prior_kw=prior_kw,
                )

            unl = 1.0 - lab
            conf_gate = (pt_var <= v_max).float()

            if r_all is not None and r_all >= 0.0:
                # pool of eligible unlabeled predictions under the uncertainty gate
                pool_mask = (unl * conf_gate).squeeze(1) > 0.5
                pu = pt_mean[pool_mask].squeeze(1)  # [N_pool]

                if pu.numel() > 0:
                    s_pos = float(np.clip(pu.mean().item(), 0.2, 0.8))
                else:
                    s_pos = 0.5

                r_pos = max(0.0, min(1.0, s_pos * r_all))
                r_neg = max(0.0, min(1.0, r_all - r_pos))

                if pu.numel() > 1:
                    tau_pos_q = float(torch.quantile(pu, 1.0 - r_pos).item()) if r_pos > 0 else 1.0
                    tau_neg_q = float(torch.quantile(pu, r_neg).item()) if r_neg > 0 else 0.0
                else:
                    tau_pos_q, tau_neg_q = 1.0, 0.0

                sel_pos = ((pt_mean >= tau_pos_q).float() * unl) * conf_gate
                sel_neg = ((pt_mean <= tau_neg_q).float() * unl) * conf_gate
                sel = torch.maximum(sel_pos, sel_neg)
            else:
                # fallback: fixed thresholds
                sel_pos = ((pt_mean >= tau_pos).float() * unl) * conf_gate
                sel_neg = ((pt_mean <= (1.0 - tau_neg)).float() * unl) * conf_gate
                sel = torch.maximum(sel_pos, sel_neg)

            # soft pseudo-labels for classification consistency
            y_hat = pt_mean.clamp(1e-6, 1 - 1e-6).detach()

            # ---- Age loss for confident unlabeled positives (interval NLL on [a,b]) ----
            age_u_w = lambda_age_u * min(1.0, epoch / max(1, age_u_ramp_epochs))
            cons_w = lambda_age_cons * min(1.0, epoch / max(1, age_u_ramp_epochs))

            # only unlabeled confident positives
            mask_u_pos = (sel_pos * (1.0 - lab)).detach()
            L_age_u = interval_gaussian_nll(mu_age_s, sd_age_s, a, b, mask_u_pos) * age_u_w

            # ---- Classification consistency (FixMatch) ----
            L_fix_all = F.binary_cross_entropy(p_s.clamp(1e-6, 1 - 1e-6), y_hat, reduction="none")
            L_fix = (L_fix_all * sel).sum() / (sel.sum().clamp_min(1.0))
            L_fix = L_fix * (lambda_fix_max * min(1.0, epoch / max(1, fix_ramp_epochs)))

            # ---- Age consistency with EMA teacher (on the SAME augmented prior) ----
            with torch.no_grad():
                _, mu_age_t, sd_age_t, _, _ = teacher(x_aug, cond, a, b)

            den_c = mask_u_pos.sum().clamp_min(1.0)
            L_age_cons = (
                ((mu_age_s - mu_age_t).pow(2) + (torch.log(sd_age_s) - torch.log(sd_age_t)).pow(2))
                * mask_u_pos
            ).sum() / den_c
            L_age_cons = cons_w * L_age_cons

            # ---- KL (student) ----
            L_kl = kld(mu, lv) * beta

            # ---- Total ----
            L_tot = L_sup + L_age + L_sd + L_pair + L_fix + L_kl + L_age_u + L_age_cons

            opt.zero_grad(set_to_none=True)
            L_tot.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()

            update_ema(student, teacher, alpha=ema_alpha)

            # logging
            logs["sup_bce"] += float(L_sup)
            logs["age"] += float(L_age)
            logs["sd"] += float(L_sd)
            logs["pair"] += float(L_pair)
            logs["fix"] += float(L_fix)
            logs["kl"] += float(L_kl)
            logs["age_u"] += float(L_age_u)
            logs["age_cons"] += float(L_age_cons)
            logs["total"] += float(L_tot)

            sel_num += float(sel.sum())
            unl_num += float((1.0 - lab).sum())
            sel_pos_num += float(sel_pos.sum())
            sel_neg_num += float(sel_neg.sum())

        nB = max(len(dl_tr), 1)
        for k in logs:
            logs[k] /= nB

        sel_rate = sel_num / max(unl_num, 1e-6) if unl_num > 0 else 0.0
        print(
            f"Epoch {epoch}/{max_epochs} "
            f"| sup_bce:{logs['sup_bce']:.4f} | age:{logs['age']:.4f} | age_u:{logs['age_u']:.4f} "
            f"| age_cons:{logs['age_cons']:.4f} | sd:{logs['sd']:.4f} | pair:{logs['pair']:.4f} "
            f"| fix:{logs['fix']:.4f} | kl:{logs['kl']:.4f} | total:{logs['total']:.4f} "
            f"| sel_rate:{sel_rate:.3f} | sel_pos:{int(sel_pos_num)} sel_neg:{int(sel_neg_num)} "
        )

        # ---- Validation (EMA teacher, no prior aug) ----
        val = evaluate(teacher, dl_va, device, lambda_sd)
        print(
            f"VAL - total:{val['total']:.4f} | bce:{val['bce']:.4f} | age:{val['age']:.4f} | sd:{val['sdreg']:.4f}"
        )

        sched.step(val["total"])

        if val["total"] < best:
            best, bad = val["total"], 0
            torch.save(
                {
                    "teacher": teacher.state_dict(),
                    "student": student.state_dict(),
                    "args": {"covariates": covariate_cols, "latent_dim": latent_dim},
                },
                model_path,
            )
        else:
            if epoch < early_stop_warmup:
                bad = 0
            else:
                bad += 1

        if bad >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best val total: {best:.4f}")
            break

    print("Training complete. Best val total:", best)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("input_path")
    ap.add_argument("train_split_path")
    ap.add_argument("val_split_path")
    ap.add_argument("covariate_str")

    ap.add_argument("--latent_dim", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=50)
    ap.add_argument("--max_beta", type=float, default=0.5)

    ap.add_argument("--lambda_sd", type=float, default=0.5)
    ap.add_argument("--lambda_pair_pull", type=float, default=0.2)
    ap.add_argument("--lambda_pair_push", type=float, default=0.2)
    ap.add_argument("--pair_margin", type=float, default=1.0)

    # FixMatch age
    ap.add_argument("--lambda_age_sup", type=float, default=1.5)
    ap.add_argument("--lambda_age_u", type=float, default=0.5)
    ap.add_argument("--age_u_ramp_epochs", type=int, default=50)
    ap.add_argument("--lambda_age_cons", type=float, default=1.0)

    # FixMatch onset
    ap.add_argument(
        "--r_all",
        type=float,
        default=0.3,
        help="Overall fraction of unlabeled to pseudo-label per batch via quantiles (>=0 enables).",
    )
    ap.add_argument("--tau_pos", type=float, default=0.80)
    ap.add_argument("--tau_neg", type=float, default=0.95)
    ap.add_argument("--ema_alpha", type=float, default=0.99)
    ap.add_argument("--lambda_fix_max", type=float, default=1.0)
    ap.add_argument("--fix_ramp_epochs", type=int, default=50)

    # Uncertainty gate
    ap.add_argument("--K_aug", type=int, default=5, help="K augmentations for uncertainty check")
    ap.add_argument("--v_max", type=float, default=1e-3, help="Max variance across aug to accept pseudo-label")

    # Prior augmentation knobs
    ap.add_argument("--prior_aug", choices=["none", "gauss", "dropout", "beta"], default="beta")
    ap.add_argument("--prior_sigma_base", type=float, default=0.01)
    ap.add_argument("--prior_sigma_extra", type=float, default=0.06)
    ap.add_argument("--prior_dropout_p", type=float, default=0.50)
    ap.add_argument("--prior_kappa", type=float, default=30.0)

    ap.add_argument("--early_stop_patience", type=int, default=12)
    ap.add_argument("--early_stop_warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    covariate_cols = [s for s in args.covariate_str.split(",") if len(s.strip())]

    train_all(
        args.model_path,
        args.input_path,
        args.train_split_path,
        args.val_split_path,
        covariate_cols,
        args.latent_dim,
        args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_beta=args.max_beta,
        lambda_sd=args.lambda_sd,
        lambda_pair_pull=args.lambda_pair_pull,
        lambda_pair_push=args.lambda_pair_push,
        pair_margin=args.pair_margin,
        lambda_age_sup=args.lambda_age_sup,
        lambda_age_u=args.lambda_age_u,
        age_u_ramp_epochs=args.age_u_ramp_epochs,
        lambda_age_cons=args.lambda_age_cons,
        tau_pos=args.tau_pos,
        tau_neg=args.tau_neg,
        r_all=args.r_all,
        ema_alpha=args.ema_alpha,
        lambda_fix_max=args.lambda_fix_max,
        fix_ramp_epochs=args.fix_ramp_epochs,
        prior_aug=args.prior_aug,
        prior_sigma_base=args.prior_sigma_base,
        prior_sigma_extra=args.prior_sigma_extra,
        prior_dropout_p=args.prior_dropout_p,
        prior_kappa=args.prior_kappa,
        K_aug=args.K_aug,
        v_max=args.v_max,
        early_stop_patience=args.early_stop_patience,
        early_stop_warmup=args.early_stop_warmup,
        seed=args.seed,
    )
