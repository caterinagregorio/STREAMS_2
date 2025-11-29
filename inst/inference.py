import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True

# -----------------------------
# Dataset
# -----------------------------
class InferDataset(Dataset):
    def __init__(self, feather_path, covariate_cols, prior_col="onset_prior"):
        df = feather.read_feather(feather_path)

        required = ["patient_id", "a", "b", prior_col]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {feather_path}")

        for c in covariate_cols:
            if c and c not in df.columns:
                raise ValueError(f"Missing covariate '{c}' in {feather_path}")

        self.df = df.reset_index(drop=True)

        # prior -> tensor (N,1)
        prior = df[prior_col].fillna(0.5).clip(1e-6, 1-1e-6).astype("float32").values
        self.x = torch.tensor(prior, dtype=torch.float32).unsqueeze(1)

        # cond -> tensor (N,C) or empty
        if len(covariate_cols):
            cond = df[covariate_cols].astype("float32").values
        else:
            cond = np.zeros((len(df), 0), dtype="float32")
        self.cond = torch.tensor(cond, dtype=torch.float32)

        # intervals
        self.a = torch.tensor(df["a"].values, dtype=torch.float32).unsqueeze(1)
        self.b = torch.tensor(df["b"].values, dtype=torch.float32).unsqueeze(1)

        # patient ids (kept as numpy to return raw)
        self.patient_id = df["patient_id"].astype(np.int64).values

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        # return ID as plain int to avoid unwanted tensor conversions
        return self.x[idx], self.cond[idx], self.a[idx], self.b[idx], int(self.patient_id[idx])

# -----------------------------
# Model 
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_lv = nn.Linear(64, latent_dim)

    def forward(self, x, cond):
        h = torch.cat([x, cond], dim=1)
        h = self.net(h)
        return self.fc_mu(h), self.fc_lv(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ReLU()
        )
        self.fc_p  = nn.Linear(64, 1)
        self.fc_mu = nn.Linear(64, 1)
        self.fc_sd = nn.Linear(64, 1)

    def forward(self, z, cond, a, b):
        h = torch.cat([z, cond], dim=1)
        h = self.net(h)
        p_onset = torch.sigmoid(self.fc_p(h))
        age_mu  = a + F.softplus(self.fc_mu(h))          
        age_sd  = F.softplus(self.fc_sd(h)) + 1e-6   
        return p_onset, age_mu, age_sd

class CVAE(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim):
        super().__init__()
        self.enc = Encoder(x_dim, cond_dim, latent_dim)
        self.dec = Decoder(latent_dim, cond_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond, a, b, deterministic: bool = False):
        mu, lv = self.enc(x, cond)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            z = mu + eps * std
        p_onset, age_mu, age_sd = self.dec(z, cond, a, b)
        return p_onset, age_mu, age_sd, mu, lv

# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def predict(model, loader, device, mc_samples: int = 1):
    model.eval()
    P, MU, SD, A, B, PIDS = [], [], [], [], [], []
    for x, cond, a, b, pid in loader:
        # pid comes as a list of ints from the default collate; keep separate
        x, cond, a, b = x.to(device), cond.to(device), a.to(device), b.to(device)
        if mc_samples == 1:
            p, mu, sd, _, _ = model(x, cond, a, b, deterministic=True)
        else:
            ps, mus, sds = [], [], []
            for _ in range(mc_samples):
                p_i, mu_i, sd_i, _, _ = model(x, cond, a, b, deterministic=False)
                ps.append(p_i); mus.append(mu_i); sds.append(sd_i)
            p  = torch.stack(ps).mean(0)
            mu = torch.stack(mus).mean(0)
            sd = torch.stack(sds).mean(0)
        
        P.append(p.squeeze(1).cpu().numpy())
        MU.append(mu.squeeze(1).cpu().numpy())
        SD.append(sd.squeeze(1).cpu().numpy())
        A.append(a.squeeze(1).cpu().numpy())
        B.append(b.squeeze(1).cpu().numpy())
        # pid is a list[int] (len=batch); convert to numpy
        if isinstance(pid, list):
            PIDS.append(np.array(pid, dtype=np.int64))
        else:
            # in case collate changed
            PIDS.append(np.asarray(pid, dtype=np.int64))

    P   = np.concatenate(P)   if P else np.zeros(0)
    MU  = np.concatenate(MU)  if MU else np.zeros(0)
    SD  = np.concatenate(SD)  if SD else np.zeros(0)
    A   = np.concatenate(A)   if A else np.zeros(0)
    B   = np.concatenate(B)   if B else np.zeros(0)
    PID = np.concatenate(PIDS) if PIDS else np.zeros(0, dtype=np.int64)

    # interval probability Φ((b-μ)/σ) - Φ((a-μ)/σ)
    t_mu = torch.tensor(MU)
    t_sd = torch.tensor(SD).clamp_min(1e-6)
    t_a  = torch.tensor(A); t_b = torch.tensor(B)
    alpha = (t_a - t_mu) / t_sd
    beta  = (t_b - t_mu) / t_sd
    cdf_a = torch.special.ndtr(alpha)
    cdf_b = torch.special.ndtr(beta)
    interval_prob = (cdf_b - cdf_a).clamp_min(0.0).numpy()

    return PID, P, MU, SD, interval_prob

def save_table(path, df):
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        feather.write_feather(df, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str, help="Path to best checkpoint (.pt)")
    ap.add_argument("infer_feather", type=str, help="Feather with patient_id,a,b,onset_prior and covariates")
    ap.add_argument("covariate_str", type=str, help="Comma-separated covariate names (same as training)")
    ap.add_argument("--latent_dim", type=int, default=5, help="Must match training")
    ap.add_argument("--out", type=str, default="predictions.feather")
    ap.add_argument("--use_student", action="store_true", help="Use student weights instead of EMA teacher")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--mc_samples", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    covs = [c.strip() for c in args.covariate_str.split(",") if c.strip()]
    ds = InferDataset(args.infer_feather, covs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    x_dim, cond_dim = 1, len(covs)
    model = CVAE(x_dim, cond_dim, args.latent_dim).to(device)

    # Load checkpoint (supports dict with teacher/student or plain state_dict)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and ("teacher" in ckpt or "student" in ckpt):
        state = ckpt["student"] if args.use_student else ckpt["teacher"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)

    pid, p, mu, sd, interval_prob = predict(model, dl, device)

    out = pd.DataFrame({
        "patient_id": pid.astype(np.int64),
        "p_onset": p.astype(np.float32),
        "age_mu": mu.astype(np.float32),
        "age_sd": sd.astype(np.float32),
        "interval_prob": interval_prob.astype(np.float32),
    })

    # Optionally emit student predictions too (if checkpoint had both) and we're using teacher
    if isinstance(ckpt, dict) and ("teacher" in ckpt and "student" in ckpt) and (not args.use_student):
        model_student = CVAE(x_dim, cond_dim, args.latent_dim).to(device)
        model_student.load_state_dict(ckpt["student"], strict=True)
        pid2, p_s, _, _, _ = predict(model_student, dl, device)
        out = out.merge(pd.DataFrame({
            "patient_id": pid2.astype(np.int64),
            "p_onset_student": p_s.astype(np.float32),
        }), on="patient_id", how="left")

    save_table(args.out, out)
    print(f"Wrote predictions to {args.out}  (rows: {len(out)})")

if __name__ == "__main__":
    main()
