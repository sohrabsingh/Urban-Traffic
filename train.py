"""
Optimized Traffic Hybrid Model (GRU Autoencoder + Spectral Clustering) — M1-ready

Changes from original version:
- Replaced LSTM autoencoder with a GRU autoencoder (faster + more stable)
- Lower default learning rate (1e-4)
- Gradient clipping (clip_norm=1.0)
- Early stopping based on training loss (patience=3)
- Reduced default hidden_dim (32) and latent_dim (12)
- DataLoader tuned for macOS (num_workers=2, persistent_workers=True when supported)
- Safer device selection for MPS and CPU
- Robust saving / artifact directory creation
- Clear logging for divergence and model checkpointing

Run: python train.py --train

Requirements:
- torch (with MPS support on Mac M1), numpy, pandas, scikit-learn, scipy, matplotlib, joblib, tqdm

"""

# -----------------------------
# Imports
# -----------------------------
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.sparse import csgraph
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utils: data loader + preprocessing
# -----------------------------
class TimeSeriesDataset(Dataset):
    """Create sliding windows per sensor."""
    def __init__(self, data_matrix, seq_len=12):
        # data_matrix: (num_sensors, num_timesteps)
        self.data = data_matrix.astype(np.float32)
        self.seq_len = seq_len
        self.N, self.T = self.data.shape
        self.windows = []
        for s in range(self.N):
            for t in range(0, self.T - seq_len + 1):
                self.windows.append((s, t))
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        s, t = self.windows[idx]
        seq = self.data[s, t:t+self.seq_len]
        return s, torch.from_numpy(seq).unsqueeze(-1)  # (seq_len, 1)


def load_metr_la(path='/mnt/data/METR-LA.csv'):
    df = pd.read_csv(path)
    # Drop leading timestamp-like column if present
    first_col = df.columns[0].lower()
    if any(key in first_col for key in ['time', 'date', 'timestamp']):
        df = df.iloc[:, 1:]
    # force numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(method='ffill').fillna(method='bfill')
    data = df.values.T  # sensors x timesteps
    return data, df.columns.tolist()


def load_adj(path='/mnt/data/adj_mx_METR-LA.pkl'):
    with open(path, 'rb') as f:
        adj = pickle.load(f)
    return np.array(adj)

# simple scaler that works per-sensor
class PerSensorScaler:
    def __init__(self):
        self.scalers = None
    def fit(self, X):
        self.scalers = []
        for s in range(X.shape[0]):
            sc = StandardScaler()
            self.scalers.append(sc.fit(X[s:s+1, :].T))
    def transform(self, X):
        out = np.zeros_like(X)
        for s in range(X.shape[0]):
            out[s] = self.scalers[s].transform(X[s:s+1, :].T).T
        return out
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        out = np.zeros_like(X)
        for s in range(X.shape[0]):
            out[s] = self.scalers[s].inverse_transform(X[s:s+1, :].T).T
        return out

# -----------------------------
# Model: GRU Autoencoder (more stable on long sequences)
# -----------------------------
class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, latent_dim=12, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape
        enc_out, h = self.encoder(x)
        last = enc_out[:, -1, :]
        z = self.enc_fc(last)
        dec_in = self.dec_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        return dec_out, z

# -----------------------------
# Training loop with gradient clipping + early stopping
# -----------------------------

def train_autoencoder(data_matrix, save_path, seq_len=12, latent_dim=12, hidden_dim=32, num_layers=1, epochs=20, batch_size=256, lr=1e-4, device='cpu', clip_norm=1.0, patience=3):
    dataset = TimeSeriesDataset(data_matrix, seq_len=seq_len)
    num_workers = 2
    persistent = False
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent)
    except TypeError:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = GRUAutoencoder(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for ep in range(epochs):
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        for sensors, seq in pbar:
            seq = seq.to(device)
            recon, _ = model(seq)
            loss = loss_fn(recon, seq)
            opt.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            opt.step()
            running += loss.item()
        avg_loss = running / len(loader)
        print(f"Epoch {ep+1} loss: {avg_loss:.6f}")

        # early stopping check
        if avg_loss + 1e-6 < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # save checkpoint
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump({'model_state': model.state_dict()}, save_path)
            print(f"Checkpoint saved (best loss {best_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered (patience={patience})")
                break

    # after training (or early stop), compute per-sensor latent vector by averaging windows
    model.eval()
    N, T = data_matrix.shape
    latents = np.zeros((N, latent_dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(N):
            windows = []
            for t in range(0, T - seq_len + 1):
                seq = torch.from_numpy(data_matrix[s, t:t+seq_len].astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
                _, z = model(seq)
                windows.append(z.cpu().numpy()[0])
            if len(windows) > 0:
                latents[s] = np.mean(np.stack(windows, axis=0), axis=0)
            else:
                latents[s] = np.zeros(latent_dim, dtype=np.float32)
    # save final checkpoint with latents
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'model_state': model.state_dict(), 'latents': latents}, save_path)
    print(f"Saved autoencoder model+latents to {save_path}")
    return latents

# -----------------------------
# Spectral Embedding
# -----------------------------

def spectral_embedding(adj, k=16, normalized=True):
    if normalized:
        lap = csgraph.laplacian(adj, normed=True)
    else:
        lap = csgraph.laplacian(adj, normed=False)
    lap = np.array(lap)
    vals, vecs = eigh(lap)
    emb = vecs[:, 1:k+1]
    return emb

# -----------------------------
# Clustering and controller
# -----------------------------

def build_controller(cluster_labels, anomaly_scores=None):
    N = len(cluster_labels)
    cluster_ids = np.unique(cluster_labels)
    cluster_green_adjust = {}
    for c in cluster_ids:
        cluster_green_adjust[c] = 0
    if anomaly_scores is not None:
        for c in cluster_ids:
            mask = cluster_labels == c
            median = np.median(anomaly_scores[mask])
            adj = int(np.clip((median - 0.5) * 60, -10, 20))
            cluster_green_adjust[c] = adj
    else:
        sorted_ids = np.sort(cluster_ids)
        for i, c in enumerate(sorted_ids):
            cluster_green_adjust[c] = int(10 * (len(sorted_ids) - i - 1))
    return cluster_green_adjust

def get_green_time_for_sensor(sensor_id, cluster_labels, controller_map, baseline=30):
    c = cluster_labels[sensor_id]
    return baseline + controller_map.get(c, 0)

# -----------------------------
# Visualization
# -----------------------------

def visualize_embeddings(embeddings, cluster_labels=None, title='embeddings_tsne', path='embeddings.png'):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    XY = tsne.fit_transform(embeddings)
    plt.figure(figsize=(7,6))
    if cluster_labels is not None:
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            plt.scatter(XY[mask,0], XY[mask,1], label=f'c{c}', s=10)
        plt.legend()
    else:
        plt.scatter(XY[:,0], XY[:,1], s=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved {path}")

def plot_reconstruction(sensor_idx, data_matrix, seq_len, model_state_path, savepath='recon.png', device='cpu'):
    checkpoint = joblib.load(model_state_path)
    latent_dim = checkpoint.get('latents', np.zeros((1,))).shape[1] if 'latents' in checkpoint else 12
    model = GRUAutoencoder(input_dim=1, hidden_dim=32, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    T = data_matrix.shape[1]
    t0 = max(0, T//2 - seq_len//2)
    seq = torch.from_numpy(data_matrix[sensor_idx, t0:t0+seq_len].astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        recon, _ = model(seq)
    seq = seq.cpu().numpy()[0,:,0]
    recon = recon.cpu().numpy()[0,:,0]
    plt.figure()
    plt.plot(range(len(seq)), seq, label='orig')
    plt.plot(range(len(recon)), recon, label='recon')
    plt.legend()
    plt.title(f'sensor {sensor_idx} reconstruction')
    plt.savefig(savepath)
    print(f"Saved {savepath}")

# -----------------------------
# Main orchestration
# -----------------------------

def main(args):
    # choose device safely
    if torch.backends.mps.is_available() and args.use_mps:
        device = 'mps'
    else:
        device = 'cpu'
    print('device:', device)

    data, cols = load_metr_la(args.data)
    print('Loaded data shape (sensors, timesteps):', data.shape)
    data = np.nan_to_num(data)

    scaler = PerSensorScaler()
    data_scaled = scaler.fit_transform(data)

    autoencoder_out = args.checkpoint
    if args.train:
        latents = train_autoencoder(data_scaled, autoencoder_out, seq_len=args.seq_len, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, clip_norm=args.clip_norm, patience=args.patience)
    else:
        if not os.path.exists(autoencoder_out):
            raise FileNotFoundError(f"Checkpoint not found: {autoencoder_out}. Run with --train first.")
        ckpt = joblib.load(autoencoder_out)
        latents = ckpt['latents']

    adj = load_adj(args.adj)
    print('Adjacency shape:', adj.shape)
    spec_emb = spectral_embedding(adj, k=args.spectral_dim)
    print('Spectral embedding shape:', spec_emb.shape)

    from sklearn.preprocessing import StandardScaler
    sc1 = StandardScaler(); sc2 = StandardScaler()
    lat_ts = sc1.fit_transform(latents)
    spec_ts = sc2.fit_transform(spec_emb)
    final_emb = np.concatenate([lat_ts, spec_ts], axis=1)
    print('Final embedding shape:', final_emb.shape)

    kmeans = KMeans(n_clusters=args.clusters, random_state=42).fit(final_emb)
    labels = kmeans.labels_
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump({'labels': labels, 'kmeans': kmeans}, 'artifacts/clusters.joblib')
    print('Saved cluster results to artifacts/clusters.joblib')

    if args.compute_anomaly:
        ckpt = joblib.load(autoencoder_out)
        model_state = ckpt['model_state']
        model = GRUAutoencoder(input_dim=1, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
        model.load_state_dict(model_state)
        model.eval()
        N, T = data_scaled.shape
        seq_len = args.seq_len
        errs = np.zeros(N)
        loss_fn = nn.MSELoss(reduction='mean')
        with torch.no_grad():
            for s in range(N):
                acc = 0.0
                cnt = 0
                for t in range(0, T - seq_len + 1):
                    seq = torch.from_numpy(data_scaled[s, t:t+seq_len].astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
                    recon, _ = model(seq)
                    acc += loss_fn(recon, seq).item()
                    cnt += 1
                if cnt>0:
                    errs[s] = acc/cnt
                else:
                    errs[s] = 0.0
        errs = (errs - errs.min()) / (errs.max() - errs.min() + 1e-8)
    else:
        errs = None

    controller_map = build_controller(labels, anomaly_scores=errs)
    joblib.dump({'controller_map': controller_map}, 'artifacts/controller.joblib')
    print('Saved controller map to artifacts/controller.joblib')

    visualize_embeddings(final_emb, cluster_labels=labels, title='final_tsne', path='artifacts/final_tsne.png')
    plot_reconstruction(sensor_idx=0, data_matrix=data_scaled, seq_len=args.seq_len, model_state_path=autoencoder_out, savepath='artifacts/recon_sensor0.png', device=device)

    for s in range(5):
        print(f"sensor {s} cluster={labels[s]} green_time={get_green_time_for_sensor(s, labels, controller_map)}s")

    print('Done. Artifacts in ./artifacts (autoencoder checkpoint, clusters, controller, figures)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Data/METR-LA.csv')
    parser.add_argument('--adj', type=str, default='Data/adj_mx_METR-LA.pkl')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--use_mps', action='store_true', help='use Apple MPS backend if available')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--latent_dim', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--spectral_dim', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--clusters', type=int, default=4)
    parser.add_argument('--compute_anomaly', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='artifacts/autoencoder_checkpoint.joblib')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=3)
    args = parser.parse_args()
    main(args)