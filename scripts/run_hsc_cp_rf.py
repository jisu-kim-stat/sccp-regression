import os
import json
import math
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans


# ----------------------------
# Conformal quantile
# ----------------------------
def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        raise ValueError("Empty score array.")
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def fit_rf(X, y, seed):
    return RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=30,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    ).fit(X, y)


def summarize(y, lo, hi):
    cov = np.mean((y >= lo) & (y <= hi))
    wid = np.mean(hi - lo)
    return cov, wid


def print_results(tag, y, lo, hi, tail_thrs=(1.146, 1.967)):
    cov, wid = summarize(y, lo, hi)
    print(f"{tag:>5} | overall: coverage={cov:.4f} | mean_width={wid:.4f}")
    for thr in tail_thrs:
        m = y > thr
        if m.sum() == 0:
            continue
        cov_t, wid_t = summarize(y[m], lo[m], hi[m])
        print(f"{tag:>5} | tail z>{thr}: n={m.sum():6d} | coverage={cov_t:.4f} | mean_width={wid_t:.4f}")


# ----------------------------
# Main
# ----------------------------
def main(args):
    alpha = args.alpha
    gamma = args.gamma
    M = args.M
    tau = args.tau
    eps = 1e-6

    base = os.path.expanduser(args.data_dir)
    train = pd.read_parquet(os.path.join(base, "HSC_train.parquet"))
    cal   = pd.read_parquet(os.path.join(base, "HSC_cal.parquet"))
    test  = pd.read_parquet(os.path.join(base, "HSC_test.parquet"))

    x_cols = ["g_cmodel_mag", "r_cmodel_mag", "i_cmodel_mag",
              "z_cmodel_mag", "y_cmodel_mag"]
    y_col = "specz_redshift"

    Xtr, ytr = train[x_cols].to_numpy(), train[y_col].to_numpy()
    Xcal, ycal = cal[x_cols].to_numpy(), cal[y_col].to_numpy()
    Xte, yte = test[x_cols].to_numpy(), test[y_col].to_numpy()

    print(f"Shapes | train={Xtr.shape}, cal={Xcal.shape}, test={Xte.shape}")

    # ----------------------------
    # Base models
    # ----------------------------
    print("\n[1] Fitting mean model μ(x)")
    m_mu = fit_rf(Xtr, ytr, seed=0)

    r_tr = np.abs(ytr - m_mu.predict(Xtr))
    print("[2] Fitting scale model σ(x)")
    m_sig = fit_rf(Xtr, r_tr, seed=1)

    # ----------------------------
    # Global CP (GCP)
    # ----------------------------
    yhat_cal = m_mu.predict(Xcal)
    sig_cal = np.maximum(m_sig.predict(Xcal), eps)
    scores_cal = np.abs(ycal - yhat_cal) / sig_cal
    q_global = conformal_quantile(scores_cal, alpha)

    yhat_te = m_mu.predict(Xte)
    sig_te = np.maximum(m_sig.predict(Xte), eps)

    lo_g = yhat_te - q_global * sig_te
    hi_g = yhat_te + q_global * sig_te

    print("\n=== TEST RESULTS ===")
    print_results("GCP", yte, lo_g, hi_g)

    # ----------------------------
    # Calibration split (γ)
    # ----------------------------
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(Xcal))
    n_clust = int(gamma * len(Xcal))

    idx_clust = perm[:n_clust]
    idx_prop  = perm[n_clust:]

    Xcal_clust = Xcal[idx_clust]
    Xcal_prop  = Xcal[idx_prop]
    ycal_prop  = ycal[idx_prop]

    yhat_prop = m_mu.predict(Xcal_prop)
    sig_prop = np.maximum(m_sig.predict(Xcal_prop), eps)
    scores_prop = np.abs(ycal_prop - yhat_prop) / sig_prop

    # ----------------------------
    # Weighted K-means Clustering
    # ----------------------------
    print(f"\n[3] KMeans clustering: M={M}, gamma={gamma}")
    y_clust = ycal[idx_clust]

    if args.bin_mode == "quantile":
        # Quantile-based bins
        Kbin = args.Kbin
        qs = np.quantile(y_clust, np.linspace(0, 1, Kbin + 1))
        qs[0] = -np.inf
        qs[-1] = np.inf

        bin_id = np.digitize(y_clust, qs[1:-1], right=True)
        counts = np.bincount(bin_id, minlength=Kbin)

        print(f"[3] Weighted KMeans (quantile bins): Kbin={Kbin}")
        print("    bin counts summary:",
            f"min={counts.min()}, p5={np.quantile(counts,0.05):.1f}, "
            f"median={np.median(counts):.1f}, max={counts.max()}")

    elif args.bin_mode == "tail":
        # Tail-focused quantile bins
        tail_q = sorted(args.tail_q)
        cut_vals = np.quantile(y_clust, tail_q)

        qs = np.concatenate(([-np.inf], cut_vals, [np.inf]))
        Kbin = len(qs) - 1

        bin_id = np.digitize(y_clust, qs[1:-1], right=True)
        counts = np.bincount(bin_id, minlength=Kbin)

        print(f"[3] Weighted KMeans (tail bins): tail_q={tail_q}")
        print("    cut values:", [float(v) for v in cut_vals])
        print("    bin counts:", counts.tolist())

    elif args.bin_mode == "uniform":
        # Uniform-width bins in y-space
        Kbin = args.Kbin
        y_min, y_max = np.min(y_clust), np.max(y_clust)

        qs = np.linspace(y_min, y_max, Kbin + 1)
        qs[0] = -np.inf
        qs[-1] = np.inf

        bin_id = np.digitize(y_clust, qs[1:-1], right=True)
        counts = np.bincount(bin_id, minlength=Kbin)

        print(f"[3] Weighted KMeans (uniform bins): Kbin={Kbin}")
        print("    y-range:",
            f"[{y_min:.3f}, {y_max:.3f}]")
        print("    bin counts summary:",
            f"min={counts.min()}, p5={np.quantile(counts,0.05):.1f}, "
            f"median={np.median(counts):.1f}, max={counts.max()}")

    else:
        raise ValueError("Unknown bin_mode")


    w = np.sqrt(counts[bin_id]).astype(float)

    kmeans = KMeans(n_clusters=M, random_state=0, n_init="auto")
    kmeans.fit(Xcal_clust, sample_weight=w)

    c_prop = kmeans.predict(Xcal_prop)
    c_te = kmeans.predict(Xte)
    
    # ----------------------------
    # CCCP
    # ----------------------------
    q_cccp = np.zeros(M)
    n_k = np.zeros(M, dtype=int)

    for k in range(M):
        idx = (c_prop == k)
        n_k[k] = idx.sum()
        q_cccp[k] = conformal_quantile(scores_prop[idx], alpha)

    lo_c = yhat_te - q_cccp[c_te] * sig_te
    hi_c = yhat_te + q_cccp[c_te] * sig_te

    print_results("CCCP", yte, lo_c, hi_c)

    # ----------------------------
    # SCCP (shrinkage)
    # ----------------------------
    tau_k = tau / (tau + n_k)
    q_sccp = (1 - tau_k) * q_cccp + tau_k * q_global

    lo_s = yhat_te - q_sccp[c_te] * sig_te
    hi_s = yhat_te + q_sccp[c_te] * sig_te

    print_results("SCCP", yte, lo_s, hi_s)

    # ----------------------------
    # Save summary
    # ----------------------------
    out = {
        "alpha": alpha,
        "gamma": gamma,
        "M": M,
        "tau": tau,
        "q_global": q_global,
        "cluster_sizes": n_k.tolist(),
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/hsc_rf_M{M}_g{gamma}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=10,
                        help="Number of clusters")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Fraction of calibration used for clustering")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Miscoverage level")
    parser.add_argument("--tau", type=float, default=100.0,
                        help="SCCP shrinkage strength")
    parser.add_argument("--data_dir", type=str,
                        default="~/sccp_regression/data/hsc",
                        help="Directory containing parquet files")
    parser.add_argument("--Kbin", type=int, default=10,
                    help="Number of pseudo-classes for weighted k-means (quantile bins of y)")
    parser.add_argument("--bin_mode", type=str, default="quantile",
                        choices=["quantile", 'tail', 'uniform'],
                        help = 'Pseudo-class binning mode for weighted k-means')
    parser.add_argument("--tail_q", type=float, nargs="*", default=[0.90, 0.95, 0.99],
                        help = 'Quantiles for tail binning mode')
    

    args = parser.parse_args()

    main(args)
