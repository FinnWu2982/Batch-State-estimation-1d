import os
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ========= Config =========
DATA_PATH = r"dataset1.mat" 
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ========= Utilities =========
def load_dataset(path):
    d = sio.loadmat(path)

    # Required keys
    t = d["t"].ravel().astype(float)            # (K,)
    x_true = d["x_true"].ravel().astype(float)  # (K,)
    v = d["v"].ravel().astype(float)            # (K,)

    # y = x_c - r
    if "l" in d:
        xc = float(d["l"])
    elif "xc" in d:
        xc = float(d["xc"])
    else:
        raise KeyError("Missing cylinder center (l or xc) in .mat")

    if "r" not in d:
        raise KeyError("Missing range r in .mat")
    r = d["r"].ravel().astype(float)
    y = xc - r

    # Optional variances provided by file
    r_var = float(d["r_var"]) if "r_var" in d else None
    v_var = float(d["v_var"]) if "v_var" in d else None

    # Sampling period (assignment normalized to 0.1 s)
    if t.size > 1:
        T_median = float(np.median(np.diff(t)))
        assert abs(T_median - 0.1) < 1e-3, f"T != 0.1? got median Δt = {T_median:.6f}"
    T = 0.1

    return t, x_true, v, y, T, r_var, v_var

def _gauss_curve(mu, std, x):
    if std <= 0:
        return np.zeros_like(x)
    return 1.0/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/std)**2)

def plot_hist_and_stats(
    data, name, unit, path_prefix,
    force_bins=None, force_range=None, 
    y_as_pdf=True                       
):
    """
    Draw histogram+Gaussian fit and QQ plot; print μ,σ,σ². Return (μ,σ,σ²).
    If force_bins/range provided, histogram uses those settings (to match assignment figures).
    """
    data = np.asarray(data).ravel()
    mu = float(np.mean(data))
    std = float(np.std(data, ddof=1) if data.size > 1 else np.std(data))
    var = std**2
    print(f"{name}: μ = {mu:.6e} {unit}, σ = {std:.6e} {unit}, σ² = {var:.6e} {unit}²")

    # Histogram + Gaussian fit
    plt.figure(figsize=(7,4))
    hist_kwargs = dict(bins=(force_bins if force_bins else 20),
                       density=y_as_pdf)
    if force_range is not None:
        hist_kwargs["range"] = force_range
    plt.hist(data, alpha=0.9, color="navy", **hist_kwargs, label="Empirical")

    if force_range is not None:
        xs = np.linspace(force_range[0], force_range[1], 400)
    else:
        xs = np.linspace(mu - 4*std, mu + 4*std, 400)
    plt.plot(xs, _gauss_curve(mu, std, xs), 'r-', lw=2, label="N(μ̂,σ̂)")


    plt.ylabel("PDF")
    plt.xlabel(f"{name.split('(')[0].strip()} [{unit}]")  
    plt.title(f"Q1: {name}\nμ={mu:.6f} {unit}, σ={std:.6f} {unit}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{path_prefix}_hist.png"); plt.close()

    # QQ plot
    plt.figure(figsize=(7,4))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q1: Q–Q plot of {name}")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_qq.png"); plt.close()

    return mu, std, var

def analyze_Q1(t, x_true, v, y, T):
    """Compute & plot Q1 stats: measurement, speed, and process noise residuals."""
    # Measurement noise residuals (observation noise)
    er = y - x_true
    #bins=20, range=[-0.1, 0.1]
    mu_r, std_r, var_r = plot_hist_and_stats(
        er, "Range error", "m",
        os.path.join(FIG_DIR, "Q1_measurement"),
        force_bins=20, force_range=(-0.1, 0.1)
    )

    # Speed noise residuals (velocity noise)
    v_true = np.gradient(x_true, t)
    ev = v - v_true
    # —— 和作业保持一致：bins=20, range=[-0.2, 0.2]
    mu_v, std_v, var_v = plot_hist_and_stats(
        ev, "V error", "m/s",
        os.path.join(FIG_DIR, "Q1_speed"),
        force_bins=20, force_range=(-0.2, 0.2)
    )

    # Process noise residuals
    w = x_true[1:] - x_true[:-1] - T * v[1:]
    mu_w, std_w, var_w = plot_hist_and_stats(
        w, "Process noise residuals (w_k = x_k - x_{k-1} - T v_k)", "m",
        os.path.join(FIG_DIR, "Q1_process")
    )

    # Theoretical relation: σ_q = T σ_v
    print("\n===== Q1 Summary =====")
    print(f"σ_r ≈ {std_r:.4e} m, σ_r² ≈ {var_r:.4e} m²")
    print(f"σ_v ≈ {std_v:.4e} m/s, σ_v² ≈ {var_v:.4e} (m/s)²")
    print(f"[from σ_v]  σ_q = T·σ_v ≈ {T*std_v:.4e} m, σ_q² ≈ {(T*std_v)**2:.4e} m²")
    print(f"[from w_k ] σ_q ≈ {std_w:.4e} m, σ_q² ≈ {var_w:.4e} m²")
    print("Means: μ_r ≈ {:.2e} m, μ_v ≈ {:.2e} m/s, μ_w ≈ {:.2e} m".format(mu_r, mu_v, mu_w))
    print("======================\n")

    return {
        "mu_r": mu_r, "std_r": std_r, "var_r": var_r,
        "mu_v": mu_v, "std_v": std_v, "var_v": var_v,
        "mu_w": mu_w, "std_w": std_w, "var_w": var_w
    }

def estimate_noise_from_data(t, x_true, v, y, T):
    er = y - x_true
    sigma_r2 = float(np.var(er, ddof=1))
    v_true = np.gradient(x_true, t)
    ev = v - v_true
    sigma_v2 = float(np.var(ev, ddof=1))
    sigma_q2 = (T**2) * sigma_v2
    return sigma_q2, sigma_r2, sigma_v2, er, ev

def make_measure_mask(K, delta):
    mask = np.zeros(K, dtype=bool)
    idxs = list(range(delta-1, K, delta))  # 0-based
    if (K-1) not in idxs:
        idxs.append(K-1)
    mask[idxs] = True
    return mask, [i+1 for i in idxs]  # 1-based list for logging

def build_H_b(u, y, T, sigma_q2, sigma_r2, I_mask, x0=None):
    """
    Normal equations Hx=b for:
      x_k = x_{k-1} + T u_k + w_k,  w_k~N(0,σ_q²)
      y_k = x_k + n_k (only when I_mask[k]), n_k~N(0,σ_r²)
    Tridiagonal H:
      diag[0]   = 1/σ_q² + I0/σ_r²
      diag[K-1] = 1/σ_q² + IK-1/σ_r²
      diag[k]   = 2/σ_q² + Ik/σ_r²,  1≤k≤K-2
      off-diag  = -1/σ_q²
    RHS:
      b[0]    = I0*y0/σ_r² + (x0 + T u1)/σ_q²   (if x0 None, treat x0=0 here)
      b[k]    = Ik*yk/σ_r² + T(u_k - u_{k+1})/σ_q², 1≤k≤K-2
      b[K-1]  = IK-1*yK-1/σ_r² + T uK/σ_q²
    """
    K = y.size
    inv_q = 1.0 / sigma_q2
    inv_r = (1.0 / sigma_r2) if sigma_r2 > 0 else 0.0

    main = np.zeros(K)
    main[0]     = inv_q + (inv_r if I_mask[0] else 0.0)
    main[1:K-1] = 2*inv_q + (I_mask[1:K-1].astype(float) * inv_r)
    main[K-1]   = inv_q + (inv_r if I_mask[K-1] else 0.0)

    off = - np.ones(K-1) * inv_q
    H = diags([off, main, off], offsets=[-1, 0, 1], format="csr")

    b = np.zeros(K)
    if x0 is not None:
        b[0] = (inv_q)*(x0 + T*u[0]) + (inv_r * y[0] if I_mask[0] else 0.0)
    else:
        b[0] = (inv_q)*(T*u[0]) + (inv_r * y[0] if I_mask[0] else 0.0)
    for j in range(1, K-1):
        meas_term = (inv_r * y[j]) if I_mask[j] else 0.0
        b[j] = meas_term + (inv_q)*T*(u[j] - u[j+1])
    b[K-1] = (inv_r * y[K-1] if I_mask[K-1] else 0.0) + (inv_q)*T*u[K-1]
    return H, b

# ========= Q3/Q5: Kalman + RTS smoother =========
def rts_smoother(y, u, T, sigma_q2, sigma_r2, I_mask, x0_prior=0.0, P0_prior=1e6):
    K = y.size
    x_f = np.zeros(K)
    P_f = np.zeros(K)
    x_f[0] = x0_prior
    P_f[0] = P0_prior

    # initial measurement update at k=0 (if available)
    if I_mask[0]:
        K0 = P_f[0] / (P_f[0] + sigma_r2)
        x_f[0] = x_f[0] + K0*(y[0] - x_f[0])
        P_f[0] = (1 - K0)*P_f[0]

    # forward
    for k in range(1, K):
        x_pred = x_f[k-1] + T*u[k]
        P_pred = P_f[k-1] + sigma_q2
        if I_mask[k]:
            Kg = P_pred / (P_pred + sigma_r2)
            x_f[k] = x_pred + Kg*(y[k] - x_pred)
            P_f[k] = (1 - Kg)*P_pred
        else:
            x_f[k] = x_pred
            P_f[k] = P_pred

    # backward RTS
    x_s = np.copy(x_f)
    P_s = np.copy(P_f)
    eps = 1e-12
    for k in range(K-2, -1, -1):
        P_pred_next = P_f[k] + sigma_q2
        den = max(P_pred_next, eps)
        C = P_f[k] / den
        x_pred_next = x_f[k] + T*u[k+1]
        x_s[k] = x_f[k] + C*(x_s[k+1] - x_pred_next)
        P_s[k] = P_f[k] + C*(P_s[k+1] - P_pred_next)*C
    return x_s, P_s, x_f, P_f

# ========= Plot helpers =========
def plot_error_with_envelope(t, err, sigma, title, path):
    plt.figure(figsize=(7,4))
    plt.plot(t, err, label='Error (x_hat - x_true)', alpha=0.9)
    plt.plot(t,  3*sigma, 'r--', label='±3σ', alpha=0.8)
    plt.plot(t, -3*sigma, 'r--', alpha=0.8)
    plt.xlabel('Time [s]'); plt.ylabel('Position Error [m]')
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path); plt.close()

def plot_sparsity(H, title, path):
    plt.figure(figsize=(5,5))
    plt.spy(H, markersize=1)
    plt.title(title); plt.tight_layout()
    plt.savefig(path); plt.close()

# ========= Main pipeline =========
def main():
    t, x_true, v, y, T, r_var_file, v_var_file = load_dataset(DATA_PATH)
    K = t.size

    q1_stats = analyze_Q1(t, x_true, v, y, T)

    if (r_var_file is None) or (v_var_file is None):
        sigma_q2, sigma_r2, sigma_v2, _, _ = estimate_noise_from_data(t, x_true, v, y, T)
        print("[Q1-used-for-Q4/5] estimated from data:")
        print(f"   sigma_r^2 ≈ {sigma_r2:.6e} m^2")
        print(f"   sigma_v^2 ≈ {sigma_v2:.6e} (m/s)^2")
        print(f"   sigma_q^2 ≈ {sigma_q2:.6e} m^2")
    else:
        sigma_r2 = r_var_file
        sigma_v2 = v_var_file
        sigma_q2 = (T**2) * sigma_v2
        print("[Q1-used-for-Q4/5] variances from file:")
        print(f"   sigma_r^2 = {sigma_r2:.6e} m^2, sigma_v^2 = {sigma_v2:.6e} (m/s)^2, sigma_q^2 = {sigma_q2:.6e} m^2")

    delta_for_spy = 10
    I_mask, _ = make_measure_mask(K, delta_for_spy)
    H, b = build_H_b(v, y, T, sigma_q2, sigma_r2, I_mask, x0=None)
    plot_sparsity(H, f"Q4: Sparsity of H (delta={delta_for_spy})",
                  os.path.join(FIG_DIR, f"Q4_spy_delta{delta_for_spy}.png"))
    print(f"[Q4] Sparsity plot saved at {os.path.join(FIG_DIR, f'Q4_spy_delta{delta_for_spy}.png')}")
    # Optional cross-check: batch solve
    try:
        x_batch = spsolve(H, b)
    except Exception as e:
        x_batch = None
        print("[Q4] spsolve(H, b) failed:", e)

    # Debug blocks
    print("\n[DEBUG] Inspect first 5x5 block of H:")
    print(H[:5, :5].toarray())
    print("\n[DEBUG] Inspect last 5x5 block of H:")
    print(H[-5:, -5:].toarray())

    deltas = [1, 10, 100, 1000]
    for delta in deltas:
        I_mask, meas_list = make_measure_mask(K, delta)
        x_hat, P_diag, _, _ = rts_smoother(y, v, T, sigma_q2, sigma_r2, I_mask,
                                           x0_prior=0.0, P0_prior=1e6)
        err = x_hat - x_true
        sigma = np.sqrt(P_diag)

        frac_in_3sigma = float(np.mean(np.abs(err) <= 3*sigma))
        corr_abs = float(np.corrcoef(np.abs(err), sigma)[0,1]) if err.size > 1 else np.nan

        print(f"[Q5] delta={delta:4d}: mean(err)={np.mean(err): .4f} m, "
              f"std(err)={np.std(err, ddof=1): .4f} m, mean(σ)={np.mean(sigma): .4f} m, "
              f"meas_used={len(meas_list)} / {K}")
        print(f"     coverage within ±3σ = {frac_in_3sigma:.3f}, corr(|err|, σ) = {corr_abs:.3f}")

        # 时域误差 + ±3σ
        plot_error_with_envelope(t, err, sigma,
                                 f"Error & ±3σ (delta={delta})",
                                 os.path.join(FIG_DIR, f"Q5_error_time_delta{delta}.png"))

        # 误差直方图 + 高斯拟合（Q5 这里保持你原本风格）
        mu_e = np.mean(err); std_e = np.std(err, ddof=1)
        plt.figure(figsize=(7,4))
        plt.hist(err, bins=60, density=True, alpha=0.7, label="Empirical")
        if std_e > 0:
            xs = np.linspace(mu_e - 4*std_e, mu_e + 4*std_e, 400)
            pdf = 1/(std_e*np.sqrt(2*np.pi)) * np.exp(-0.5*((xs-mu_e)/std_e)**2)
            plt.plot(xs, pdf, 'r-', lw=2, label="N(μ̂,σ̂)")
        plt.xlabel("Error [m]"); plt.ylabel("PDF")
        plt.title(f"Histogram of errors (delta={delta})\nμ={mu_e:.4f} m, σ={std_e:.4f} m")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"Q5_error_hist_delta{delta}.png")); plt.close()

        # Optional: compare batch vs RTS for the delta used in H,b
        if (x_batch is not None) and (delta == delta_for_spy):
            linf = np.max(np.abs(x_batch - x_hat))
            print(f"     ||x_batch - x_rts||_inf (delta={delta_for_spy}) = {linf:.3e}")

    print(f"\nAll figures saved under: {FIG_DIR}/")

if __name__ == "__main__":
    main()
