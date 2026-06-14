"""
Illustrate the three terms of the LIG signal-harvesting action

    S[gamma, mu] = INT rho(t)^2 dt              (path distortion)
                 - lambda INT |grad f . gdot| mu dt   (signal harvesting)
                 + (tau/2) INT mu^2 dt           (measure regulariser)

along the STRAIGHT-LINE path baseline -> input on a real image, so the three
quantities are computed from the actual model f (ResNet-50), not a toy.

NOTE on rho (panel 1)
---------------------
The continuous definition rho(t) = (1/2 gdot^T H gdot)/(grad f . gdot) uses the
Hessian. ResNet-50 is piecewise-linear (ReLU), so an *analytic* Hessian-vector
product is ~0 almost everywhere and gives a degenerate, flat rho ~ 1e-17.
We instead measure the linearisation error the way the paper's discrete objective
does (Eqs. 4-6): the residual between the actual output change and the
gradient-predicted change,

    d_k   = grad f(x_k) . (x_{k+1}-x_k)          (gradient-predicted change)
    df_k  = f(x_{k+1}) - f(x_k)                  (actual change)
    r_k   = df_k - d_k                           (linearisation residual)
    rho_k = r_k / d_k    ( = 1 - phi_k,  phi_k = d_k/df_k )

This captures the curvature the straight line ignores -- including the ReLU
piecewise nonlinearity that an analytic HVP misses -- and is exactly the
quantity the Hessian-free surrogate Var_nu(phi) is built on.

Reuses the preprocessing / model conventions from saturation_integrate.py.
Edit IMAGE_PATH below.
Run: python lig_three_terms.py
"""

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_PATH = "church.JPEG"

STEPS = 64                  # nodes along the straight-line path
LAMBDA = 1.0               # signal-harvesting weight (matches paper default)
TAUS = [1.0, 0.05, 0.005]  # measure-regulariser strengths to sweep in panel 3
RHO_CLIP = 3.0             # clip |rho| for display (degenerate steps blow up)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# --------------------------------------------------------------------------- #
# Model / preprocessing  (same conventions as saturation_integrate.py)
# --------------------------------------------------------------------------- #
def load_model():
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(DEVICE).eval()
    return model, weights


def preprocess(pil_img):
    """RGB float [0,1], 224x224, HxWxC, not normalized."""
    tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
    img = tf(pil_img.convert("RGB"))
    return np.asarray(img).astype(np.float32) / 255.0


def to_model_tensor(batch_hwc):
    """(N,H,W,C) float [0,1] -> normalized (N,C,H,W) tensor."""
    x = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict_logits(model, batch_hwc, bs=64):
    out = []
    for i in range(0, len(batch_hwc), bs):
        chunk = to_model_tensor(batch_hwc[i:i + bs]).to(DEVICE)
        out.append(model(chunk).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Per-step quantities along the straight line
# --------------------------------------------------------------------------- #
def path_quantities(model, image, label, steps=STEPS, baseline=None):
    """
    Straight line x(alpha) = baseline + alpha (image - baseline), alpha in [0,1].

    Node-level (length `steps`):
        alphas[k]  : interpolation fraction
        f[k]       : class logit  f(x_k)
        gdotd[k]   : directional derivative grad f(x_k).(image-baseline)
                     (the gradient-predicted slope along the path direction)

    Interval-level (length `steps-1`):
        a_mid[k]   : midpoint alpha of interval k
        d[k]       : gradient-predicted change over the interval (left-endpoint)
        df[k]      : actual output change f(x_{k+1}) - f(x_k)
        r[k]       : linearisation residual df[k] - d[k]
        rho[k]     : r[k] / d[k]      ( = 1 - phi_k )

    'dir' is the path direction (image - baseline) in the *normalized* model-input
    space, since gradients are wrt the normalized input.
    """
    if baseline is None:
        baseline = np.zeros_like(image)

    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    path = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]

    # full straight-line displacement in normalized space, shape (3,H,W)
    dir_full = (image - baseline).transpose(2, 0, 1) / IMAGENET_STD[:, None, None]
    dir_t = torch.tensor(dir_full, dtype=torch.float32, device=DEVICE)

    f = np.empty(steps, np.float32)
    gdotd = np.empty(steps, np.float32)   # grad f . (full displacement)

    for k in range(steps):
        x = to_model_tensor(path[k:k + 1]).to(DEVICE).requires_grad_(True)
        logit = model(x)[0, label]
        (g,) = torch.autograd.grad(logit, x)
        dd = (g * dir_t[None]).sum()                  # grad f . dir_full
        f[k] = float(logit.detach())
        gdotd[k] = float(dd.detach())

    # interval quantities (left-endpoint convention, uniform step length 1/(N-1))
    n_int = steps - 1
    step_len = 1.0 / n_int
    a_mid = 0.5 * (alphas[:-1] + alphas[1:])
    d = gdotd[:-1] * step_len                          # d_k = grad.(x_{k+1}-x_k)
    df = np.diff(f)                                    # actual change per interval
    r = df - d                                         # linearisation residual

    # rho_k = r_k / d_k, defined where |d_k| not tiny
    eps = 1e-3 * (np.abs(d).max() + 1e-12)
    safe = np.abs(d) > eps
    rho = np.full(n_int, np.nan, np.float32)
    rho[safe] = r[safe] / d[safe]

    return alphas, f, gdotd, a_mid, d, df, r, rho, safe


# --------------------------------------------------------------------------- #
# Measure QP:  min_mu  -lambda sum mu_k a_k + (tau/2)||mu||^2,  mu>=0, sum=1
# closed-form simplex solution via water-filling on the KKT stationarity
#   mu_k = max(0, (lambda a_k - eta)/tau),  eta chosen so sum mu = 1
# --------------------------------------------------------------------------- #
def solve_measure_qp(a, lam, tau):
    a = np.asarray(a, np.float64)
    n = len(a)
    s = lam * a                       # mu_k = max(0,(s_k - eta)/tau)
    order = np.argsort(s)[::-1]
    s_sorted = s[order]
    prefix = np.cumsum(s_sorted)

    mu_full = None
    for m in range(1, n + 1):
        # require sum over active set of (s_k - eta)/tau = 1
        eta = (prefix[m - 1] - tau) / m
        # active condition: smallest active value positive, next one (if any) non-positive
        smallest_active = s_sorted[m - 1] - eta
        next_inactive = (s_sorted[m] - eta) if m < n else -np.inf
        if smallest_active > 0 and next_inactive <= 0:
            mu_sorted = np.maximum(0.0, (s_sorted - eta) / tau)
            mu_full = np.zeros(n)
            mu_full[order] = mu_sorted
            break

    if mu_full is None:               # fallback: uniform
        mu_full = np.full(n, 1.0 / n)

    mu_full = np.maximum(mu_full, 0.0)
    mu_full /= mu_full.sum()
    return mu_full


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    label = int(predict_logits(model, image[None])[0].argmax())
    print(f"Predicted: {categories[label]}")

    alphas, f, gdotd, a_mid, d, df, r, rho, safe = path_quantities(model, image, label)

    # sanity: completeness  sum d ~ f(1)-f(0)?
    print(f"f(0)={f[0]:.3f}  f(1)={f[-1]:.3f}  f(1)-f(0)={f[-1]-f[0]:.3f}")
    print(f"sum d_k = {d.sum():.3f}   sum df_k = {df.sum():.3f}")
    print(f"rho: defined on {safe.sum()}/{len(rho)} steps, "
          f"median |rho|={np.nanmedian(np.abs(rho)):.3f}")

    # signal per interval: |grad f . dir| at left node
    sig = np.abs(gdotd[:-1])
    # |d_k| drives the measure QP
    dk = np.abs(d)

    # clip rho for display
    rho_disp = np.clip(rho, -RHO_CLIP, RHO_CLIP)

    # ----- figure layout: image + 3 term panels -----
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0],
                  hspace=0.45, wspace=0.30)

    # (0) input image -------------------------------------------------------- #
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"input  ({categories[label]})", fontsize=11)
    ax_img.axis("off")

    # (1) PATH DISTORTION: rho(t) from the real residual -------------------- #
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.plot(a_mid, rho_disp, color="#b5179e", lw=1.8,
             label=r"$\rho_k=(\Delta f_k-d_k)/d_k$")
    rbar = np.nanmean(rho)
    if np.isfinite(rbar):
        rbar_c = np.clip(rbar, -RHO_CLIP, RHO_CLIP)
        ax1.axhline(rbar_c, color="#0a7d24", ls="--", lw=1.6,
                    label=r"E-L target  $\rho=$const")
    ax1.fill_between(a_mid, 0, rho_disp, color="#b5179e", alpha=0.12)
    ax1.set_title(r"(1) path distortion  $\int\rho(t)^2\,dt$"
                  "\nlinearisation error the straight line ignores", fontsize=10)
    ax1.set_xlabel(r"$\alpha$  (0 = baseline $\to$ 1 = input)")
    ax1.set_ylabel(r"$\rho$  (clipped at $\pm$%.0f)" % RHO_CLIP)
    ax1.set_ylim(-RHO_CLIP * 1.1, RHO_CLIP * 1.1)
    ax1.grid(alpha=0.22)
    ax1.legend(fontsize=8, frameon=False, loc="upper right")

    # (2) SIGNAL HARVESTING: |grad f . gdot| -------------------------------- #
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(a_mid, sig, color="#1f4e8c", lw=2.0)
    ax2.fill_between(a_mid, 0, sig, color="#1f4e8c", alpha=0.15)
    if sig.sum() > 0:
        c = np.cumsum(sig) / sig.sum()
        lo = a_mid[np.searchsorted(c, 0.10)]
        hi = a_mid[min(np.searchsorted(c, 0.90), len(a_mid) - 1)]
        ax2.axvspan(lo, hi, color="#1f4e8c", alpha=0.06)
        ax2.annotate("transition region\n(signal lives here)",
                     xy=(0.5 * (lo + hi), sig.max() * 0.7),
                     fontsize=8, color="#1f4e8c", ha="center")
    ax2.set_title(r"(2) signal harvesting  $-\lambda\int|\nabla f\cdot\dot\gamma|\,\mu\,dt$"
                  "\nreward steps that actually move $f$", fontsize=10)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$|\nabla f\cdot\dot\gamma|$")
    ax2.grid(alpha=0.22)

    # (3) MEASURE REGULARISER: mu* for several tau -------------------------- #
    ax3 = fig.add_subplot(gs[1, :])
    colors = ["#0a7d24", "#1f4e8c", "#d00000"]
    n_int = len(dk)
    ax3.axhline(1.0 / n_int, color="0.5", ls=":", lw=1.4,
                label=r"uniform  $\mu_k=1/N$  ($\tau\to\infty$)")
    for col, tau in zip(colors, TAUS):
        mu = solve_measure_qp(dk, LAMBDA, tau)
        ax3.plot(a_mid, mu, color=col, lw=1.8, marker="o", ms=2.5,
                 label=fr"$\mu^\star$  ($\tau={tau:g}$)")
    ax3.set_title(r"(3) measure regulariser  $\frac{\tau}{2}\int\mu^2\,dt$"
                  r"   —   prevents $\mu$ collapsing onto a single step"
                  "\nsmall $\\tau$ $\\to$ Dirac-like spike on the best step;"
                  "  large $\\tau$ $\\to$ spreads toward uniform", fontsize=10)
    ax3.set_xlabel(r"$\alpha$  (0 = baseline $\to$ 1 = input)")
    ax3.set_ylabel(r"measure weight  $\mu_k$")
    ax3.grid(alpha=0.22)
    ax3.legend(fontsize=9, frameon=False, ncol=4, loc="upper center")

    fig.suptitle(r"Three terms of the signal-harvesting action $S[\gamma,\mu]$"
                 "  (straight-line path on a real image)",
                 fontsize=13, y=0.98)

    plt.savefig("lig_three_terms.png", bbox_inches="tight", dpi=150)
    print("Saved: lig_three_terms.png")


if __name__ == "__main__":
    main()