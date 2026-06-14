"""
Illustrate the three terms of the LIG signal-harvesting action

    S[gamma, mu] = INT rho(t)^2 dt              (path distortion)
                 - lambda INT |grad f . gdot| mu dt   (signal harvesting)
                 + (tau/2) INT mu^2 dt           (measure regulariser)

along the STRAIGHT-LINE path baseline -> input on a real image, so the three
quantities are computed from the actual model f (ResNet-50), not a toy.

What each panel shows
---------------------
(0) the input image.
(1) PATH DISTORTION: the per-step relative linearisation error
        rho_k = (1/2 gdot^T H gdot) / (grad f . gdot)            [Eq. 11]
    computed via a Hessian-vector product (no explicit Hessian). High |rho|
    marks steps where curvature -- the part the straight line ignores --
    dominates the output change. The term rho^2 penalises exactly these.
    The Euler-Lagrange target is rho = const (error spread evenly), NOT the
    spiky profile the straight line produces.
(2) SIGNAL HARVESTING: |grad f . gdot|, the directional derivative magnitude.
    Large where f actually moves (transition region), ~0 in flat/saturated
    regions. The signal term rewards putting weight here.
(3) MEASURE REGULARISER: the optimal measure mu* solving the convex QP
        min_mu  -lambda sum_k mu_k |d_k| + (tau/2) ||mu||^2,  mu in simplex
    for several tau. Small tau -> mu collapses toward a Dirac on the best
    step; large tau -> mu spreads to uniform. The term (tau/2)||mu||^2 is the
    force that prevents the single-step collapse.

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

STEPS = 64                # nodes along the straight-line path
LAMBDA = 1.0              # signal-harvesting weight (matches paper default)
TAUS = [1.0, 0.05, 0.005]  # measure-regulariser strengths to sweep in panel 3
HVP_HUTCHINSON = False    # rho uses an exact HVP along the path direction (not stochastic)

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

    For each node k returns:
        alphas[k]   : interpolation fraction
        f[k]        : class logit  f(x_k)
        gdotd[k]    : directional derivative grad f(x_k) . dir   (= "d_k"/N up to scale)
        curv[k]     : second-order term  (1/2) dir^T H(x_k) dir   via HVP
        rho[k]      : curv / gdotd                                [Eq. 11]
        df[k]       : actual output change f(x_{k+1}) - f(x_k)    (per interval)

    'dir' is the path direction (image - baseline) expressed in the *normalized*
    model-input space, since gradients/Hessian are taken wrt the normalized input.
    """
    if baseline is None:
        baseline = np.zeros_like(image)

    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    path = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]

    # path direction in normalized input space, shape (3,H,W)
    dir_np = (image - baseline).transpose(2, 0, 1) / IMAGENET_STD[:, None, None]
    dir_t = torch.tensor(dir_np, dtype=torch.float32, device=DEVICE)

    f = np.empty(steps, np.float32)
    gdotd = np.empty(steps, np.float32)
    curv = np.empty(steps, np.float32)

    for k in range(steps):
        x = to_model_tensor(path[k:k + 1]).to(DEVICE).requires_grad_(True)
        logit = model(x)[0, label]

        # first-order: grad, then directional derivative <grad, dir>
        (g,) = torch.autograd.grad(logit, x, create_graph=True)
        dd = (g * dir_t[None]).sum()                  # grad f . dir  (scalar)

        # second-order: Hessian-vector product H @ dir = grad_x (grad f . dir)
        (hvp,) = torch.autograd.grad(dd, x, retain_graph=False)
        curv_k = 0.5 * (hvp * dir_t[None]).sum()      # (1/2) dir^T H dir

        f[k] = float(logit.detach())
        gdotd[k] = float(dd.detach())
        curv[k] = float(curv_k.detach())

    # guard the denominator: rho defined where |grad f . dir| is not tiny
    denom = gdotd.copy()
    eps = 1e-3 * (np.abs(gdotd).max() + 1e-12)
    safe = np.abs(denom) > eps
    rho = np.full(steps, np.nan, np.float32)
    rho[safe] = curv[safe] / denom[safe]

    df = np.diff(f)                                    # per-interval actual change
    return alphas, f, gdotd, curv, rho, df


# --------------------------------------------------------------------------- #
# Measure QP:  min_mu  -lambda sum mu_k a_k + (tau/2)||mu||^2,  mu>=0, sum=1
# closed-form projected solution (water-filling on the simplex)
# --------------------------------------------------------------------------- #
def solve_measure_qp(a, lam, tau):
    """
    Exact minimiser of a strictly convex QP over the probability simplex.
    Stationarity: mu_k = (lambda a_k - eta)/tau on the active set, then project.
    Returns mu (sums to 1, mu>=0).
    """
    a = np.asarray(a, np.float64)
    n = len(a)
    # candidate (unconstrained-in-eta) value; solve for eta so that sum mu = 1
    # mu_k = max(0, (lambda a_k - eta)/tau). Find eta by sorting (water-filling).
    s = lam * a
    order = np.argsort(s)[::-1]
    s_sorted = s[order]
    # try active sets of decreasing size
    prefix = np.cumsum(s_sorted)
    mu = None
    for m in range(1, n + 1):
        eta = (prefix[m - 1] - tau) / m          # from sum_{active}(s_k - eta)/tau = 1
        if m == n or s_sorted[m - 1] - eta > 0 >= s_sorted[m] - eta if m < n else True:
            mu_sorted = np.maximum(0.0, (s_sorted - eta) / tau)
            mu_full = np.zeros(n)
            mu_full[order] = mu_sorted
            mu = mu_full
            break
    if mu is None:                               # fallback: uniform
        mu = np.full(n, 1.0 / n)
    # numerical clean-up
    mu = np.maximum(mu, 0)
    mu = mu / mu.sum()
    return mu


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

    alphas, f, gdotd, curv, rho, df = path_quantities(model, image, label)
    a_mid = 0.5 * (alphas[:-1] + alphas[1:])          # interval midpoints

    # per-interval signal |grad f . dir| (use node value at left endpoint)
    sig = np.abs(gdotd[:-1])
    # |d_k| proxy for the measure QP: directional derivative * step length (uniform)
    dk = np.abs(gdotd[:-1]) / (len(alphas) - 1)

    # ----- figure layout: image + 3 term panels -----
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0],
                  hspace=0.42, wspace=0.30)

    # (0) input image -------------------------------------------------------- #
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"input  ({categories[label]})", fontsize=11)
    ax_img.axis("off")

    # (1) PATH DISTORTION: rho(t) ------------------------------------------- #
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.plot(alphas, rho, color="#b5179e", lw=2.0, label=r"$\rho(t)$ (relative lin. error)")
    # the Euler-Lagrange target: constant rho (here: mean over defined steps)
    rbar = np.nanmean(rho)
    ax1.axhline(rbar, color="#0a7d24", ls="--", lw=1.6,
                label=r"E-L target  $\rho=$const")
    ax1.fill_between(alphas, 0, rho, where=~np.isnan(rho), color="#b5179e", alpha=0.12)
    ax1.set_title(r"(1) path distortion  $\int\rho(t)^2\,dt$"
                  "\npenalise curvature the straight line ignores", fontsize=10)
    ax1.set_xlabel(r"$\alpha$  (0 = baseline $\to$ 1 = input)")
    ax1.set_ylabel(r"$\rho$")
    ax1.grid(alpha=0.22)
    ax1.legend(fontsize=8, frameon=False, loc="upper right")

    # (2) SIGNAL HARVESTING: |grad f . gdot| -------------------------------- #
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(a_mid, sig, color="#1f4e8c", lw=2.0)
    ax2.fill_between(a_mid, 0, sig, color="#1f4e8c", alpha=0.15)
    # mark the transition region (where most signal lives)
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
    width = (a_mid[1] - a_mid[0]) if len(a_mid) > 1 else 0.01
    # uniform reference
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