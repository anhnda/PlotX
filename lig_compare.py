"""
Compare the straight-line IG path against an LIG-optimized (gamma, mu) on a real
image, showing the three action terms for BOTH so the optimisation effect is visible.

Straight line  : gamma = line,           mu = uniform   (Standard IG)
LIG-optimized  : gamma = grouped-velocity path (defer high-|grad| dims, Algorithm 1
                 Phase 2 heuristic),      mu = convex-QP optimum (Phase 1, exact)

What to look for
----------------
Panel (1) rho(t): the straight line gives a spiky, non-constant rho (it ignores
curvature). The LIG path aims to FLATTEN rho toward the Euler-Lagrange target
rho = const -- spreading linearisation error evenly. We print std(rho) for both;
lower std = closer to the E-L target. On small-kappa images (church) the straight
line is already decent, so the gain is modest and the figure reports it honestly.

Panel (2) |grad f . gdot|: signal profile along each path.
Panel (3) mu: uniform (IG) vs QP-optimal mu on each path.

rho here uses the residual definition (Eqs. 4-6), not an analytic Hessian, because
ResNet-50 is piecewise-linear (ReLU) and an analytic HVP is ~0.

Edit IMAGE_PATH. Run: python lig_compare.py
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

STEPS = 64
LAMBDA = 1.0
TAU = 0.05                 # single tau for the optimized measure in the comparison
RHO_CLIP = 3.0

N_GROUPS = 16              # feature groups for the velocity schedule (Algorithm 1)
PATH_ITERS = 6            # outer alternating rounds
DEFER_STRENGTH = 2.0      # how aggressively high-|grad| groups are pushed to late steps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# --------------------------------------------------------------------------- #
# Model / preprocessing
# --------------------------------------------------------------------------- #
def load_model():
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(DEVICE).eval()
    return model, weights


def preprocess(pil_img):
    tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
    img = tf(pil_img.convert("RGB"))
    return np.asarray(img).astype(np.float32) / 255.0


def to_model_tensor(batch_hwc):
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
# Evaluate f and grad along an ARBITRARY discrete path (list of HWC nodes)
# --------------------------------------------------------------------------- #
def eval_path(model, nodes_hwc, label):
    """
    nodes_hwc: (N, H, W, C) float[0,1] path nodes (node 0 = baseline, node N-1 = input).
    Returns f (N,), and grads_norm (N, 3, H, W) gradient wrt the *normalized* input.
    """
    N, H, W, C = nodes_hwc.shape
    f = np.empty(N, np.float32)
    grads = np.empty((N, C, H, W), np.float32)            # gradient wrt normalized input
    for k in range(N):
        x = to_model_tensor(nodes_hwc[k:k + 1]).to(DEVICE).requires_grad_(True)
        logit = model(x)[0, label]
        (g,) = torch.autograd.grad(logit, x)
        f[k] = float(logit.detach())
        grads[k] = g.detach().cpu().numpy()[0]
    return f, grads


def interval_quantities(f, grads, disp_norm):
    """
    Given node f (N,), node grads (N,3,H,W) wrt normalized input, and the per-interval
    displacement in normalized space disp_norm (N-1, 3, H, W), compute:
        a_mid, d_k, df_k, rho_k, sig_k (=|grad.disp| at left node), dk=|d_k|
    Left-endpoint convention.
    """
    N = len(f)
    n_int = N - 1
    # directional derivative at left node along that interval's displacement
    gdotd = np.einsum('kchw,kchw->k', grads[:-1], disp_norm)   # (n_int,)
    d = gdotd.copy()                                           # already the interval change
    df = np.diff(f)
    r = df - d
    eps = 1e-3 * (np.abs(d).max() + 1e-12)
    safe = np.abs(d) > eps
    rho = np.full(n_int, np.nan, np.float32)
    rho[safe] = r[safe] / d[safe]
    sig = np.abs(gdotd)
    return d, df, rho, sig, safe


# --------------------------------------------------------------------------- #
# Measure QP (water-filling on the simplex)
# --------------------------------------------------------------------------- #
def solve_measure_qp(a, lam, tau):
    a = np.asarray(a, np.float64)
    n = len(a)
    s = lam * a
    order = np.argsort(s)[::-1]
    s_sorted = s[order]
    prefix = np.cumsum(s_sorted)
    mu_full = None
    for m in range(1, n + 1):
        eta = (prefix[m - 1] - tau) / m
        smallest_active = s_sorted[m - 1] - eta
        next_inactive = (s_sorted[m] - eta) if m < n else -np.inf
        if smallest_active > 0 and next_inactive <= 0:
            mu_sorted = np.maximum(0.0, (s_sorted - eta) / tau)
            mu_full = np.zeros(n)
            mu_full[order] = mu_sorted
            break
    if mu_full is None:
        mu_full = np.full(n, 1.0 / n)
    mu_full = np.maximum(mu_full, 0.0)
    mu_full /= mu_full.sum()
    return mu_full


# --------------------------------------------------------------------------- #
# LIG path: grouped-velocity schedule that defers high-|grad| groups to late steps
# (Algorithm 1, Phase 2 heuristic, simplified to a deterministic schedule)
# --------------------------------------------------------------------------- #
def build_velocity_schedule(group_importance, steps, strength):
    """
    group_importance: (G,) nonneg, larger = higher gradient = deliver LATER.
    Returns V (G, n_int) nonneg velocities; row g sums to 1 (fraction of that
    group's displacement delivered at each interval). High-importance groups put
    more mass near the end; low-importance groups near the start.
    """
    n_int = steps - 1
    G = len(group_importance)
    t = (np.arange(n_int) + 0.5) / n_int                  # interval midpoints in (0,1)
    imp = group_importance / (group_importance.max() + 1e-12)   # in [0,1]
    V = np.empty((G, n_int), np.float64)
    for g in range(G):
        # high imp -> weight concentrated near t=1 ; low imp -> near t=0
        # use a power profile whose center shifts with importance
        center = imp[g]                                   # in [0,1]
        w = np.exp(-strength * (t - center) ** 2 / 0.05)  # gaussian bump around center
        if w.sum() < 1e-12:
            w = np.ones(n_int)
        V[g] = w / w.sum()
    return V


def construct_lig_path(baseline, image, groups, V):
    """
    baseline,image: (H,W,C). groups: (H*W*C,) int group id per feature. V: (G,n_int).
    Returns nodes_hwc (N,H,W,C) and disp_norm (n_int,3,H,W) in normalized space.
    Feature i moves by total (image-image_baseline)_i, delivered across intervals
    in proportion to V[group(i)].
    """
    H, W, C = image.shape
    n_int = V.shape[1]
    steps = n_int + 1
    total = (image - baseline).reshape(-1)                # (P,)
    P = total.size
    # cumulative fraction delivered by end of each interval, per feature
    Vfeat = V[groups]                                     # (P, n_int)
    cumfrac = np.cumsum(Vfeat, axis=1)                    # (P, n_int), ends at 1
    cumfrac = cumfrac / cumfrac[:, -1:]                   # normalize to end at 1
    frac_nodes = np.concatenate([np.zeros((P, 1)), cumfrac], axis=1)  # (P, steps)

    base_flat = baseline.reshape(-1)
    nodes = base_flat[:, None] + frac_nodes * total[:, None]          # (P, steps)
    nodes_hwc = nodes.T.reshape(steps, H, W, C).astype(np.float32)
    nodes_hwc = np.clip(nodes_hwc, 0.0, 1.0)

    # per-interval displacement in normalized space (3,H,W) per interval
    disp = np.diff(nodes_hwc, axis=0)                     # (n_int,H,W,C)
    disp_norm = (disp / IMAGENET_STD[None, None, None, :]).transpose(0, 3, 1, 2)
    return nodes_hwc, disp_norm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    label = int(predict_logits(model, image[None])[0].argmax())
    print(f"Predicted: {categories[label]}")
    H, W, C = image.shape
    baseline = np.zeros_like(image)

    alphas = np.linspace(0, 1, STEPS, dtype=np.float32)
    a_mid = 0.5 * (alphas[:-1] + alphas[1:])

    # ---------------- straight-line path ---------------- #
    line_nodes = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]
    line_nodes = line_nodes.astype(np.float32)
    disp_line = np.diff(line_nodes, axis=0)
    disp_line_norm = (disp_line / IMAGENET_STD[None, None, None, :]).transpose(0, 3, 1, 2)
    f_line, g_line = eval_path(model, line_nodes, label)
    d_l, df_l, rho_l, sig_l, safe_l = interval_quantities(f_line, g_line, disp_line_norm)
    mu_line_unif = np.full(STEPS - 1, 1.0 / (STEPS - 1))

    # ---------------- LIG path (alternating: build path from gradients, then QP mu) --- #
    # group features by |grad f(input)| (Algorithm 1 init)
    x_in = to_model_tensor(image[None]).to(DEVICE).requires_grad_(True)
    logit_in = model(x_in)[0, label]
    (g_in,) = torch.autograd.grad(logit_in, x_in)
    gmag = np.abs(g_in.detach().cpu().numpy()[0]).transpose(1, 2, 0).reshape(-1)  # (P,)
    # quantile-based group ids: high gmag -> high group id (deferred later)
    order = np.argsort(gmag)
    groups = np.empty_like(order)
    groups[order] = (np.arange(len(order)) * N_GROUPS / len(order)).astype(int)
    groups = np.clip(groups, 0, N_GROUPS - 1)
    # group importance = mean |grad| in group
    group_imp = np.array([gmag[groups == g].mean() if np.any(groups == g) else 0.0
                          for g in range(N_GROUPS)])

    V = build_velocity_schedule(group_imp, STEPS, DEFER_STRENGTH)
    lig_nodes, disp_lig_norm = construct_lig_path(baseline, image, groups, V)

    best = None
    for it in range(PATH_ITERS):
        f_lig, g_lig = eval_path(model, lig_nodes, label)
        d_g, df_g, rho_g, sig_g, safe_g = interval_quantities(f_lig, g_lig, disp_lig_norm)
        std_rho = np.nanstd(rho_g)
        if best is None or std_rho < best[0]:
            best = (std_rho, f_lig, d_g, df_g, rho_g, sig_g, safe_g, lig_nodes, disp_lig_norm)
        # (kept deterministic; schedule fixed. Loop here is a hook for future
        #  stochastic refinement; we keep the best-std path.)
        break

    std_rho, f_lig, d_g, df_g, rho_g, sig_g, safe_g, lig_nodes, disp_lig_norm = best
    dk_lig = np.abs(d_g)
    mu_lig = solve_measure_qp(dk_lig, LAMBDA, TAU)

    # diagnostics
    print(f"[straight] f(0)={f_line[0]:.3f} f(1)={f_line[-1]:.3f} "
          f"std(rho)={np.nanstd(rho_l):.3f}")
    print(f"[LIG]      f(0)={f_lig[0]:.3f} f(1)={f_lig[-1]:.3f} "
          f"std(rho)={np.nanstd(rho_g):.3f}")

    rho_l_disp = np.clip(rho_l, -RHO_CLIP, RHO_CLIP)
    rho_g_disp = np.clip(rho_g, -RHO_CLIP, RHO_CLIP)

    # ---------------- figure ---------------- #
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0], hspace=0.46, wspace=0.30)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"input  ({categories[label]})", fontsize=11)
    ax_img.axis("off")

    # (1) rho: straight vs LIG
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.plot(a_mid, rho_l_disp, color="#c77dff", lw=1.5, alpha=0.9,
             label=fr"straight  (std={np.nanstd(rho_l):.2f})")
    ax1.plot(a_mid, rho_g_disp, color="#7b2cbf", lw=2.0,
             label=fr"LIG  (std={np.nanstd(rho_g):.2f})")
    ax1.axhline(np.nanmean(rho_g), color="#0a7d24", ls="--", lw=1.4,
                label=r"E-L target  $\rho=$const")
    ax1.set_title(r"(1) path distortion  $\int\rho^2\,dt$"
                  "\nLIG flattens $\\rho$ toward const", fontsize=10)
    ax1.set_xlabel(r"$\alpha$"); ax1.set_ylabel(r"$\rho$ (clip $\pm$%.0f)" % RHO_CLIP)
    ax1.set_ylim(-RHO_CLIP * 1.1, RHO_CLIP * 1.1)
    ax1.grid(alpha=0.22); ax1.legend(fontsize=8, frameon=False, loc="upper right")

    # (2) signal: straight vs LIG
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(a_mid, sig_l, color="#90b4d6", lw=1.6, label="straight")
    ax2.plot(a_mid, sig_g, color="#1f4e8c", lw=2.0, label="LIG")
    ax2.fill_between(a_mid, 0, sig_g, color="#1f4e8c", alpha=0.10)
    ax2.set_title(r"(2) signal harvesting  $-\lambda\int|\nabla f\cdot\dot\gamma|\mu\,dt$"
                  "\nsignal profile along each path", fontsize=10)
    ax2.set_xlabel(r"$\alpha$"); ax2.set_ylabel(r"$|\nabla f\cdot\dot\gamma|$")
    ax2.grid(alpha=0.22); ax2.legend(fontsize=8, frameon=False, loc="upper right")

    # (3) measure: uniform (IG) vs QP-optimal on LIG path
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axhline(1.0 / (STEPS - 1), color="0.5", ls=":", lw=1.4,
                label=r"uniform  $\mu=1/N$  (Standard IG)")
    ax3.plot(a_mid, solve_measure_qp(np.abs(d_l), LAMBDA, TAU), color="#c77dff",
             lw=1.5, marker="o", ms=2.2, alpha=0.9,
             label=fr"QP $\mu^\star$ on straight path ($\tau={TAU:g}$)")
    ax3.plot(a_mid, mu_lig, color="#7b2cbf", lw=2.0, marker="o", ms=2.5,
             label=fr"QP $\mu^\star$ on LIG path ($\tau={TAU:g}$)")
    ax3.set_title(r"(3) measure regulariser  $\frac{\tau}{2}\int\mu^2\,dt$"
                  "   —   uniform (IG) vs joint $(\\gamma,\\mu)$ optimum (LIG)", fontsize=10)
    ax3.set_xlabel(r"$\alpha$  (0 = baseline $\to$ 1 = input)")
    ax3.set_ylabel(r"measure weight  $\mu_k$")
    ax3.grid(alpha=0.22); ax3.legend(fontsize=9, frameon=False, ncol=3, loc="upper center")

    fig.suptitle(r"Straight-line IG  vs  LIG-optimized $(\gamma,\mu)$  —  three action terms",
                 fontsize=13, y=0.98)

    plt.savefig("lig_compare.png", bbox_inches="tight", dpi=150)
    print("Saved: lig_compare.png")


if __name__ == "__main__":
    main()