"""
Straight-line IG  vs  LIG-optimized (gamma, mu)  -- three action terms.

Follows the reference lig.py logic:
  * objective = Var_nu(phi) - lam * sum_k mu_k |d_k| + (tau/2)||mu||^2
    (this is what LIG actually minimises; we report Var_nu(phi) as the
     path-distortion diagnostic, NOT raw std(rho) which degenerate steps inflate)
  * alternating minimisation: Phase 1 optimise mu (QP), Phase 2 optimise path
  * REGRESSION GUARD: a new path is accepted only if the objective improves;
    otherwise we keep the straight line. On small-kappa images (church) this
    means LIG may barely move -- and the figure says so honestly.

phi_k = d_k / df_k,  nu_k = mu_k df_k^2 / sum_j mu_j df_j^2,
Var_nu(phi) = sum_k nu_k (phi_k - phibar)^2,  phibar = sum_k nu_k phi_k.

rho here uses the residual definition (Eqs. 4-6); ResNet-50 is ReLU so an analytic
Hessian-vector product is ~0 and useless. rho is shown (clipped) only as a visual;
the load-bearing number is Var_nu(phi).

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

STEPS = 64                 # N+1 nodes -> N intervals
LAMBDA = 1.0
TAU = 0.01                 # matches reference default
RHO_CLIP = 3.0

N_GROUPS = 16              # spatial feature groups (Algorithm 1)
N_ALTERNATING = 4          # outer alternating rounds (mu <-> path)
DEFER_STRENGTHS = [0.0, 1.0, 2.0, 3.5]  # path candidates per round (0 = straight)

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
# Evaluate f and grad along a discrete path of HWC nodes
# --------------------------------------------------------------------------- #
def eval_path(model, nodes_hwc, label):
    N, H, W, C = nodes_hwc.shape
    f = np.empty(N, np.float32)
    grads = np.empty((N, C, H, W), np.float32)
    for k in range(N):
        x = to_model_tensor(nodes_hwc[k:k + 1]).to(DEVICE).requires_grad_(True)
        logit = model(x)[0, label]
        (g,) = torch.autograd.grad(logit, x)
        f[k] = float(logit.detach())
        grads[k] = g.detach().cpu().numpy()[0]
    return f, grads


def interval_quantities(f, grads, disp_norm):
    """Left-endpoint convention. Returns d, df, rho, sig, safe (per interval)."""
    n_int = len(f) - 1
    gdotd = np.einsum('kchw,kchw->k', grads[:-1], disp_norm)   # grad.disp = d_k
    d = gdotd.copy()
    df = np.diff(f)
    r = df - d
    eps = 1e-3 * (np.abs(d).max() + 1e-12)
    safe = np.abs(d) > eps
    rho = np.full(n_int, np.nan, np.float32)
    rho[safe] = r[safe] / d[safe]
    sig = np.abs(gdotd)
    return d, df, rho, sig, safe


# --------------------------------------------------------------------------- #
# Paper objective pieces:  phi, nu, Var_nu(phi), full objective
# --------------------------------------------------------------------------- #
def phi_and_varnu(d, df, mu):
    """
    phi_k = d_k/df_k (where df!=0); nu_k = mu_k df_k^2 / sum; Var_nu(phi).
    Flat steps (df~0) get zero effective weight, exactly as in the paper.
    """
    d = np.asarray(d, np.float64); df = np.asarray(df, np.float64)
    mu = np.asarray(mu, np.float64)
    nz = np.abs(df) > 1e-8 * (np.abs(df).max() + 1e-12)
    phi = np.zeros_like(d)
    phi[nz] = d[nz] / df[nz]
    w = mu * df ** 2
    if w.sum() <= 0:
        return phi, 0.0
    nu = w / w.sum()
    phibar = float((nu * phi).sum())
    var = float((nu * (phi - phibar) ** 2).sum())
    return phi, var


def full_objective(d, df, mu, lam, tau):
    """Var_nu(phi) - lam sum mu_k|d_k| + (tau/2)||mu||^2."""
    _, var = phi_and_varnu(d, df, mu)
    mu = np.asarray(mu, np.float64)
    sig = float((mu * np.abs(d)).sum())
    reg = 0.5 * tau * float((mu ** 2).sum())
    return var - lam * sig + reg, var


# --------------------------------------------------------------------------- #
# Phase 1: measure QP  min -lam sum mu_k|d_k| + (tau/2)||mu||^2  on the simplex
# closed-form water-filling (the Var_nu term is treated as fixed wrt mu at the
# leading order used by the reference's QP step)
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
# Phase 2: build candidate path via grouped-velocity schedule (defer high-|grad|)
# --------------------------------------------------------------------------- #
def build_velocity_schedule(group_importance, steps, strength):
    n_int = steps - 1
    G = len(group_importance)
    t = (np.arange(n_int) + 0.5) / n_int
    imp = group_importance / (group_importance.max() + 1e-12)
    V = np.empty((G, n_int), np.float64)
    for g in range(G):
        center = imp[g]                       # high grad -> deliver later (center ~1)
        w = np.exp(-strength * (t - center) ** 2 / 0.05)
        if w.sum() < 1e-12:
            w = np.ones(n_int)
        V[g] = w / w.sum()
    return V


def construct_path(baseline, image, groups, V):
    H, W, C = image.shape
    n_int = V.shape[1]; steps = n_int + 1
    total = (image - baseline).reshape(-1)
    P = total.size
    Vfeat = V[groups]
    cumfrac = np.cumsum(Vfeat, axis=1)
    cumfrac = cumfrac / cumfrac[:, -1:]
    frac_nodes = np.concatenate([np.zeros((P, 1)), cumfrac], axis=1)
    base_flat = baseline.reshape(-1)
    nodes = base_flat[:, None] + frac_nodes * total[:, None]
    nodes_hwc = np.clip(nodes.T.reshape(steps, H, W, C).astype(np.float32), 0, 1)
    disp = np.diff(nodes_hwc, axis=0)
    disp_norm = (disp / IMAGENET_STD[None, None, None, :]).transpose(0, 3, 1, 2)
    return nodes_hwc, disp_norm


def straight_path(baseline, image, steps):
    alphas = np.linspace(0, 1, steps, dtype=np.float32)
    nodes = (baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]).astype(np.float32)
    disp = np.diff(nodes, axis=0)
    disp_norm = (disp / IMAGENET_STD[None, None, None, :]).transpose(0, 3, 1, 2)
    return nodes, disp_norm


# --------------------------------------------------------------------------- #
# Main: alternating minimisation with regression guard (reference logic)
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    label = int(predict_logits(model, image[None])[0].argmax())
    print(f"Predicted: {categories[label]}")
    baseline = np.zeros_like(image)
    alphas = np.linspace(0, 1, STEPS, dtype=np.float32)
    a_mid = 0.5 * (alphas[:-1] + alphas[1:])

    # group features by |grad f(input)| (Algorithm 1 init)
    x_in = to_model_tensor(image[None]).to(DEVICE).requires_grad_(True)
    logit_in = model(x_in)[0, label]
    (g_in,) = torch.autograd.grad(logit_in, x_in)
    gmag = np.abs(g_in.detach().cpu().numpy()[0]).transpose(1, 2, 0).reshape(-1)
    order = np.argsort(gmag)
    groups = np.empty_like(order)
    groups[order] = (np.arange(len(order)) * N_GROUPS / len(order)).astype(int)
    groups = np.clip(groups, 0, N_GROUPS - 1)
    group_imp = np.array([gmag[groups == g].mean() if np.any(groups == g) else 0.0
                          for g in range(N_GROUPS)])

    # ---- straight-line reference (Standard IG state) ----
    nodes_line, disp_line = straight_path(baseline, image, STEPS)
    f_line, g_line = eval_path(model, nodes_line, label)
    d_l, df_l, rho_l, sig_l, _ = interval_quantities(f_line, g_line, disp_line)
    mu_unif = np.full(STEPS - 1, 1.0 / (STEPS - 1))
    _, var_line_unif = phi_and_varnu(d_l, df_l, mu_unif)

    # ============ ALTERNATING MINIMISATION with REGRESSION GUARD ============ #
    # state initialised at the straight line + uniform mu
    cur_nodes, cur_disp = nodes_line, disp_line
    cur_f, cur_g = f_line, g_line
    cur_d, cur_df = d_l, df_l
    cur_mu = mu_unif.copy()
    best_obj, best_var = full_objective(cur_d, cur_df, cur_mu, LAMBDA, TAU)
    best = dict(nodes=cur_nodes, disp=cur_disp, f=cur_f, g=cur_g,
                d=cur_d, df=cur_df, mu=cur_mu, var=best_var, rho=rho_l, sig=sig_l)
    print(f"[init/straight] Var_nu(phi)={best_var:.4f}  obj={best_obj:.4f}")

    for s in range(N_ALTERNATING):
        # ---- Phase 1: optimise mu on the CURRENT best path ----
        mu_new = solve_measure_qp(np.abs(best['d']), LAMBDA, TAU)
        obj1, var1 = full_objective(best['d'], best['df'], mu_new, LAMBDA, TAU)
        if obj1 <= best_obj + 1e-12:
            best_obj, best_var = obj1, var1
            best['mu'] = mu_new; best['var'] = var1

        # ---- Phase 2: try path candidates, accept only if objective improves ----
        for strength in DEFER_STRENGTHS:
            if strength == 0.0:
                continue  # that's the straight line, already the baseline state
            V = build_velocity_schedule(group_imp, STEPS, strength)
            cand_nodes, cand_disp = construct_path(baseline, image, groups, V)
            cf, cg = eval_path(model, cand_nodes, label)
            cd, cdf, crho, csig, _ = interval_quantities(cf, cg, cand_disp)
            # re-optimise mu for this candidate, then score
            cmu = solve_measure_qp(np.abs(cd), LAMBDA, TAU)
            cobj, cvar = full_objective(cd, cdf, cmu, LAMBDA, TAU)
            if cobj < best_obj:          # REGRESSION GUARD
                best_obj, best_var = cobj, cvar
                best = dict(nodes=cand_nodes, disp=cand_disp, f=cf, g=cg,
                            d=cd, df=cdf, mu=cmu, var=cvar, rho=crho, sig=csig)
        print(f"[round {s}] best Var_nu(phi)={best_var:.4f}  obj={best_obj:.4f}")

    # final LIG state
    lig = best
    _, var_lig = phi_and_varnu(lig['d'], lig['df'], lig['mu'])
    moved = not np.allclose(lig['nodes'], nodes_line)
    print(f"[straight] Var_nu(phi) (uniform mu) = {var_line_unif:.4f}")
    print(f"[LIG]      Var_nu(phi) = {var_lig:.4f}   path moved: {moved}")

    rho_l_disp = np.clip(rho_l, -RHO_CLIP, RHO_CLIP)
    rho_g_disp = np.clip(lig['rho'], -RHO_CLIP, RHO_CLIP)

    # --------------------------- figure --------------------------- #
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0], hspace=0.46, wspace=0.30)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"input  ({categories[label]})", fontsize=11)
    ax_img.axis("off")

    # (1) path distortion: rho (visual) + Var_nu(phi) (load-bearing number)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.plot(a_mid, rho_l_disp, color="#c77dff", lw=1.4, alpha=0.9, label="straight")
    ax1.plot(a_mid, rho_g_disp, color="#7b2cbf", lw=2.0, label="LIG")
    ax1.axhline(1.0, color="#0a7d24", ls="--", lw=1.4, label=r"target $\phi=1\ (\rho=0)$")
    ax1.set_title(r"(1) path distortion  $\int\rho^2\,dt$"
                  "\n" fr"$\mathrm{{Var}}_\nu(\phi)$:  straight={var_line_unif:.3f}  $\to$  LIG={var_lig:.3f}",
                  fontsize=10)
    ax1.set_xlabel(r"$\alpha$"); ax1.set_ylabel(r"$\rho$ (clip $\pm$%.0f)" % RHO_CLIP)
    ax1.set_ylim(-RHO_CLIP * 1.1, RHO_CLIP * 1.1)
    ax1.grid(alpha=0.22); ax1.legend(fontsize=8, frameon=False, loc="upper right")

    # (2) signal harvesting
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(a_mid, sig_l, color="#90b4d6", lw=1.6, label="straight")
    ax2.plot(a_mid, lig['sig'], color="#1f4e8c", lw=2.0, label="LIG")
    ax2.fill_between(a_mid, 0, lig['sig'], color="#1f4e8c", alpha=0.10)
    ax2.set_title(r"(2) signal harvesting  $-\lambda\int|\nabla f\cdot\dot\gamma|\mu\,dt$"
                  "\nsignal profile along each path", fontsize=10)
    ax2.set_xlabel(r"$\alpha$"); ax2.set_ylabel(r"$|\nabla f\cdot\dot\gamma|$")
    ax2.grid(alpha=0.22); ax2.legend(fontsize=8, frameon=False, loc="upper right")

    # (3) measure
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axhline(1.0 / (STEPS - 1), color="0.5", ls=":", lw=1.4,
                label=r"uniform  $\mu=1/N$  (Standard IG)")
    ax3.plot(a_mid, solve_measure_qp(np.abs(d_l), LAMBDA, TAU), color="#c77dff",
             lw=1.4, marker="o", ms=2.2, alpha=0.9,
             label=fr"$\mu^\star$ on straight path")
    ax3.plot(a_mid, lig['mu'], color="#7b2cbf", lw=2.0, marker="o", ms=2.5,
             label=fr"$\mu^\star$ on LIG path")
    title3 = (r"(3) measure regulariser  $\frac{\tau}{2}\int\mu^2\,dt$"
              "   —   uniform (IG) vs joint $(\\gamma,\\mu)$ optimum (LIG)")
    if not moved:
        title3 += "\n[regression guard: LIG path = straight on this image; only $\\mu$ changed]"
    ax3.set_title(title3, fontsize=10)
    ax3.set_xlabel(r"$\alpha$  (0 = baseline $\to$ 1 = input)")
    ax3.set_ylabel(r"measure weight  $\mu_k$")
    ax3.grid(alpha=0.22); ax3.legend(fontsize=9, frameon=False, ncol=3, loc="upper center")

    fig.suptitle(r"Straight-line IG  vs  LIG-optimized $(\gamma,\mu)$  —  three action terms"
                 r"  (objective = $\mathrm{Var}_\nu(\phi)$, with regression guard)",
                 fontsize=12.5, y=0.98)

    plt.savefig("lig_compare.png", bbox_inches="tight", dpi=150)
    print("Saved: lig_compare.png")


if __name__ == "__main__":
    main()