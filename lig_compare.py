"""
Straight-line IG  vs  LIG-optimized (gamma, mu)  -- action terms + FINAL attributions.

Follows the reference lig.py logic:
  * objective = Var_nu(phi) - lam * sum_k mu_k |d_k| + (tau/2)||mu||^2
  * alternating minimisation: Phase 1 optimise mu (QP), Phase 2 optimise path
  * REGRESSION GUARD: a path candidate is accepted only if the objective improves.

Top row  : input + the three action-term diagnostics (rho, signal, ... )
Bottom   : the THING THAT MATTERS -- the final attribution maps of both methods,
           side by side, with insertion/deletion AUC so the comparison is
           quantitative, not just visual.

Baseline note: "zeros" (black) is small-kappa on church and the straight line is
already near-optimal, so LIG barely moves. "noise" pushes the straight line
through off-manifold / higher-curvature regions (larger kappa), giving the joint
(gamma,mu) update something real to fix -- the regime where the paper reports gains.

rho uses the residual definition (Eqs. 4-6); ResNet-50 is ReLU so an analytic
Hessian-vector product is ~0. The load-bearing path-distortion number is Var_nu(phi).

Edit IMAGE_PATH / BASELINE. Run: python lig_compare.py
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
IMAGE_PATH = "frenchh.JPEG"

STEPS = 64
LAMBDA = 1.0
TAU = 0.01
RHO_CLIP = 3.0

N_GROUPS = 16
N_ALTERNATING = 4
DEFER_STRENGTHS = [0.0, 1.0, 2.0, 3.5]

BASELINE = "noise"         # "zeros" | "noise"
NOISE_STD = 1.0

# attribution-metric config (insertion/deletion)
METRIC_STEPS = 50          # number of pixel-reveal steps for AUC
METRIC_BASELINE = "black"  # what removed pixels are set to

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


@torch.no_grad()
def softmax_prob(model, batch_hwc, label, bs=64):
    out = []
    for i in range(0, len(batch_hwc), bs):
        chunk = to_model_tensor(batch_hwc[i:i + bs]).to(DEVICE)
        p = torch.softmax(model(chunk), dim=1)[:, label]
        out.append(p.cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Path eval + interval quantities
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
    n_int = len(f) - 1
    gdotd = np.einsum('kchw,kchw->k', grads[:-1], disp_norm)
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
# Objective pieces
# --------------------------------------------------------------------------- #
def phi_and_varnu(d, df, mu):
    d = np.asarray(d, np.float64); df = np.asarray(df, np.float64); mu = np.asarray(mu, np.float64)
    nz = np.abs(df) > 1e-8 * (np.abs(df).max() + 1e-12)
    phi = np.zeros_like(d); phi[nz] = d[nz] / df[nz]
    w = mu * df ** 2
    if w.sum() <= 0:
        return phi, 0.0
    nu = w / w.sum()
    phibar = float((nu * phi).sum())
    var = float((nu * (phi - phibar) ** 2).sum())
    return phi, var


def full_objective(d, df, mu, lam, tau):
    _, var = phi_and_varnu(d, df, mu)
    mu = np.asarray(mu, np.float64)
    sig = float((mu * np.abs(d)).sum())
    reg = 0.5 * tau * float((mu ** 2).sum())
    return var - lam * sig + reg, var


def solve_measure_qp(a, lam, tau):
    a = np.asarray(a, np.float64); n = len(a)
    s = lam * a
    order = np.argsort(s)[::-1]
    s_sorted = s[order]; prefix = np.cumsum(s_sorted)
    mu_full = None
    for m in range(1, n + 1):
        eta = (prefix[m - 1] - tau) / m
        smallest_active = s_sorted[m - 1] - eta
        next_inactive = (s_sorted[m] - eta) if m < n else -np.inf
        if smallest_active > 0 and next_inactive <= 0:
            mu_sorted = np.maximum(0.0, (s_sorted - eta) / tau)
            mu_full = np.zeros(n); mu_full[order] = mu_sorted
            break
    if mu_full is None:
        mu_full = np.full(n, 1.0 / n)
    mu_full = np.maximum(mu_full, 0.0); mu_full /= mu_full.sum()
    return mu_full


# --------------------------------------------------------------------------- #
# Path construction
# --------------------------------------------------------------------------- #
def build_velocity_schedule(group_importance, steps, strength):
    n_int = steps - 1; G = len(group_importance)
    t = (np.arange(n_int) + 0.5) / n_int
    imp = group_importance / (group_importance.max() + 1e-12)
    V = np.empty((G, n_int), np.float64)
    for g in range(G):
        center = imp[g]
        w = np.exp(-strength * (t - center) ** 2 / 0.05)
        if w.sum() < 1e-12:
            w = np.ones(n_int)
        V[g] = w / w.sum()
    return V


def construct_path(baseline, image, groups, V):
    H, W, C = image.shape
    n_int = V.shape[1]; steps = n_int + 1
    total = (image - baseline).reshape(-1); P = total.size
    Vfeat = V[groups]
    cumfrac = np.cumsum(Vfeat, axis=1); cumfrac = cumfrac / cumfrac[:, -1:]
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
# Attribution from a path state (sum_k mu_k * grad_k * step_k), completeness-rescaled
# --------------------------------------------------------------------------- #
def attribution_from_state(state, target_change):
    """
    state: dict with nodes (N,H,W,C), g (N,C,H,W grads), mu (N-1,), disp (N-1,...)
    Returns per-pixel attribution map (H,W) = sum over channels of
      sum_k mu_k * grad_k . step_k   (in normalized space), then completeness-rescaled.
    """
    nodes = state['nodes']; grads = state['g']; mu = state['mu']; disp = state['disp']
    n_int = len(mu)
    # per-feature contribution a_{k,i} = mu_k * grad_k,i * disp_k,i  (normalized space)
    contrib = np.zeros_like(disp[0])                       # (C,H,W)
    for k in range(n_int):
        contrib += mu[k] * grads[k] * disp[k]
    # collapse channels -> per-pixel saliency (signed sum); use abs for ranking maps
    sal = contrib.sum(axis=0)                              # (H,W) signed
    # completeness rescale so total matches f(x)-f(baseline)
    tot = sal.sum()
    if abs(tot) > 1e-8:
        sal = sal * (target_change / tot)
    return sal


# --------------------------------------------------------------------------- #
# Insertion / Deletion AUC on a saliency map (higher Ins, lower Del = better)
# --------------------------------------------------------------------------- #
def insertion_deletion(model, image, label, sal, steps=METRIC_STEPS, mode="black"):
    H, W = sal.shape
    order = np.argsort(np.abs(sal).reshape(-1))[::-1]      # most salient first
    P = H * W
    blackbg = np.zeros_like(image) if mode == "black" else \
              np.full_like(image, 0.5)
    flat_img = image.reshape(P, -1)
    flat_bg = blackbg.reshape(P, -1)

    # insertion: start from background, reveal salient pixels
    ins_imgs = []
    cur = flat_bg.copy()
    ks = np.linspace(0, P, steps + 1).astype(int)
    for j in range(steps + 1):
        if j > 0:
            idx = order[ks[j - 1]:ks[j]]
            cur[idx] = flat_img[idx]
        ins_imgs.append(cur.reshape(image.shape).copy())
    ins_p = softmax_prob(model, np.stack(ins_imgs), label)

    # deletion: start from image, remove salient pixels
    del_imgs = []
    cur = flat_img.copy()
    for j in range(steps + 1):
        if j > 0:
            idx = order[ks[j - 1]:ks[j]]
            cur[idx] = flat_bg[idx]
        del_imgs.append(cur.reshape(image.shape).copy())
    del_p = softmax_prob(model, np.stack(del_imgs), label)

    xs = np.linspace(0, 1, steps + 1)
    ins_auc = float(np.trapz(ins_p, xs))
    del_auc = float(np.trapz(del_p, xs))
    return ins_auc, del_auc, ins_p, del_p


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

    if BASELINE == "noise":
        rng = np.random.default_rng(SEED)
        baseline = np.clip(rng.normal(0.5, NOISE_STD, size=image.shape), 0, 1).astype(np.float32)
    else:
        baseline = np.zeros_like(image)

    f_bl = float(predict_logits(model, baseline[None])[0, label])
    f_x = float(predict_logits(model, image[None])[0, label])
    target_change = f_x - f_bl
    print(f"baseline={BASELINE}  f(baseline)={f_bl:.3f}  f(x)={f_x:.3f}  target={target_change:.3f}")

    alphas = np.linspace(0, 1, STEPS, dtype=np.float32)
    a_mid = 0.5 * (alphas[:-1] + alphas[1:])

    # group features by |grad f(input)|
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

    # ---- straight-line reference ----
    nodes_line, disp_line = straight_path(baseline, image, STEPS)
    f_line, g_line = eval_path(model, nodes_line, label)
    d_l, df_l, rho_l, sig_l, _ = interval_quantities(f_line, g_line, disp_line)
    mu_unif = np.full(STEPS - 1, 1.0 / (STEPS - 1))
    _, var_line_unif = phi_and_varnu(d_l, df_l, mu_unif)

    straight_state = dict(nodes=nodes_line, disp=disp_line, g=g_line,
                          mu=mu_unif, d=d_l, df=df_l, rho=rho_l, sig=sig_l)

    # ============ ALTERNATING + REGRESSION GUARD ============ #
    best = dict(nodes=nodes_line, disp=disp_line, f=f_line, g=g_line,
                d=d_l, df=df_l, mu=mu_unif.copy(), var=var_line_unif,
                rho=rho_l, sig=sig_l)
    best_obj, _ = full_objective(d_l, df_l, mu_unif, LAMBDA, TAU)

    for s in range(N_ALTERNATING):
        mu_new = solve_measure_qp(np.abs(best['d']), LAMBDA, TAU)
        obj1, var1 = full_objective(best['d'], best['df'], mu_new, LAMBDA, TAU)
        if obj1 <= best_obj + 1e-12:
            best_obj = obj1; best['mu'] = mu_new; best['var'] = var1
        for strength in DEFER_STRENGTHS:
            if strength == 0.0:
                continue
            V = build_velocity_schedule(group_imp, STEPS, strength)
            cand_nodes, cand_disp = construct_path(baseline, image, groups, V)
            cf, cg = eval_path(model, cand_nodes, label)
            cd, cdf, crho, csig, _ = interval_quantities(cf, cg, cand_disp)
            cmu = solve_measure_qp(np.abs(cd), LAMBDA, TAU)
            cobj, cvar = full_objective(cd, cdf, cmu, LAMBDA, TAU)
            if cobj < best_obj:
                best_obj = cobj
                best = dict(nodes=cand_nodes, disp=cand_disp, f=cf, g=cg,
                            d=cd, df=cdf, mu=cmu, var=cvar, rho=crho, sig=csig)
        print(f"[round {s}] best Var_nu(phi)={best['var']:.4f}  obj={best_obj:.4f}")

    lig_state = best
    moved = not np.allclose(lig_state['nodes'], nodes_line)
    print(f"[straight] Var_nu={var_line_unif:.4f}   [LIG] Var_nu={lig_state['var']:.4f}   path moved: {moved}")

    # ---- FINAL attributions ----
    sal_straight = attribution_from_state(straight_state, target_change)
    sal_lig = attribution_from_state(lig_state, target_change)

    ins_s, del_s, _, _ = insertion_deletion(model, image, label, sal_straight, mode=METRIC_BASELINE)
    ins_g, del_g, _, _ = insertion_deletion(model, image, label, sal_lig, mode=METRIC_BASELINE)
    print(f"[straight] Ins={ins_s:.3f} Del={del_s:.3f} Ins-Del={ins_s-del_s:.3f}")
    print(f"[LIG]      Ins={ins_g:.3f} Del={del_g:.3f} Ins-Del={ins_g-del_g:.3f}")

    rho_l_disp = np.clip(rho_l, -RHO_CLIP, RHO_CLIP)
    rho_g_disp = np.clip(lig_state['rho'], -RHO_CLIP, RHO_CLIP)

    # absolute saliency overlays, percentile-normalised for display
    def norm_map(m):
        a = np.abs(m)
        hi = np.percentile(a, 99) + 1e-12
        return np.clip(a / hi, 0, 1)
    smap_s = norm_map(sal_straight)
    smap_g = norm_map(sal_lig)

    # --------------------------- figure --------------------------- #
    C_STRAIGHT = "#e08a1e"   # orange
    C_LIG = "#7b2cbf"        # purple

    fig = plt.figure(figsize=(14.5, 11.0))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.15],
                  hspace=0.5, wspace=0.30)

    # row 0: input + rho + signal
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image); ax_img.set_title(f"input  ({categories[label]})", fontsize=11)
    ax_img.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.plot(a_mid, rho_l_disp, color=C_STRAIGHT, lw=1.5, label="straight")
    ax1.plot(a_mid, rho_g_disp, color=C_LIG, lw=2.0, ls="--", label="LIG")
    ax1.axhline(1.0, color="#0a7d24", ls=":", lw=1.4, label=r"$\phi=1$ ($\rho=0$)")
    ax1.set_title(r"(1) path distortion $\int\rho^2 dt$"
                  "\n" fr"$\mathrm{{Var}}_\nu(\phi)$: straight={var_line_unif:.3f} $\to$ LIG={lig_state['var']:.3f}"
                  "\n(y = per-step $\\rho$, clipped; number = whole-path objective)",
                  fontsize=9)
    ax1.set_xlabel(r"$\alpha$"); ax1.set_ylabel(r"$\rho$ (clip $\pm$%.0f)" % RHO_CLIP)
    ax1.set_ylim(-RHO_CLIP * 1.1, RHO_CLIP * 1.1)
    ax1.grid(alpha=0.22); ax1.legend(fontsize=8, frameon=False, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(a_mid, sig_l, color=C_STRAIGHT, lw=1.6, label="straight")
    ax2.plot(a_mid, lig_state['sig'], color=C_LIG, lw=2.0, ls="--", label="LIG")
    ax2.set_title(r"(2) signal harvesting $-\lambda\int|\nabla f\cdot\dot\gamma|\mu\,dt$"
                  "\nsignal profile along each path", fontsize=9)
    ax2.set_xlabel(r"$\alpha$"); ax2.set_ylabel(r"$|\nabla f\cdot\dot\gamma|$")
    ax2.grid(alpha=0.22); ax2.legend(fontsize=8, frameon=False, loc="upper right")

    # row 1: measure (full width)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axhline(1.0 / (STEPS - 1), color="0.5", ls=":", lw=1.4,
                label=r"uniform $\mu=1/N$ (Standard IG)")
    ax3.plot(a_mid, solve_measure_qp(np.abs(d_l), LAMBDA, TAU), color=C_STRAIGHT,
             lw=1.6, marker="o", ms=2.4, label=r"$\mu^\star$ on straight path")
    ax3.plot(a_mid, lig_state['mu'], color=C_LIG, lw=2.0, ls="--", marker="s", ms=2.6,
             label=r"$\mu^\star$ on LIG path")
    t3 = r"(3) measure regulariser $\frac{\tau}{2}\int\mu^2 dt$   —   uniform (IG) vs joint $(\gamma,\mu)$ (LIG)"
    if not moved:
        t3 += "\n[regression guard kept the straight path on this image; only $\\mu$ changed]"
    ax3.set_title(t3, fontsize=10)
    ax3.set_xlabel(r"$\alpha$  (0 = baseline $\to$ 1 = input)")
    ax3.set_ylabel(r"$\mu_k$")
    ax3.grid(alpha=0.22); ax3.legend(fontsize=9, frameon=False, ncol=3, loc="upper center")

    # row 2: FINAL ATTRIBUTIONS (what actually matters)
    axA = fig.add_subplot(gs[2, 0])
    axA.imshow(image); axA.imshow(smap_s, cmap="inferno", alpha=0.65)
    axA.set_title(f"Straight-line IG attribution\nIns={ins_s:.3f}  Del={del_s:.3f}  "
                  f"Ins$-$Del={ins_s-del_s:.3f}", fontsize=10)
    axA.axis("off")

    axB = fig.add_subplot(gs[2, 1])
    axB.imshow(image); axB.imshow(smap_g, cmap="inferno", alpha=0.65)
    axB.set_title(f"LIG attribution (ours)\nIns={ins_g:.3f}  Del={del_g:.3f}  "
                  f"Ins$-$Del={ins_g-del_g:.3f}", fontsize=10)
    axB.axis("off")

    # bar comparison of the final metrics
    axC = fig.add_subplot(gs[2, 2])
    metrics = ["Ins $\\uparrow$", "Del $\\downarrow$", "Ins$-$Del $\\uparrow$"]
    sv = [ins_s, del_s, ins_s - del_s]
    gv = [ins_g, del_g, ins_g - del_g]
    xpos = np.arange(3); w = 0.36
    axC.bar(xpos - w / 2, sv, w, color=C_STRAIGHT, label="straight IG")
    axC.bar(xpos + w / 2, gv, w, color=C_LIG, label="LIG")
    axC.set_xticks(xpos); axC.set_xticklabels(metrics, fontsize=9)
    axC.set_title("final faithfulness metrics", fontsize=10)
    axC.grid(alpha=0.22, axis="y"); axC.legend(fontsize=8, frameon=False)

    fig.suptitle(r"Straight-line IG  vs  LIG-optimized $(\gamma,\mu)$  "
                 fr"(baseline = {BASELINE};  objective = $\mathrm{{Var}}_\nu(\phi)$, regression guard)",
                 fontsize=12.5, y=0.995)

    plt.savefig("lig_compare.png", bbox_inches="tight", dpi=150)
    print("Saved: lig_compare.png")


if __name__ == "__main__":
    main()