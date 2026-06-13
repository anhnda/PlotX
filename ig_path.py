"""
"Saturation -> integrate" figure.

Single gradient at the confident input is ~0 (saturated), but accumulating the
slopes d f / d alpha along the whole baseline->input path recovers the full
change in the model output. This is the IG completeness intuition.

Plots, vs interpolation alpha (0 = black baseline -> 1 = input image):
  - f(alpha): target-class score (logit) along the path        [the saturating curve]
  - shaded area = integral of slopes = total change f(1) - f(0)
  - a few local tangent slopes, with the near-input one ~flat (saturated)
  - inset: mean |gradient| (the per-step slope magnitude) collapsing to ~0

Reuses the same model / preprocessing. Edit IMAGE_PATH below.
Run: python saturation_integrate.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_PATH = "church.JPEG"

STEPS = 128               # interpolation points along the path
BATCH = 16
USE_LOGIT = True          # plot raw logit (clearer saturation) vs softmax prob
N_TANGENTS = 3            # tangents drawn: 1 steep (in the rise) + rest on the plateau

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
# f(alpha) and df/dalpha along the path
# --------------------------------------------------------------------------- #
def path_value_and_slope(model, image, label, steps=STEPS, baseline=None):
    """
    Returns alphas (S,), f (S,), and dfdalpha (S,) where
      f(alpha)        = target score at x(alpha) = baseline + alpha*(image-baseline)
      dfdalpha(alpha) = directional derivative of the score along the path
                      = sum_pixels grad_x f . (image-baseline)
    Note grad is wrt normalized input, so we feed the normalized direction too.
    """
    if baseline is None:
        baseline = np.zeros_like(image)
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    path = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]

    # normalized direction (image - baseline) in model input space
    diff = (image - baseline).transpose(2, 0, 1) / IMAGENET_STD[:, None, None]  # (3,H,W)
    diff_t = torch.tensor(diff, dtype=torch.float32, device=DEVICE)

    f = np.empty(steps, dtype=np.float32)
    dfd = np.empty(steps, dtype=np.float32)
    for i in range(0, steps, BATCH):
        x = to_model_tensor(path[i:i + BATCH]).to(DEVICE).requires_grad_(True)
        logits = model(x)
        sel = logits[:, label]
        f[i:i + len(sel)] = sel.detach().cpu().numpy()
        score = sel.sum()
        g, = torch.autograd.grad(score, x)               # (b,3,H,W)
        # directional derivative: <grad, direction> per sample
        dd = (g * diff_t[None]).sum(dim=(1, 2, 3))
        dfd[i:i + len(dd)] = dd.detach().cpu().numpy()
    return alphas, f, dfd


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    label = int(predict_logits(model, image[None])[0].argmax())
    print(f"Predicted: {categories[label]}")

    alphas, f, dfd = path_value_and_slope(model, image, label)

    # integral of the slope = total change (completeness check)
    integral = np.trapz(dfd, alphas)
    total_change = f[-1] - f[0]
    print(f"f(0)={f[0]:.3f}  f(1)={f[-1]:.3f}  f(1)-f(0)={total_change:.3f}")
    print(f"∫ df/dα dα = {integral:.3f}   (should ≈ f(1)-f(0))")

    # --- figure --- #
    fig, ax = plt.subplots(figsize=(9, 5.5))

    yspan = f.max() - f.min()

    # the saturating curve f(alpha)
    ax.plot(alphas, f, color="#1f4e8c", lw=2.5, zorder=3, label=r"$f(\alpha)$ = class score")
    # shade the accumulated change (the integral)
    ax.fill_between(alphas, f[0], f, color="#1f4e8c", alpha=0.12, zorder=1,
                    label="accumulated change\n= ∫ slopes along path")

    # --- pick tangent alphas adaptively ---
    # one in the steep rise (alpha of max slope), the rest on the plateau.
    absd = np.abs(dfd)
    steep_a = float(alphas[int(absd.argmax())])
    flat_as = list(np.linspace(0.55, 0.92, N_TANGENTS - 1))
    tangent_alphas = sorted([steep_a] + flat_as)
    flat_thresh = 0.15 * absd.max()

    # alternate annotation heights so labels never overlap; keep them inside axes
    for k, a in enumerate(tangent_alphas):
        idx = int(np.argmin(np.abs(alphas - a)))
        slope = dfd[idx]
        seg = 0.06
        xs = np.array([a - seg, a + seg])
        ys = f[idx] + slope * (xs - a)
        flat = abs(slope) < flat_thresh
        col = "crimson" if flat else "#0a7d24"
        ax.plot(xs, ys, color=col, lw=2.5, zorder=4)
        ax.scatter([a], [f[idx]], s=30, color=col, zorder=5)
        tag = "slope ≈ 0\n(saturated)" if flat else f"steep slope\n(df/dα={slope:.0f})"
        # place flat labels below the plateau, steep label below-left in the open area
        if flat:
            ytxt = f[idx] - 0.16 * yspan
            ax.annotate(tag, xy=(a, f[idx]), xytext=(a, ytxt),
                        fontsize=9, color=col, ha="center", va="top",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1))
        else:
            ax.annotate(tag, xy=(a, f[idx]), xytext=(a + 0.10, f[idx] - 0.28 * yspan),
                        fontsize=9, color=col, ha="left", va="top",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1))

    ax.scatter([alphas[0]], [f[0]], s=60, color="black", zorder=6)
    ax.scatter([alphas[-1]], [f[-1]], s=60, color="black", zorder=6)
    ax.annotate("baseline (black)", xy=(alphas[0], f[0]), xytext=(0.04, f[0] + 0.05 * yspan),
                fontsize=9, va="bottom", ha="left")
    ax.annotate("input", xy=(alphas[-1], f[-1]), xytext=(0.95, f[-1] - 0.06 * yspan),
                fontsize=9, va="top", ha="right")

    ax.set_xlabel(r"interpolation $\alpha$   (0 = baseline → 1 = input)")
    ax.set_ylabel("target class score" + ("  (logit)" if USE_LOGIT else "  (prob)"))
    ax.set_title("Saturation → integrate\none gradient saturates; the path integral recovers the whole",
                 fontsize=12, pad=10)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(f.min() - 0.30 * yspan, f.max() + 0.12 * yspan)  # headroom for labels
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    # inset: slope magnitude collapsing -- placed lower-left, clear of the curve & labels
    axin = ax.inset_axes([0.40, 0.30, 0.34, 0.32])
    axin.plot(alphas, np.abs(dfd), color="crimson", lw=1.8)
    axin.set_title("slope magnitude |df/dα|", fontsize=8)
    axin.set_xlabel("α", fontsize=8)
    axin.tick_params(labelsize=7)
    axin.grid(alpha=0.2)
    axin.axhline(0, color="0.6", lw=0.6)

    plt.savefig("saturation_integrate.png", bbox_inches="tight", dpi=150)
    print("Saved: saturation_integrate.png")


if __name__ == "__main__":
    main()