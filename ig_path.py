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
TANGENT_ALPHAS = [0.15, 0.45, 0.92]   # where to draw local tangent slopes

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
    print(f"f(0)={f[0]:.3f}  f(1)={f[1 - 1 + len(f) - 1]:.3f}  "
          f"f(1)-f(0)={total_change:.3f}")
    print(f"∫ df/dα dα = {integral:.3f}   (should ≈ f(1)-f(0))")

    # --- figure --- #
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # the saturating curve f(alpha)
    ax.plot(alphas, f, color="#1f4e8c", lw=2.5, zorder=3, label=r"$f(\alpha)$ = class score")
    # shade the accumulated change (the integral)
    ax.fill_between(alphas, f[0], f, color="#1f4e8c", alpha=0.12, zorder=1,
                    label="accumulated change\n= ∫ slopes along path")

    # local tangent slopes
    for a in TANGENT_ALPHAS:
        idx = int(np.argmin(np.abs(alphas - a)))
        slope = dfd[idx]
        seg = 0.10
        xs = np.array([a - seg, a + seg])
        ys = f[idx] + slope * (xs - a)
        flat = abs(slope) < 0.15 * np.max(np.abs(dfd))
        col = "crimson" if flat else "0.35"
        ax.plot(xs, ys, color=col, lw=2, zorder=4)
        ax.scatter([a], [f[idx]], s=28, color=col, zorder=5)
        tag = "slope ≈ 0\n(saturated)" if flat else f"slope={slope:.1f}"
        ax.annotate(tag, xy=(a, f[idx]), xytext=(a, f[idx] + 0.10 * (f.max() - f.min()) + 1.0),
                    fontsize=9, color=col, ha="center")

    ax.scatter([alphas[0]], [f[0]], s=60, color="black", zorder=6)
    ax.scatter([alphas[-1]], [f[-1]], s=60, color="black", zorder=6)
    ax.text(0.0, f[0], "  baseline (black)", va="center", ha="left", fontsize=9)
    ax.text(1.0, f[-1], "input  ", va="center", ha="right", fontsize=9)

    ax.set_xlabel(r"interpolation $\alpha$   (0 = baseline → 1 = input)")
    ax.set_ylabel("target class score" + ("  (logit)" if USE_LOGIT else "  (prob)"))
    ax.set_title("Saturation → integrate:  one gradient saturates, "
                 "the path integral recovers the whole")
    ax.set_xlim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    # inset: slope magnitude collapsing
    axin = ax.inset_axes([0.10, 0.58, 0.34, 0.34])
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