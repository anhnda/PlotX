"""
IG saturation figure: mean |gradient| along the interpolation path black -> image.

Top:    strip of interpolation thumbnails (dimmed), the real image (alpha=1) highlighted.
Bottom: saturation curve mean|grad| vs alpha, showing gradient -> ~0 at a confident input.

Reuses the same model / preprocessing conventions. Edit IMAGE_PATH below.
Run: python ig_saturation.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_PATH = "church.JPEG"

IG_STEPS = 64             # interpolation points along the path
IG_BATCH = 16
N_THUMBS = 9             # thumbnails shown in the strip (subset of the path)

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
def predict_probs(model, batch_hwc, bs=64):
    out = []
    for i in range(0, len(batch_hwc), bs):
        chunk = to_model_tensor(batch_hwc[i:i + bs]).to(DEVICE)
        logits = model(chunk)
        out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Per-step gradient magnitude along the path
# --------------------------------------------------------------------------- #
def path_gradient_magnitudes(model, image, label, steps=IG_STEPS, baseline=None):
    """
    For each interpolation point x(alpha) = baseline + alpha*(image-baseline),
    compute mean over pixels/channels of |d logit_label / d x|.
    Returns alphas (S,) and mean_abs_grad (S,).
    """
    if baseline is None:
        baseline = np.zeros_like(image)
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    path = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]  # (S,H,W,3)

    mags = np.empty(steps, dtype=np.float32)
    for i in range(0, steps, IG_BATCH):
        x = to_model_tensor(path[i:i + IG_BATCH]).to(DEVICE).requires_grad_(True)
        logits = model(x)
        score = logits[:, label].sum()
        g, = torch.autograd.grad(score, x)        # (b,3,H,W), grad wrt normalized input
        per_point = g.abs().mean(dim=(1, 2, 3))   # mean over channels+pixels
        mags[i:i + len(per_point)] = per_point.detach().cpu().numpy()
    return alphas, mags


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    base_prob = predict_probs(model, image[None])[0]
    label = int(base_prob.argmax())
    print(f"Predicted: {categories[label]} (p={base_prob[label]:.3f})")

    alphas, mags = path_gradient_magnitudes(model, image, label)

    # pick thumbnail alphas (evenly spaced incl. endpoints) and their nearest curve points
    thumb_alphas = np.linspace(0.0, 1.0, N_THUMBS, dtype=np.float32)
    thumb_idx = [int(np.argmin(np.abs(alphas - a))) for a in thumb_alphas]

    # --- layout: thumbnail strip on top, curve below --- #
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, N_THUMBS, height_ratios=[1, 2.2], hspace=0.35, wspace=0.08)

    baseline = np.zeros_like(image)
    for col, (a, idx) in enumerate(zip(thumb_alphas, thumb_idx)):
        ax = fig.add_subplot(gs[0, col])
        frame = baseline + a * (image - baseline)
        is_current = (col == N_THUMBS - 1)        # alpha = 1 -> the real image
        ax.imshow(frame, alpha=1.0 if is_current else 0.35)  # dim the path, highlight current
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"α={a:.2f}", fontsize=8,
                     color="black" if is_current else "0.5")
        if is_current:
            for s in ax.spines.values():
                s.set_edgecolor("crimson"); s.set_linewidth(2.5)
        else:
            for s in ax.spines.values():
                s.set_edgecolor("0.8"); s.set_linewidth(0.5)

    # curve
    axc = fig.add_subplot(gs[1, :])
    axc.plot(alphas, mags, color="0.55", lw=2, zorder=1)
    axc.scatter(alphas[thumb_idx[:-1]], mags[thumb_idx[:-1]],
                s=30, color="0.6", zorder=2)
    # highlight the current input (alpha=1)
    axc.scatter([alphas[-1]], [mags[-1]], s=120, color="crimson",
                zorder=3, label="current image (α=1)")
    axc.annotate("gradient ≈ 0\nat a confident input",
                 xy=(alphas[-1], mags[-1]),
                 xytext=(0.62, max(mags) * 0.55),
                 fontsize=11, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson"))

    axc.set_xlabel("interpolation α   (0 = black baseline → 1 = input image)")
    axc.set_ylabel("mean |∂ logit / ∂ x|")
    axc.set_title(f"Gradient saturation along the IG path  ·  {categories[label]}")
    axc.set_xlim(-0.02, 1.02)
    axc.set_ylim(0, max(mags) * 1.1)
    axc.grid(alpha=0.25)
    axc.legend(loc="upper right", frameon=False)

    plt.savefig("ig_saturation.png", bbox_inches="tight", dpi=150)
    print("Saved: ig_saturation.png")


if __name__ == "__main__":
    main()