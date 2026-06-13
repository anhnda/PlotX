"""
Expected Gradients (EG) style figure: a cloud of baselines, many IG paths.

Layout (bottom -> top), one column per baseline + a shared mean/input column block:

  row 3 (top)    :                      original input
  row 2          :                      mean IG over baselines  (the EG estimate)
  row 1          :  IG | IG | IG         per-baseline IG attribution maps
  row 0 (bottom) :  black | white | noise  the three baselines

Baselines: black, white, white-noise. IG is integrated along each
baseline->input straight-line path; EG approximates averaging IG over a
baseline distribution by taking the mean of the per-baseline maps.

Reuses the same model / preprocessing. Edit IMAGE_PATH below.
Run: python eg_baselines.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_PATH = "church.JPEG"

IG_STEPS = 64
IG_BATCH = 16
NOISE_STD = 1.0           # white-noise baseline: uniform[0,1] if 1.0; gaussian otherwise unused
DISP_PCT = 99             # percentile clip for heatmap display
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# Integrated Gradients (pixel-level), given an explicit baseline image
# --------------------------------------------------------------------------- #
def integrated_gradients(model, image, label, baseline, steps=IG_STEPS):
    """image, baseline: (H,W,3) float [0,1]. Returns (H,W) saliency = |sum over channels|."""
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    path = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]  # (S,H,W,3)

    total = torch.zeros(3, image.shape[0], image.shape[1], device=DEVICE)
    for i in range(0, len(path), IG_BATCH):
        x = to_model_tensor(path[i:i + IG_BATCH]).to(DEVICE).requires_grad_(True)
        logits = model(x)
        score = logits[:, label].sum()
        g, = torch.autograd.grad(score, x)
        total += g.sum(dim=0)
    avg_grad = (total / steps).cpu().numpy()                      # (3,H,W), wrt normalized input

    diff = (image - baseline).transpose(2, 0, 1) / IMAGENET_STD[:, None, None]  # match grad space
    ig = diff * avg_grad                                          # (3,H,W)
    return np.abs(ig).sum(axis=0)                                 # (H,W)


def make_baselines(image, rng):
    black = np.zeros_like(image)
    white = np.ones_like(image)
    noise = rng.random(image.shape).astype(np.float32)           # white noise, uniform[0,1]
    return {"black": black, "white": white, "white-noise": noise}


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    label = int(predict_logits(model, image[None])[0].argmax())
    print(f"Predicted: {categories[label]}")

    baselines = make_baselines(image, rng)
    names = list(baselines.keys())                               # black, white, white-noise

    ig_maps = {}
    for name in names:
        ig_maps[name] = integrated_gradients(model, image, label, baselines[name])
        print(f"IG done: baseline={name}")

    mean_ig = np.mean(np.stack([ig_maps[n] for n in names], axis=0), axis=0)

    # shared display scale across the 3 maps + mean (fair comparison)
    all_maps = list(ig_maps.values()) + [mean_ig]
    vmax = np.percentile(np.concatenate([m.ravel() for m in all_maps]), DISP_PCT)
    norm = lambda m: np.clip(m / (vmax + 1e-12), 0, 1)

    def show_heat(ax, m, title, border=None):
        ax.imshow(image, alpha=0.35)
        ax.imshow(norm(m), cmap="hot", alpha=0.65)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if border:
            for s in ax.spines.values():
                s.set_edgecolor(border); s.set_linewidth(2.5)

    def show_img(ax, im, title, border=None):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if border:
            for s in ax.spines.values():
                s.set_edgecolor(border); s.set_linewidth(2.5)

    # --- layout --- #
    # 4 rows x 3 cols. Top two rows span all 3 cols (input, mean IG centered).
    fig = plt.figure(figsize=(10.5, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.28, wspace=0.10,
                  height_ratios=[1.0, 1.0, 1.0, 1.0])

    # row 0 (top): original input, centered (use middle col, blank sides)
    ax_in = fig.add_subplot(gs[0, 1])
    show_img(ax_in, image, f"input\n{categories[label]}", border="#1f4e8c")

    # row 1: mean IG (the EG estimate), centered
    ax_mean = fig.add_subplot(gs[1, 1])
    show_heat(ax_mean, mean_ig, "mean IG over baselines\n(Expected Gradients)", border="crimson")

    # downward arrows linking rows (annotation in figure coords)
    fig.text(0.5, 0.755, "↑ average", ha="center", va="center", fontsize=11, color="crimson")
    fig.text(0.5, 0.515, "↑ many IG paths", ha="center", va="center", fontsize=11, color="0.4")

    # row 2: per-baseline IG maps
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[2, j])
        show_heat(ax, ig_maps[name], f"IG  ·  {name}")

    # row 3 (bottom): the baselines themselves
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[3, j])
        b = baselines[name]
        show_img(ax, np.clip(b, 0, 1), f"baseline: {name}")

    fig.text(0.5, 0.275, "↑ integrate baseline → input", ha="center", va="center",
             fontsize=11, color="0.4")

    fig.suptitle("Expected Gradients: a cloud of baselines, many IG paths",
                 y=0.93, fontsize=13)

    plt.savefig("eg_baselines.png", bbox_inches="tight", dpi=150)
    print("Saved: eg_baselines.png")


if __name__ == "__main__":
    main()