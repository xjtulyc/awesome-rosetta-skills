---
name: image-analysis-art
description: >
  Use this Skill for computational art history: CNN feature extraction, style
  classification on WikiArt, color palette analysis, and brushstroke texture metrics.
tags:
  - art-history
  - image-analysis
  - CNN
  - style-classification
  - color-analysis
  - WikiArt
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - torch>=2.0
    - torchvision>=0.15
    - opencv-python>=4.8
    - numpy>=1.23
    - matplotlib>=3.6
    - scikit-learn>=1.2
last_updated: "2026-03-18"
status: stable
---

# Computational Art History with CNN Feature Extraction

> **TL;DR** — Analyse paintings computationally: extract deep CNN features from
> ResNet50/VGG16, fine-tune for WikiArt style classification, extract dominant
> color palettes, and quantify brushstroke texture with Gabor filters and LBP.

---

## When to Use

Use this Skill when you need to:

- Extract CNN features (penultimate layer embeddings) from artwork images for
  similarity search or dimensionality reduction
- Fine-tune a pretrained ResNet50/VGG16 on WikiArt artist/style/genre labels
- Perform k-means dominant color palette extraction (5-color)
- Compare color histograms between paintings using Bhattacharyya distance
- Quantify brushstroke texture using Gabor filter banks and Local Binary Patterns
- Visualize style clusters with UMAP or t-SNE embeddings
- Produce GradCAM attribution maps to understand which image regions drive predictions

| Task | Recommended Approach |
|---|---|
| Style classification | Fine-tuned ResNet50, 27 WikiArt style classes |
| Artist attribution | CNN features + SVM or kNN classifier |
| Color similarity | HSV histogram + Bhattacharyya distance |
| Texture analysis | Gabor filterbank (8 orientations, 5 scales) |
| Visualization | UMAP on 2048-dim ResNet features |

---

## Background

### WikiArt Dataset

WikiArt (<https://www.wikiart.org/>) is the standard benchmark dataset for
computational art history. It contains approximately 80,000 paintings across:

- **27 art styles**: Impressionism, Cubism, Surrealism, Renaissance, Baroque, etc.
- **45 genres**: portrait, landscape, abstract, religious painting, etc.
- **135+ artists**: Monet, Picasso, Van Gogh, Rembrandt, etc.

The dataset is organized hierarchically: `wikiart/{style}/{artist_name}_{work_title}.jpg`.

### CNN Feature Extraction

Transfer learning from ImageNet-pretrained networks provides powerful visual
representations for art images:

- **ResNet50**: 2048-dimensional feature vector from `avgpool` layer (before `fc`)
- **VGG16**: 4096-dimensional vector from `classifier[3]` (second FC layer)

These features capture mid-level textures and high-level semantic content
simultaneously — a property especially useful for style discrimination.

### Color Analysis

- **Dominant palette**: Run k-means (k=5) on all pixel colors (after resizing to
  100×100 to reduce computation). Cluster centers are the palette colors.
- **Bhattacharyya distance**: Measures overlap between two color histograms.
  Range [0, ∞); 0 = identical distributions.
- **Color space**: HSV is more perceptually meaningful than RGB for art analysis;
  hue encodes color category, saturation encodes purity, value encodes brightness.

### Texture: Gabor Filters and LBP

- **Gabor filter**: A sinusoidal wave modulated by a Gaussian envelope, sensitive
  to specific spatial frequencies and orientations. An 8-orientation × 5-scale
  filterbank captures rich texture descriptors.
- **LBP (Local Binary Pattern)**: For each pixel, compare neighbors to center;
  encode as a binary number. Rotation-invariant uniform LBP gives a 59-dimensional
  histogram per region — efficient and robust for brushstroke analysis.

---

## Environment Setup

```bash
# Create environment
conda create -n art-cnn python=3.11 -y
conda activate art-cnn
pip install "torch>=2.0" "torchvision>=0.15" "opencv-python>=4.8" \
            "numpy>=1.23" "matplotlib>=3.6" "scikit-learn>=1.2" \
            umap-learn

# Verify GPU availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Download WikiArt (requires registration):

```bash
# Alternative: use the Hugging Face dataset mirror
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('huggan/wikiart', split='train[:1000]')"
```

---

## Core Workflow

### Step 1 — CNN Feature Extraction and Style PCA Visualization

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import List, Tuple, Dict


def build_feature_extractor(backbone: str = 'resnet50') -> Tuple[nn.Module, transforms.Compose]:
    """
    Build a CNN feature extractor by removing the classification head.

    Args:
        backbone: 'resnet50' or 'vgg16'

    Returns:
        Tuple of (model, preprocessing_transform). Model outputs feature vectors.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer — output: 2048-dim
        model = nn.Sequential(*list(model.children())[:-1])
    elif backbone == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use features + avgpool; strip final classifier
        model.classifier = nn.Sequential(*list(model.classifier.children())[:4])
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet50' or 'vgg16'.")

    model.eval()
    return model, preprocess


def extract_features_batch(
    image_paths: List[str],
    model: nn.Module,
    preprocess: transforms.Compose,
    device: str = 'cpu',
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract CNN feature vectors for a list of image paths.

    Args:
        image_paths: List of absolute paths to image files.
        model:       Feature extractor (output of build_feature_extractor).
        preprocess:  torchvision transform pipeline.
        device:      'cuda' or 'cpu'.
        batch_size:  Number of images per forward pass.

    Returns:
        numpy array of shape (N, feature_dim).
    """
    model = model.to(device)
    model.eval()
    features = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                tensors.append(preprocess(img))
            except Exception as e:
                print(f"Warning: could not load {p}: {e}")
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = model(batch)
        out = out.view(out.size(0), -1).cpu().numpy()
        features.append(out)

    return np.vstack(features)


def visualize_style_pca(
    features: np.ndarray,
    labels: List[str],
    output_path: str = 'style_pca.png',
) -> None:
    """
    Reduce features to 2D with PCA and plot colored by style label.

    Args:
        features:    (N, D) feature matrix.
        labels:      List of style label strings.
        output_path: Where to save the plot.
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features)

    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    n_classes = len(le.classes_)

    cmap = plt.cm.get_cmap('tab20', n_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=label_ids, cmap=cmap, alpha=0.6, s=15)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_classes))
    cbar.set_ticklabels(le.classes_)
    ax.set_title('CNN Feature PCA by Art Style')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved PCA plot to {output_path}")


# --- Demo usage ---
# model, preprocess = build_feature_extractor('resnet50')
# image_paths = sorted(Path('wikiart').glob('**/*.jpg'))[:500]
# labels = [p.parent.name for p in image_paths]
# feats = extract_features_batch([str(p) for p in image_paths], model, preprocess)
# visualize_style_pca(feats, labels)
```

### Step 2 — Dominant Color Palette Extraction

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def extract_dominant_palette(
    image_path: str,
    n_colors: int = 5,
    resize_to: Tuple[int, int] = (100, 100),
) -> np.ndarray:
    """
    Extract n dominant colors from an image using k-means clustering in RGB space.

    Args:
        image_path: Path to the image file.
        n_colors:   Number of dominant colors to extract.
        resize_to:  Resize image to this size before clustering (for speed).

    Returns:
        numpy array of shape (n_colors, 3) with RGB values in [0, 255].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for speed
    img_small = cv2.resize(img_rgb, resize_to)
    pixels = img_small.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # Sort by cluster size (largest first)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    palette = kmeans.cluster_centers_[sorted_idx].astype(np.uint8)

    return palette


def compare_color_histograms(path1: str, path2: str) -> float:
    """
    Compare two images by their HSV color histograms using Bhattacharyya distance.
    Lower distance = more similar color distributions.

    Args:
        path1: Path to first image.
        path2: Path to second image.

    Returns:
        Bhattacharyya distance in [0, inf).
    """
    def compute_hist(path: str) -> np.ndarray:
        img = cv2.imread(path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img_hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
        return hist

    h1 = compute_hist(path1)
    h2 = compute_hist(path2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


def plot_palette(palette: np.ndarray, title: str = 'Dominant Palette') -> None:
    """Visualize a color palette as horizontal swatches."""
    n = len(palette)
    fig, ax = plt.subplots(figsize=(n * 1.5, 1.5))
    for i, color in enumerate(palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                   color=color / 255.0))
        hex_str = '#{:02X}{:02X}{:02X}'.format(*color)
        ax.text(i + 0.5, -0.15, hex_str, ha='center',
                va='top', fontsize=8)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.3, 1)
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig('color_palette.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved color_palette.png")


# --- Demo ---
# palette = extract_dominant_palette('monet_haystacks.jpg', n_colors=5)
# plot_palette(palette, title='Monet — Haystacks (dominant colors)')
# dist = compare_color_histograms('monet_haystacks.jpg', 'monet_waterlilies.jpg')
# print(f"Color histogram distance: {dist:.4f}")
```

### Step 3 — Gabor Texture Features for Brushstroke Analysis

```python
import cv2
import numpy as np
from sklearn.preprocessing import normalize


def gabor_filterbank_features(
    image_path: str,
    n_orientations: int = 8,
    n_scales: int = 5,
) -> np.ndarray:
    """
    Compute Gabor filterbank features for brushstroke texture analysis.

    For each (orientation, scale) pair, apply a Gabor filter to the grayscale
    image and compute mean and variance of the response magnitude.

    Args:
        image_path:     Path to the painting image.
        n_orientations: Number of filter orientations (evenly spaced in [0, pi)).
        n_scales:       Number of spatial frequency scales.

    Returns:
        1D numpy array of length 2 * n_orientations * n_scales (mean + var per filter).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img = img.astype(np.float32) / 255.0

    features = []
    orientations = [np.pi * k / n_orientations for k in range(n_orientations)]
    wavelengths = [2 ** (s + 1) for s in range(n_scales)]

    for lam in wavelengths:
        for theta in orientations:
            kernel = cv2.getGaborKernel(
                ksize=(int(lam * 3) | 1, int(lam * 3) | 1),  # must be odd
                sigma=lam * 0.56,
                theta=theta,
                lambd=lam,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            response = cv2.filter2D(img, cv2.CV_32F, kernel)
            magnitude = np.abs(response)
            features.append(magnitude.mean())
            features.append(magnitude.var())

    feat_vec = np.array(features, dtype=np.float32)
    return feat_vec


def lbp_features(image_path: str, n_points: int = 8, radius: int = 1) -> np.ndarray:
    """
    Compute rotation-invariant uniform LBP histogram for brushstroke texture.

    Args:
        image_path: Path to image.
        n_points:   Number of circularly symmetric neighbor points.
        radius:     Radius of the LBP circle.

    Returns:
        Normalized LBP histogram (59-bin for uniform patterns with n_points=8).
    """
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        raise ImportError("scikit-image required: pip install scikit-image")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    n_bins = n_points + 2  # uniform patterns + 1 non-uniform bin
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def texture_feature_matrix(image_paths: List[str]) -> np.ndarray:
    """
    Compute combined Gabor + LBP texture feature matrix for a list of images.

    Returns:
        (N, D) feature matrix, L2-normalized.
    """
    rows = []
    for p in image_paths:
        try:
            gabor = gabor_filterbank_features(p)
            lbp = lbp_features(p)
            rows.append(np.concatenate([gabor, lbp]))
        except Exception as e:
            print(f"Warning: {p}: {e}")
            rows.append(np.zeros(2 * 8 * 5 + 10))

    mat = np.vstack(rows)
    return normalize(mat, norm='l2')


# --- Demo ---
# from pathlib import Path
# imgs = sorted(Path('wikiart/Impressionism').glob('*.jpg'))[:50]
# feat_mat = texture_feature_matrix([str(p) for p in imgs])
# print(f"Texture feature matrix shape: {feat_mat.shape}")
```

---

## Advanced Usage

### Fine-Tuning ResNet50 on WikiArt Styles

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class WikiArtDataset(Dataset):
    """WikiArt dataset: folder per style, images inside."""

    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            for img_path in (self.root / cls).glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def finetune_resnet50_wikiart(
    data_dir: str,
    n_classes: int = 27,
    n_epochs: int = 10,
    lr: float = 1e-4,
    device: str = 'cuda',
) -> nn.Module:
    """
    Fine-tune ResNet50 on WikiArt style classification.
    Replaces the final FC layer with a new n_classes head.
    Only trains the last layer + layer4 (partial fine-tuning).
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = WikiArtDataset(data_dir, transform=train_tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Freeze all layers except layer4 and fc
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    model.fc = nn.Linear(2048, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{n_epochs} — loss: {total_loss/len(dataset):.4f}, acc: {acc:.3f}")

    return model
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large for GPU | Reduce `batch_size` to 8 or 16 |
| `OSError: image file is truncated` | Corrupted image file | Use `from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True` |
| Gabor kernel size error (even number) | Even kernel size passed to `getGaborKernel` | Force odd: `ksize = (int(lam*3) | 1, int(lam*3) | 1)` |
| Low accuracy after fine-tuning | Frozen layers include batch norm | Set `model.train()` after freezing or use `eval()` on frozen BN layers |
| `sklearn.cluster.KMeans` slow on large images | Full-resolution pixel matrix is huge | Always resize to 100×100 before clustering |
| LBP import error | scikit-image not installed | `pip install scikit-image` |

---

## External Resources

- WikiArt Dataset: <https://www.wikiart.org/>
- WikiArt on Hugging Face: <https://huggingface.co/datasets/huggan/wikiart>
- Saleh, B. & Elgammal, A. (2015). "Large-scale Classification of Fine-Art Paintings."
  arXiv:1505.00855. <https://arxiv.org/abs/1505.00855>
- Gatys, L.A. et al. (2016). "Image Style Transfer Using Convolutional Neural Networks."
  CVPR 2016. <https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf>
- Selvaraju, R.R. et al. (2017). "Grad-CAM." ICCV 2017. <https://arxiv.org/abs/1610.02391>
- PyTorch torchvision model zoo: <https://pytorch.org/vision/stable/models.html>

---

## Examples

### Example 1 — Extract Features and Cluster Paintings by Artist

```python
from sklearn.cluster import KMeans
from pathlib import Path

model, preprocess = build_feature_extractor('resnet50')
image_paths = sorted(Path('wikiart').glob('**/*.jpg'))[:200]
labels = [p.parent.name for p in image_paths]

feats = extract_features_batch([str(p) for p in image_paths], model, preprocess)
print(f"Feature matrix: {feats.shape}")

# k-means clustering in feature space
km = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_ids = km.fit_predict(feats)

# Purity: how well clusters align with true style labels
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
true_ids = LabelEncoder().fit_transform(labels)
ari = adjusted_rand_score(true_ids, cluster_ids)
nmi = normalized_mutual_info_score(true_ids, cluster_ids)
print(f"Clustering ARI: {ari:.3f}, NMI: {nmi:.3f}")
```

### Example 2 — Color Palette Comparison Across Impressionists

```python
from pathlib import Path
import pandas as pd
import itertools

artists = ['Monet', 'Renoir', 'Pissarro']
image_paths = {
    artist: sorted(Path(f'wikiart/Impressionism/{artist}').glob('*.jpg'))[:10]
    for artist in artists
}

distances = []
for (a1, paths1), (a2, paths2) in itertools.combinations(image_paths.items(), 2):
    dists = []
    for p1 in paths1:
        for p2 in paths2:
            try:
                d = compare_color_histograms(str(p1), str(p2))
                dists.append(d)
            except Exception:
                pass
    if dists:
        distances.append({'pair': f'{a1} vs {a2}', 'mean_bhatt_dist': round(sum(dists)/len(dists), 4)})

df = pd.DataFrame(distances)
print("Color histogram distances between Impressionist artists:")
print(df.to_string(index=False))
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — CNN features, palette extraction, Gabor/LBP texture |
