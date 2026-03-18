---
name: historical-ocr
description: >
  Use this Skill to transcribe historical documents: Tesseract 5 OCR with preprocessing,
  Kraken for historical fonts, confidence filtering, and post-OCR correction with symspellpy.
tags:
  - history
  - OCR
  - Tesseract
  - Kraken
  - handwritten-text-recognition
  - digital-humanities
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
    - pytesseract>=0.3
    - Pillow>=10.0
    - opencv-python>=4.8
    - numpy>=1.23
    - pandas>=1.5
    - symspellpy>=6.7
  system:
    - tesseract>=5.0
last_updated: "2026-03-18"
status: stable
---

# Historical Document OCR

> **TL;DR** — Transcribe printed and handwritten historical documents using Tesseract 5 LSTM
> with OpenCV preprocessing, Kraken for historical typefaces, confidence filtering, and
> symspellpy post-correction to reduce word error rate on archaic vocabulary.

---

## When to Use

Use this Skill when you need to:

- Digitize printed historical documents (early modern newspapers, books, pamphlets)
- Transcribe manuscripts with non-standard letterforms (long-s, ligatures, blackletter)
- Process large batches of archival page scans and generate quality reports
- Apply post-OCR correction to reduce errors from unusual historical spelling
- Extract word-level bounding boxes and confidence scores for downstream NLP

Do **not** use this Skill for:

- Modern documents with clean typography (use a simple `pytesseract.image_to_string` call)
- Real-time camera OCR on mobile (latency budget too low)
- Highly degraded or damaged parchment requiring specialist HTR models (use Transkribus)

---

## Background

Tesseract 5 introduced an LSTM-based recognition engine that substantially outperforms
the legacy pattern-matching engine (Tesseract 3/4) on degraded historical prints. Key
concepts:

| Concept | Explanation |
|---|---|
| LSTM engine (--oem 1) | Neural sequence model; best accuracy for historical text |
| Page segmentation mode (--psm) | PSM 6 = assume uniform block of text; PSM 3 = fully automatic |
| hOCR output | HTML with word bounding boxes, baselines, and per-word confidence |
| Language packs | `eng`, `lat`, `deu`, `fra`, `spa`; install with `apt install tesseract-ocr-[lang]` |
| Long-s / ligatures | Characters unique to early modern printing; require historical traineddata |
| Confidence score | Per-word integer 0–100 from `image_to_data()`; filter below 60 as uncertain |

Kraken is an alternative OCR engine trained explicitly on historical typefaces and
handwriting. It uses a segmentation-first workflow: `kraken.pageseg.segment()` detects
text lines, then a trained model reads each line.

Post-OCR correction with symspellpy applies edit-distance lookup against a historical
vocabulary dictionary. Words with confidence below threshold are replaced by the closest
dictionary entry, reducing character error rate by 15–30% on 17th-century German texts.

---

## Environment Setup

```bash
# Install Tesseract 5 (Ubuntu / Debian)
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng \
  tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-lat

# Verify Tesseract version
tesseract --version

# Create Python environment
conda create -n hist-ocr python=3.11 -y
conda activate hist-ocr

# Install Python dependencies
pip install pytesseract "Pillow>=10.0" "opencv-python>=4.8" \
    "numpy>=1.23" "pandas>=1.5" "symspellpy>=6.7"

# Optional: Kraken for historical fonts
pip install kraken
# Download a historical model (e.g. 15th-century German incunabula)
# kraken get 10.5281/zenodo.6657808

# Verify setup
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

On macOS:

```bash
brew install tesseract tesseract-lang
pip install pytesseract "Pillow>=10.0" "opencv-python>=4.8" "numpy>=1.23" "pandas>=1.5" "symspellpy>=6.7"
```

---

## Core Workflow

### Step 1 — Image Preprocessing Pipeline

Raw archival scans typically require noise removal, binarization, and deskew before OCR.
The pipeline below uses OpenCV and converts to a PIL Image for pytesseract.

```python
import cv2
import numpy as np
from PIL import Image
import pytesseract


def preprocess_historical_image(
    image_path: str,
    deskew: bool = True,
    method: str = "otsu",
) -> tuple[np.ndarray, Image.Image]:
    """
    Preprocess a historical document scan for Tesseract OCR.

    Steps:
      1. Load in colour, convert to grayscale.
      2. Denoise with a Gaussian blur (light despeckling).
      3. Binarize with Otsu or adaptive thresholding.
      4. Optionally deskew using the Hough line transform.

    Args:
        image_path: Absolute path to the input image (JPG, PNG, TIFF).
        deskew:     Whether to correct rotational skew.
        method:     Binarization method: "otsu" | "adaptive".

    Returns:
        Tuple of (binary numpy array, PIL Image ready for pytesseract).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Despeckling: Gaussian blur removes salt-and-pepper noise
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarization
    if method == "otsu":
        _, binary = cv2.threshold(denoised, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'otsu' or 'adaptive'.")

    # Morphological closing to reconnect broken character strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Deskew
    if deskew:
        binary = _deskew(binary)

    pil_image = Image.fromarray(binary)
    return binary, pil_image


def _deskew(binary: np.ndarray) -> np.ndarray:
    """
    Correct rotational skew using minAreaRect on the foreground text pixels.

    Args:
        binary: Binarized image as numpy array (white text on black OR inverted).

    Returns:
        Deskewed binary image.
    """
    # Invert so text pixels are white (foreground)
    inverted = cv2.bitwise_not(binary)
    coords = np.column_stack(np.where(inverted > 0))
    if len(coords) < 10:
        return binary  # not enough foreground pixels

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    (h, w) = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed


def run_tesseract_with_confidence(
    pil_image: Image.Image,
    lang: str = "eng",
    confidence_threshold: int = 60,
    psm: int = 6,
) -> dict:
    """
    Run Tesseract OCR and return full text plus confidence-filtered word list.

    Args:
        pil_image:            Preprocessed PIL Image.
        lang:                 Tesseract language code(s), e.g. "eng+lat".
        confidence_threshold: Discard words with confidence below this value.
        psm:                  Page segmentation mode (6 = uniform text block).

    Returns:
        Dict with keys: full_text (str), words (list of dicts with
        keys text/conf/left/top/width/height), low_conf_words (list).
    """
    config = f"--oem 1 --psm {psm}"
    full_text = pytesseract.image_to_string(pil_image, lang=lang, config=config)

    data = pytesseract.image_to_data(
        pil_image, lang=lang, config=config,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    low_conf_words = []
    for i, word_text in enumerate(data["text"]):
        word_text = word_text.strip()
        if not word_text:
            continue
        conf = int(data["conf"][i])
        entry = {
            "text": word_text,
            "conf": conf,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
        }
        words.append(entry)
        if conf < confidence_threshold:
            low_conf_words.append(entry)

    return {
        "full_text": full_text,
        "words": words,
        "low_conf_words": low_conf_words,
        "mean_confidence": float(np.mean([w["conf"] for w in words])) if words else 0.0,
    }
```

### Step 2 — Batch Folder OCR with Quality Report

```python
import os
import pandas as pd
from pathlib import Path


def batch_ocr_folder(
    input_dir: str,
    output_dir: str,
    lang: str = "deu",
    extensions: tuple = (".jpg", ".jpeg", ".tif", ".tiff", ".png"),
    confidence_threshold: int = 60,
) -> pd.DataFrame:
    """
    Run OCR on every image in a folder and produce a quality report CSV.

    Args:
        input_dir:            Directory containing page scans.
        output_dir:           Directory to write .txt transcript files.
        lang:                 Tesseract language code(s).
        extensions:           Accepted image file extensions.
        confidence_threshold: Confidence threshold for quality flagging.

    Returns:
        DataFrame with one row per image: filename, mean_conf, low_conf_pct,
        word_count, flagged (bool).
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    records = []
    for img_path in image_paths:
        try:
            _, pil_img = preprocess_historical_image(str(img_path))
            result = run_tesseract_with_confidence(
                pil_img, lang=lang,
                confidence_threshold=confidence_threshold,
            )
        except Exception as exc:
            print(f"[WARN] Failed on {img_path.name}: {exc}")
            records.append({
                "filename": img_path.name,
                "mean_conf": 0.0,
                "low_conf_pct": 100.0,
                "word_count": 0,
                "flagged": True,
                "error": str(exc),
            })
            continue

        # Write transcript
        out_path = Path(output_dir) / (img_path.stem + ".txt")
        out_path.write_text(result["full_text"], encoding="utf-8")

        word_count = len(result["words"])
        low_conf_pct = (
            100.0 * len(result["low_conf_words"]) / word_count
            if word_count > 0 else 100.0
        )

        records.append({
            "filename": img_path.name,
            "mean_conf": round(result["mean_confidence"], 1),
            "low_conf_pct": round(low_conf_pct, 1),
            "word_count": word_count,
            "flagged": result["mean_confidence"] < confidence_threshold,
            "error": "",
        })
        print(f"  {img_path.name}: mean_conf={result['mean_confidence']:.1f}  "
              f"words={word_count}  low_conf={low_conf_pct:.1f}%")

    report_df = pd.DataFrame(records)
    report_path = Path(output_dir) / "quality_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\nQuality report saved to {report_path}")
    print(report_df.describe())
    return report_df
```

### Step 3 — symspellpy Post-OCR Correction

```python
from symspellpy import SymSpell, Verbosity


def build_symspell(
    dictionary_path: str,
    max_edit_distance: int = 2,
) -> SymSpell:
    """
    Load a frequency dictionary into SymSpell for post-OCR correction.

    Args:
        dictionary_path: Path to a SymSpell-format frequency dictionary
                         (word SPACE frequency, one per line).
        max_edit_distance: Maximum edit distance for lookup (2 recommended).

    Returns:
        Loaded SymSpell instance.
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
    loaded = sym_spell.load_dictionary(
        dictionary_path, term_index=0, count_index=1, encoding="utf-8"
    )
    if not loaded:
        raise ValueError(f"Failed to load dictionary from {dictionary_path}")
    return sym_spell


def correct_ocr_text(
    words: list[dict],
    sym_spell: SymSpell,
    confidence_threshold: int = 60,
    max_edit_distance: int = 2,
) -> str:
    """
    Apply symspellpy correction to low-confidence OCR words.

    Only words below confidence_threshold are looked up; high-confidence
    words are kept as-is to avoid over-correction.

    Args:
        words:                List of word dicts from run_tesseract_with_confidence().
        sym_spell:            Loaded SymSpell instance.
        confidence_threshold: Words below this confidence are candidates for correction.
        max_edit_distance:    Maximum edit distance for the lookup.

    Returns:
        Corrected full text string.
    """
    corrected_tokens = []
    for w in words:
        token = w["text"]
        if w["conf"] < confidence_threshold and token.isalpha():
            suggestions = sym_spell.lookup(
                token.lower(),
                Verbosity.CLOSEST,
                max_edit_distance=max_edit_distance,
            )
            if suggestions:
                candidate = suggestions[0].term
                # Preserve capitalisation heuristic
                if token[0].isupper():
                    candidate = candidate.capitalize()
                corrected_tokens.append(candidate)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) = (S + D + I) / N.

    Args:
        reference:  Ground-truth transcription string.
        hypothesis: OCR output string.

    Returns:
        WER as a float between 0.0 and 1.0.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n = len(ref_words)
    if n == 0:
        return 0.0

    # Dynamic programming edit distance on word sequences
    dp = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words):
        new_dp = [i + 1]
        for j, hw in enumerate(hyp_words):
            if rw == hw:
                new_dp.append(dp[j])
            else:
                new_dp.append(1 + min(dp[j], dp[j + 1], new_dp[-1]))
        dp = new_dp

    return dp[len(hyp_words)] / n
```

---

## Advanced Usage

### hOCR Output with Coordinates

```python
def extract_hocr(
    pil_image: Image.Image,
    lang: str = "eng",
    output_path: str = None,
) -> str:
    """
    Generate hOCR (HTML with bounding boxes) from a preprocessed image.

    The hOCR format encodes word coordinates as:
      title="bbox left top right bottom; x_wconf NN"
    which is useful for downstream alignment with page facsimiles.

    Args:
        pil_image:   Preprocessed PIL Image.
        lang:        Tesseract language code.
        output_path: If given, write the hOCR HTML to this file.

    Returns:
        hOCR HTML string.
    """
    config = "--oem 1 --psm 6"
    hocr = pytesseract.image_to_pdf_or_hocr(
        pil_image, lang=lang, config=config, extension="hocr"
    )
    hocr_str = hocr.decode("utf-8")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(hocr_str)
        print(f"hOCR saved to {output_path}")
    return hocr_str
```

### Kraken Segmentation for Historical Fonts

```bash
# Install Kraken and download a historical Latin model
pip install kraken
kraken get 10.5281/zenodo.6657808

# Binarize input image
kraken -i page_001.jpg page_001_bin.png binarize

# Segment text lines
kraken -i page_001_bin.png page_001_seg.json segment

# Transcribe with a historical model
kraken -i page_001_bin.png page_001.txt ocr -m historical_latin.mlmodel
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `TesseractNotFoundError` | Tesseract binary not on PATH | Set `pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'` |
| Empty output on good image | Wrong PSM mode | Try `--psm 3` (auto) or `--psm 11` (sparse text) |
| Very low confidence (<30) across page | Binarization threshold too aggressive | Switch from Otsu to adaptive (`method="adaptive"`) |
| Long-s (ſ) misread as f | No historical traineddata | Download `enm` (Middle English) or custom traineddata |
| `SymSpell: dictionary not loaded` | Wrong path or encoding | Ensure UTF-8; check term/count column indices |
| Deskew makes image worse | Very curved binding warps | Set `deskew=False`; use manual GCP-based correction |
| `cv2.error` on TIFF | Unsupported TIFF compression | Open with Pillow first: `pil_img = Image.open(path).convert("RGB")`; convert via `numpy` |

---

## External Resources

- Tesseract 5 documentation: <https://tesseract-ocr.github.io/tessdoc/>
- Tesseract language data files: <https://github.com/tesseract-ocr/tessdata_best>
- Kraken HTR engine: <https://kraken.re/>
- Transkribus (HTR platform for manuscripts): <https://readcoop.eu/transkribus/>
- symspellpy PyPI: <https://pypi.org/project/symspellpy/>
- Historical frequency dictionaries: <https://www.corpusvitarum.de/>
- OpenCV image processing docs: <https://docs.opencv.org/4.x/>

---

## Examples

### Example 1 — Single Page Transcription with Correction

```python
# Transcribe one page of an 18th-century German newspaper and correct low-confidence words
from pathlib import Path

image_path = "/data/historical/zeitung_1790_p003.tif"
dict_path = "/data/dicts/deu_historical_frequency.txt"

# Preprocess
_, pil_img = preprocess_historical_image(image_path, deskew=True, method="otsu")

# OCR with German language pack
result = run_tesseract_with_confidence(pil_img, lang="deu", confidence_threshold=60)
print(f"Mean confidence: {result['mean_confidence']:.1f}")
print(f"Low-confidence words: {len(result['low_conf_words'])}")

# Post-correction
sym_spell = build_symspell(dict_path, max_edit_distance=2)
corrected_text = correct_ocr_text(result["words"], sym_spell, confidence_threshold=60)

# Save transcript
out = Path("/data/output/zeitung_1790_p003_corrected.txt")
out.write_text(corrected_text, encoding="utf-8")
print(f"Saved corrected transcript to {out}")

# WER against manual ground truth (if available)
gt_path = "/data/ground_truth/zeitung_1790_p003.txt"
if Path(gt_path).exists():
    reference = Path(gt_path).read_text(encoding="utf-8")
    wer_raw = calculate_wer(reference, result["full_text"])
    wer_corrected = calculate_wer(reference, corrected_text)
    print(f"WER before correction: {wer_raw:.3f}")
    print(f"WER after correction:  {wer_corrected:.3f}")
    print(f"Improvement: {(wer_raw - wer_corrected) / wer_raw * 100:.1f}%")
```

### Example 2 — Batch Archive Folder Processing

```python
# Process an entire folder of scanned newspaper pages
report = batch_ocr_folder(
    input_dir="/data/archive/1850_newspapers/",
    output_dir="/data/output/1850_transcripts/",
    lang="eng",
    extensions=(".tif", ".tiff"),
    confidence_threshold=60,
)

# Summarise quality
flagged = report[report["flagged"]]
print(f"\nTotal pages: {len(report)}")
print(f"Flagged for review (mean_conf < 60): {len(flagged)}")
print(f"Average word count per page: {report['word_count'].mean():.0f}")
print(f"Median mean confidence: {report['mean_conf'].median():.1f}")

# Pages needing manual review
if len(flagged) > 0:
    print("\nPages requiring manual review:")
    print(flagged[["filename", "mean_conf", "low_conf_pct"]].to_string(index=False))
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Tesseract 5, OpenCV preprocessing, batch OCR, symspellpy post-correction, WER metric |
