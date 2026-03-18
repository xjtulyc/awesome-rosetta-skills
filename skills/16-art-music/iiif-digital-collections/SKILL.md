---
name: iiif-digital-collections
description: >
  Use this Skill to work with IIIF digital collection APIs: manifest parsing,
  image region download, annotation extraction, and cross-institution collection search.
tags:
  - art-history
  - IIIF
  - digital-collections
  - manifest
  - museum-APIs
  - image-annotation
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
    - requests>=2.28
    - Pillow>=10.0
    - pandas>=1.5
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# IIIF Digital Collections: Manifest Parsing and Image Access

> **TL;DR** — Work programmatically with IIIF Presentation and Image APIs:
> parse manifests from Europeana, BnF, Wellcome, and British Library; download
> image regions; extract W3C Web Annotations; and build thumbnail grids for
> cross-institution comparison.

---

## When to Use

Use this Skill when you need to:

- Fetch and parse IIIF Presentation API 3.0 manifests from any institution
- Download full images or specific spatial regions using the IIIF Image API URL syntax
- Extract scholarly annotations (transcriptions, tags, commentary) from annotation pages
- Build cross-institution search workflows using IIIF-compliant endpoints
- Create thumbnail grids for comparative art historical research
- Parse manifest metadata: title, creator, date, rights, description

| Institution | IIIF Manifest Base URL |
|---|---|
| Wellcome Collection | `https://iiif.wellcomecollection.org/presentation/v3/{id}` |
| British Library | `https://api.bl.uk/metadata/iiif/{id}/manifest.json` |
| Europeana | `https://iiif.europeana.eu/presentation/{id}/manifest` |
| BnF (Gallica) | `https://gallica.bnf.fr/iiif/{ark}/manifest.json` |
| Harvard Art Museums | `https://iiif.harvardartmuseums.org/manifests/object/{id}` |

---

## Background

### IIIF Standards Overview

The **International Image Interoperability Framework** (IIIF) defines a family of
open APIs that enable institutions to share digital objects in a standardized way.

**Presentation API 3.0** defines how to describe and structure a digital object:

```
Manifest (the digital object)
  ├─ id: manifest URL
  ├─ label: human-readable title (multilingual)
  ├─ metadata: list of {label, value} pairs (creator, date, rights, etc.)
  ├─ thumbnail: array of image services
  └─ items: list of Canvases (pages/views)
       └─ Canvas
            ├─ id: canvas URL
            ├─ width, height: pixel dimensions
            ├─ label: page label (e.g., "f. 3r")
            └─ items: AnnotationPage
                 └─ Annotation (motivation: "painting")
                      └─ body: Image resource (IIIF Image API service URL)
```

**Image API 3.0** provides a RESTful URL pattern for image delivery:

```
{server}/{prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}
```

- `region`: `full` | `square` | `x,y,w,h` (pixels) | `pct:x,y,w,h` (percent)
- `size`: `max` | `w,` | `,h` | `w,h` | `pct:n`
- `rotation`: `0` | `90` | `180` | `270`
- `quality`: `default` | `color` | `gray` | `bitonal`

### W3C Web Annotation Model

Scholarly annotations on IIIF resources follow the W3C Web Annotation Data Model:

```json
{
  "@type": "Annotation",
  "motivation": "commenting",
  "target": {
    "source": "https://example.org/canvas/1",
    "selector": {"type": "FragmentSelector", "value": "xywh=100,200,300,400"}
  },
  "body": {"type": "TextualBody", "value": "This figure depicts..."}
}
```

Motivations: `painting` (the primary image), `commenting`, `tagging`,
`transcribing`, `describing`, `bookmarking`.

---

## Environment Setup

```bash
conda create -n iiif-env python=3.11 -y
conda activate iiif-env
pip install "requests>=2.28" "Pillow>=10.0" "pandas>=1.5" "matplotlib>=3.6"

# Test connectivity to a public IIIF endpoint
python -c "
import requests
url = 'https://iiif.wellcomecollection.org/presentation/v3/b18035723'
r = requests.get(url, timeout=10)
print(f'Status: {r.status_code}, Type: {r.headers.get(\"Content-Type\", \"\")}')
"
```

---

## Core Workflow

### Step 1 — Manifest Fetch, Canvas List, and Metadata Extraction

```python
import requests
import pandas as pd
from typing import Dict, List, Optional


def fetch_manifest(manifest_url: str, timeout: int = 30) -> dict:
    """
    Fetch a IIIF Presentation API 3.0 (or 2.x) manifest from a URL.

    Args:
        manifest_url: Full URL to the manifest JSON.
        timeout:      Request timeout in seconds.

    Returns:
        Parsed manifest as a Python dict.

    Raises:
        requests.HTTPError: If the server returns a non-2xx response.
    """
    headers = {
        'Accept': 'application/ld+json;profile="http://iiif.io/api/presentation/3/context.json",'
                  'application/json',
        'User-Agent': 'iiif-research-client/1.0',
    }
    response = requests.get(manifest_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_manifest_metadata(manifest: dict) -> dict:
    """
    Extract key metadata fields from a IIIF Presentation API manifest.

    Handles both v2.x and v3.0 label/metadata structures.

    Args:
        manifest: Parsed manifest dict.

    Returns:
        Dict with keys: title, creator, date, rights, description,
        canvas_count, manifest_id, api_version.
    """

    def _label_to_str(label) -> str:
        """Convert IIIF label (str, list, or language map) to plain string."""
        if label is None:
            return ''
        if isinstance(label, str):
            return label
        if isinstance(label, list):
            return str(label[0]) if label else ''
        if isinstance(label, dict):
            # language map: {'en': ['Title'], 'none': ['Title']}
            for lang in ('en', 'none', 'fr', 'de'):
                if lang in label:
                    v = label[lang]
                    return v[0] if isinstance(v, list) else str(v)
            # any language
            first_val = next(iter(label.values()))
            return first_val[0] if isinstance(first_val, list) else str(first_val)
        return str(label)

    title = _label_to_str(manifest.get('label', ''))
    api_version = '3.0' if manifest.get('@context', '').endswith('/3/context.json') else '2.x'

    # Parse metadata block
    metadata_raw = manifest.get('metadata', [])
    meta_dict: Dict[str, str] = {}
    for item in metadata_raw:
        k = _label_to_str(item.get('label', ''))
        v = _label_to_str(item.get('value', ''))
        meta_dict[k.lower()] = v

    # Canvas count
    items = manifest.get('items', manifest.get('sequences', [{}]))
    if isinstance(items, list) and items:
        if api_version == '3.0':
            canvas_count = len(items)
        else:
            # v2: sequences[0].canvases
            canvas_count = len(items[0].get('canvases', []))
    else:
        canvas_count = 0

    return {
        'title': title,
        'creator': meta_dict.get('creator', meta_dict.get('author', meta_dict.get('artist', ''))),
        'date': meta_dict.get('date', meta_dict.get('year', '')),
        'rights': manifest.get('rights', meta_dict.get('rights', meta_dict.get('license', ''))),
        'description': _label_to_str(manifest.get('summary', manifest.get('description', ''))),
        'canvas_count': canvas_count,
        'manifest_id': manifest.get('id', manifest.get('@id', '')),
        'api_version': api_version,
    }


def list_canvases(manifest: dict) -> List[dict]:
    """
    Return a list of canvas descriptors from a manifest.

    Args:
        manifest: Parsed IIIF manifest dict.

    Returns:
        List of dicts: {canvas_id, label, width, height, image_url}
    """
    api_version = '3.0' if '3/context.json' in manifest.get('@context', '') else '2.x'
    canvases = []

    if api_version == '3.0':
        items = manifest.get('items', [])
        for canvas in items:
            image_url = _extract_image_url_v3(canvas)
            canvases.append({
                'canvas_id': canvas.get('id', ''),
                'label': _label_to_str(canvas.get('label', '')),
                'width': canvas.get('width', 0),
                'height': canvas.get('height', 0),
                'image_url': image_url,
            })
    else:
        for seq in manifest.get('sequences', []):
            for canvas in seq.get('canvases', []):
                image_url = _extract_image_url_v2(canvas)
                canvases.append({
                    'canvas_id': canvas.get('@id', ''),
                    'label': canvas.get('label', ''),
                    'width': canvas.get('width', 0),
                    'height': canvas.get('height', 0),
                    'image_url': image_url,
                })
    return canvases


def _extract_image_url_v3(canvas: dict) -> str:
    """Extract painting annotation body URL from a v3 canvas."""
    for anno_page in canvas.get('items', []):
        for anno in anno_page.get('items', []):
            if anno.get('motivation') == 'painting':
                body = anno.get('body', {})
                if isinstance(body, list):
                    body = body[0]
                # Try IIIF Image API service
                for svc in body.get('service', []):
                    if isinstance(svc, dict) and 'id' in svc:
                        return svc['id'] + '/full/max/0/default.jpg'
                return body.get('id', '')
    return ''


def _extract_image_url_v2(canvas: dict) -> str:
    """Extract image resource URL from a v2 canvas."""
    for image in canvas.get('images', []):
        resource = image.get('resource', {})
        svc = resource.get('service', {})
        if '@id' in svc:
            return svc['@id'] + '/full/max/0/default.jpg'
        return resource.get('@id', '')
    return ''


# --- Demo ---
manifest_url = 'https://iiif.wellcomecollection.org/presentation/v3/b18035723'
manifest = fetch_manifest(manifest_url)
meta = extract_manifest_metadata(manifest)
print(f"Title:    {meta['title']}")
print(f"Creator:  {meta['creator']}")
print(f"Date:     {meta['date']}")
print(f"Canvases: {meta['canvas_count']}")

canvases = list_canvases(manifest)
df_canvases = pd.DataFrame(canvases)
print(df_canvases[['label', 'width', 'height']].head(10).to_string(index=False))
```

### Step 2 — Image Region Download via IIIF Image API

```python
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional


def build_iiif_image_url(
    service_base: str,
    region: str = 'full',
    size: str = 'max',
    rotation: int = 0,
    quality: str = 'default',
    fmt: str = 'jpg',
) -> str:
    """
    Construct a IIIF Image API 3.0 URL.

    Args:
        service_base: Base URL of the image service (no trailing slash).
        region:       'full', 'square', or 'x,y,w,h' (pixel coords) or 'pct:x,y,w,h'.
        size:         'max', 'w,' (width-only), ',h' (height-only), or 'w,h'.
        rotation:     0, 90, 180, or 270.
        quality:      'default', 'color', 'gray', or 'bitonal'.
        fmt:          'jpg', 'png', 'tif', 'webp'.

    Returns:
        Full IIIF Image API URL string.
    """
    service_base = service_base.rstrip('/')
    return f"{service_base}/{region}/{size}/{rotation}/{quality}.{fmt}"


def download_iiif_region(
    service_base: str,
    x: int,
    y: int,
    w: int,
    h: int,
    output_size: str = '800,',
    output_path: Optional[str] = None,
) -> Image.Image:
    """
    Download a specific pixel region from a IIIF Image API endpoint.

    Args:
        service_base: Base URL of the IIIF image service.
        x, y:         Top-left corner of region in canvas pixels.
        w, h:         Width and height of region in canvas pixels.
        output_size:  Desired output size (IIIF size parameter).
        output_path:  If set, save the image to this file path.

    Returns:
        PIL Image of the requested region.
    """
    region = f"{x},{y},{w},{h}"
    url = build_iiif_image_url(service_base, region=region, size=output_size)
    print(f"Fetching: {url}")

    headers = {'User-Agent': 'iiif-research-client/1.0'}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content))

    if output_path:
        img.save(output_path)
        print(f"Saved region to: {output_path}")

    return img


def thumbnail_grid(
    image_urls: List[str],
    labels: Optional[List[str]] = None,
    thumb_size: str = '150,',
    cols: int = 4,
    output_path: str = 'thumbnail_grid.png',
) -> None:
    """
    Download thumbnails and display as a grid plot.

    Args:
        image_urls: List of IIIF image URLs (full image or region URLs).
        labels:     Optional list of caption strings.
        thumb_size: IIIF size parameter for thumbnails.
        cols:       Number of columns in the grid.
        output_path: Where to save the grid image.
    """
    images = []
    for url in image_urls:
        try:
            # Replace size parameter in URL or use as-is
            thumb_url = url.replace('/full/max/', f'/full/{thumb_size}/').replace(
                '/full/full/', f'/full/{thumb_size}/')
            r = requests.get(thumb_url, timeout=20, headers={'User-Agent': 'iiif-research/1.0'})
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: could not load {url}: {e}")
            images.append(Image.new('RGB', (150, 150), color=(200, 200, 200)))

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis('off')
        if labels and i < len(labels):
            ax.set_title(labels[i], fontsize=7, wrap=True)

    for ax in axes[len(images):]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Thumbnail grid saved to: {output_path}")


# --- Demo ---
# service_base = 'https://iiif.wellcomecollection.org/image/b18035723_0001.jp2'
# region_img = download_iiif_region(service_base, x=200, y=300, w=500, h=400,
#                                   output_path='detail_region.jpg')
# region_img.show()
```

### Step 3 — Annotation Layer Parsing and Text Extraction

```python
import requests
import pandas as pd
from typing import List, Dict, Optional


def parse_annotation_pages(manifest: dict) -> List[dict]:
    """
    Extract all W3C Web Annotations from all canvases in a manifest.

    Searches for annotation pages with motivations other than 'painting'
    (i.e., scholarly comments, transcriptions, tags, descriptions).

    Args:
        manifest: Parsed IIIF Presentation API manifest dict.

    Returns:
        List of annotation dicts: {canvas_id, motivation, target_selector,
        body_text, body_language, body_type}
    """
    annotations = []
    api_v3 = '3/context.json' in manifest.get('@context', '')

    canvases = manifest.get('items', []) if api_v3 else \
        manifest.get('sequences', [{}])[0].get('canvases', [])

    for canvas in canvases:
        canvas_id = canvas.get('id', canvas.get('@id', ''))

        # v3: canvas.items = painting annotation pages; canvas.annotations = commentary pages
        anno_pages = canvas.get('annotations', [])
        # Also check items for non-painting annotations
        for anno_page in canvas.get('items', []):
            for anno in anno_page.get('items', []):
                if anno.get('motivation', '') != 'painting':
                    anno_pages.append(anno_page)
                    break

        for anno_page in anno_pages:
            # Annotation page may be a URL reference — fetch if needed
            if isinstance(anno_page, str):
                try:
                    r = requests.get(anno_page, timeout=15,
                                     headers={'User-Agent': 'iiif-research/1.0'})
                    anno_page = r.json()
                except Exception:
                    continue

            page_items = anno_page.get('items', anno_page.get('resources', []))
            for anno in page_items:
                motivation = anno.get('motivation', 'unknown')
                if motivation == 'painting':
                    continue

                # Parse target selector
                target = anno.get('target', '')
                selector = ''
                if isinstance(target, dict):
                    sel = target.get('selector', {})
                    if isinstance(sel, dict):
                        selector = sel.get('value', sel.get('@value', ''))
                    elif isinstance(sel, list) and sel:
                        selector = sel[0].get('value', '')

                # Parse body
                body = anno.get('body', {})
                if isinstance(body, list):
                    body = body[0] if body else {}

                body_text = ''
                body_lang = ''
                body_type = ''
                if isinstance(body, dict):
                    body_type = body.get('type', body.get('@type', ''))
                    body_text = body.get('value', body.get('@value', ''))
                    body_lang = body.get('language', '')
                elif isinstance(body, str):
                    body_text = body

                annotations.append({
                    'canvas_id': canvas_id,
                    'motivation': motivation,
                    'target_selector': selector,
                    'body_text': body_text,
                    'body_language': body_lang,
                    'body_type': body_type,
                })

    return annotations


def search_manifests_by_metadata(
    manifest_urls: List[str],
    search_term: str,
    fields: List[str] = None,
) -> pd.DataFrame:
    """
    Fetch multiple manifests and return those whose metadata contains a search term.

    Args:
        manifest_urls: List of IIIF manifest URLs to query.
        search_term:   Case-insensitive string to search for.
        fields:        Metadata fields to search (default: title, creator, description).

    Returns:
        DataFrame of matching manifests with their metadata.
    """
    if fields is None:
        fields = ['title', 'creator', 'description']

    results = []
    for url in manifest_urls:
        try:
            manifest = fetch_manifest(url, timeout=15)
            meta = extract_manifest_metadata(manifest)
            term_lower = search_term.lower()
            match = any(
                term_lower in str(meta.get(f, '')).lower()
                for f in fields
            )
            if match:
                meta['manifest_url'] = url
                results.append(meta)
        except Exception as e:
            print(f"Warning: could not fetch {url}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


# --- Demo ---
manifest = fetch_manifest('https://iiif.wellcomecollection.org/presentation/v3/b18035723')
annotations = parse_annotation_pages(manifest)
df_anno = pd.DataFrame(annotations)
if not df_anno.empty:
    print(f"Found {len(df_anno)} annotations")
    print(df_anno[['motivation', 'target_selector', 'body_text']].head(10).to_string(index=False))
else:
    print("No non-painting annotations found in this manifest.")
```

---

## Advanced Usage

### Cross-Institution Manifest Search

```python
INSTITUTION_ENDPOINTS = {
    'wellcome': 'https://iiif.wellcomecollection.org/presentation/v3/{}',
    'harvard': 'https://iiif.harvardartmuseums.org/manifests/object/{}',
    'europeana': 'https://iiif.europeana.eu/presentation/{}/manifest',
}


def multi_institution_search(
    identifiers: Dict[str, List[str]],
    search_term: str,
) -> pd.DataFrame:
    """
    Search for a term across manifests from multiple institutions.

    Args:
        identifiers: Dict mapping institution key to list of object IDs.
        search_term: Term to search in manifest metadata.

    Returns:
        Combined DataFrame of matching manifests with institution column.
    """
    all_urls = []
    institution_map = {}

    for inst, ids in identifiers.items():
        template = INSTITUTION_ENDPOINTS.get(inst, '')
        for obj_id in ids:
            url = template.format(obj_id)
            all_urls.append(url)
            institution_map[url] = inst

    df = search_manifests_by_metadata(all_urls, search_term)
    if not df.empty:
        df['institution'] = df['manifest_url'].map(institution_map)
    return df
```

### Extract Image Service URL from Canvas

```python
def get_image_service_url(manifest: dict, canvas_index: int = 0) -> Optional[str]:
    """
    Extract the IIIF Image API service base URL for a specific canvas.

    Returns:
        Service base URL (suitable for constructing image region URLs).
    """
    canvases = list_canvases(manifest)
    if not canvases or canvas_index >= len(canvases):
        return None

    canvas = canvases[canvas_index]
    img_url = canvas.get('image_url', '')
    # Strip the path suffix to get service base
    if '/full/' in img_url:
        return img_url.split('/full/')[0]
    return img_url
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `403 Forbidden` | Institution requires authentication or referrer header | Add `Referer: https://yoursite.org` header; some endpoints require OAuth |
| `JSONDecodeError` on manifest | Server returns HTML error page | Check `response.status_code`; try adding `Accept: application/json` |
| Empty `items` list | Manifest is IIIF v2.x not v3.0 | Check `@context` URL; use `sequences[0].canvases` for v2 |
| Image region `400 Bad Request` | Region coords exceed canvas dimensions | Query canvas `width`/`height` first and clamp coordinates |
| Annotation page is a URL not inline | Lazy-loaded annotation pages | Fetch annotation page URL with `requests.get()` separately |
| Slow manifest fetching | Large manifests or slow institution servers | Cache manifests locally with `json.dump()` / `json.load()` |

---

## External Resources

- IIIF Presentation API 3.0: <https://iiif.io/api/presentation/3.0/>
- IIIF Image API 3.0: <https://iiif.io/api/image/3.0/>
- W3C Web Annotation Data Model: <https://www.w3.org/TR/annotation-model/>
- IIIF Cookbook (recipes and examples): <https://iiif.io/api/cookbook/>
- Europeana APIs: <https://apis.europeana.eu/api/record-v3>
- Wellcome Collection IIIF: <https://developers.wellcomecollection.org/api/iiif>
- Harvard Art Museums API: <https://github.com/harvardartmuseums/api-docs>
- iiif-prezi3 Python library: <https://github.com/iiif-prezi/iiif-prezi3>

---

## Examples

### Example 1 — Download First Five Canvases as Thumbnails

```python
manifest_url = 'https://iiif.wellcomecollection.org/presentation/v3/b18035723'
manifest = fetch_manifest(manifest_url)
meta = extract_manifest_metadata(manifest)
print(f"Loading: {meta['title']} ({meta['canvas_count']} canvases)")

canvases = list_canvases(manifest)
urls = [c['image_url'] for c in canvases[:8] if c['image_url']]
labels = [c['label'] for c in canvases[:8]]

thumbnail_grid(urls, labels=labels, thumb_size='200,', cols=4,
               output_path='wellcome_thumbnails.png')
print("Grid saved.")
```

### Example 2 — Extract and Export All Transcription Annotations

```python
import json

manifest = fetch_manifest('https://iiif.wellcomecollection.org/presentation/v3/b18035723')
annotations = parse_annotation_pages(manifest)
transcriptions = [a for a in annotations if a['motivation'] in ('transcribing', 'commenting', 'describing')]

if transcriptions:
    df = pd.DataFrame(transcriptions)
    df.to_csv('annotations_export.csv', index=False)
    print(f"Exported {len(df)} annotations to annotations_export.csv")
    print(df[['motivation', 'body_text']].head(5).to_string(index=False))
else:
    print("No transcription annotations found. Try a manuscript manifest with editorial markup.")

# Save full annotation dump as JSON
with open('annotations_raw.json', 'w', encoding='utf-8') as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)
print("Raw annotations saved to annotations_raw.json")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — manifest parsing, region download, annotation extraction |
