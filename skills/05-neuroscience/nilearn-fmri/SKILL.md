---
name: nilearn-fmri
description: >
  fMRI neuroimaging analysis with nilearn: GLM, resting-state functional connectivity,
  ICA, MVPA decoding, and brain map visualization.
tags:
  - neuroscience
  - fmri
  - neuroimaging
  - nilearn
  - connectivity
  - mvpa
version: "1.0.0"
authors:
  - name: "awesome-rosetta-skills contributors"
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - nilearn>=0.10.3
  - nibabel>=5.2.0
  - numpy>=1.24.0
  - scipy>=1.11.0
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  - matplotlib>=3.7.0
  - joblib>=1.3.0
last_updated: "2026-03-17"
---

# fMRI Neuroimaging with nilearn

This skill covers the full neuroimaging analysis pipeline using nilearn and nibabel:
task-based fMRI general linear models (GLMs), second-level group analyses,
resting-state functional connectivity, independent component analysis (ICA),
multivariate pattern analysis (MVPA) with support vector machines, and
publication-quality brain map visualization.

## Prerequisites

```bash
pip install nilearn nibabel numpy scipy pandas scikit-learn matplotlib joblib
```

For GPU-accelerated ICA (optional):

```bash
pip install cuml-cu12  # RAPIDS cuML, CUDA 12.x
```

## Core Functions

### 1. First-Level GLM (Single Subject, Task fMRI)

```python
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import clean_img, concat_imgs, mean_img, index_img
from nilearn import plotting


def run_first_level_glm(
    fmri_img,
    events_df: pd.DataFrame,
    confounds: pd.DataFrame | None = None,
    t_r: float = 2.0,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 0.01,
    smoothing_fwhm: float = 6.0,
    noise_model: str = "ar1",
    standardize: bool = False,
) -> FirstLevelModel:
    """
    Fit a first-level GLM for a single-subject task fMRI run.

    Parameters
    ----------
    fmri_img : Nifti1Image or str
        4D fMRI image or path to NIfTI file.
    events_df : pd.DataFrame
        Events table with columns: 'onset', 'duration', 'trial_type'.
        All times in seconds from the start of the scan.
    confounds : pd.DataFrame | None
        Motion parameters and other nuisance regressors. Column names
        become regressor names in the design matrix.
    t_r : float
        Repetition time in seconds.
    hrf_model : str
        HRF model name: 'spm', 'glover', 'spm + derivative'.
    drift_model : str
        Low-frequency drift model: 'cosine', 'polynomial', None.
    high_pass : float
        High-pass cutoff in Hz (used only when drift_model='cosine').
    smoothing_fwhm : float
        Spatial smoothing kernel FWHM in mm (0 to disable).
    noise_model : str
        Temporal autocorrelation model: 'ar1', 'ols'.
    standardize : bool
        Whether to standardize each voxel's time series.

    Returns
    -------
    FirstLevelModel
        Fitted nilearn FirstLevelModel. Access .design_matrices_,
        .r_square_, and use .compute_contrast() for stat maps.
    """
    if isinstance(fmri_img, str):
        fmri_img = nib.load(fmri_img)

    n_scans = fmri_img.shape[3]
    frame_times = np.arange(n_scans) * t_r

    # Build design matrix
    confound_array = confounds.values if confounds is not None else None
    confound_names = confounds.columns.tolist() if confounds is not None else None

    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        add_regs=confound_array,
        add_reg_names=confound_names,
    )

    glm = FirstLevelModel(
        t_r=t_r,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        smoothing_fwhm=smoothing_fwhm if smoothing_fwhm > 0 else None,
        noise_model=noise_model,
        standardize=standardize,
        verbose=1,
    )
    glm.fit(fmri_img, events_df, confounds=confounds)
    return glm


def plot_contrast(
    stat_map,
    threshold: float = 3.0,
    title: str = "Contrast Map",
    display_mode: str = "ortho",
    colorbar: bool = True,
    output_file: str | None = None,
):
    """
    Plot a statistical contrast map as both glass brain and slice overlay.

    Parameters
    ----------
    stat_map : Nifti1Image or str
        Statistical map (z-score or t-map).
    threshold : float
        Voxel-level display threshold (z or t value).
    title : str
        Figure title.
    display_mode : str
        Orientation for plot_stat_map: 'ortho', 'z', 'x', 'y'.
    colorbar : bool
        Whether to show colorbar.
    output_file : str | None
        If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Glass brain
    display1 = plotting.plot_glass_brain(
        stat_map,
        threshold=threshold,
        colorbar=colorbar,
        title=f"{title} (glass brain, threshold={threshold})",
        axes=axes[0],
        plot_abs=False,
    )

    # Stat map on MNI template
    display2 = plotting.plot_stat_map(
        stat_map,
        threshold=threshold,
        colorbar=colorbar,
        title=f"{title} (MNI overlay)",
        display_mode=display_mode,
        axes=axes[1],
    )

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_file}")
    plt.show()
```

### 2. Second-Level Group Analysis

```python
from nilearn.glm.second_level import SecondLevelModel


def run_second_level_onesample(
    first_level_imgs: list,
    smoothing_fwhm: float = 8.0,
    contrast_id: str = "intercept",
) -> dict:
    """
    One-sample t-test at the group level (second-level GLM).

    Parameters
    ----------
    first_level_imgs : list
        List of subject-level contrast NIfTI images (same contrast for all).
    smoothing_fwhm : float
        Additional group-level smoothing in mm.
    contrast_id : str
        Label for the output stat map.

    Returns
    -------
    dict
        Keys: 'z_map', 't_map', 'design_matrix', 'model'.
    """
    n_subjects = len(first_level_imgs)
    design_matrix = pd.DataFrame(
        {"intercept": np.ones(n_subjects)}
    )

    second_level_model = SecondLevelModel(
        smoothing_fwhm=smoothing_fwhm,
        verbose=1,
    )
    second_level_model.fit(first_level_imgs, design_matrix=design_matrix)

    z_map = second_level_model.compute_contrast(
        second_level_contrast=contrast_id,
        output_type="z_score",
    )
    t_map = second_level_model.compute_contrast(
        second_level_contrast=contrast_id,
        output_type="stat",
    )

    return {
        "z_map": z_map,
        "t_map": t_map,
        "design_matrix": design_matrix,
        "model": second_level_model,
    }
```

### 3. Resting-State Functional Connectivity

```python
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal


def extract_roi_timeseries(
    fmri_img,
    atlas_name: str = "schaefer200",
    t_r: float = 2.0,
    low_pass: float | None = 0.1,
    high_pass: float | None = 0.01,
    smoothing_fwhm: float = 6.0,
    detrend: bool = True,
    standardize: bool = True,
    confounds: pd.DataFrame | None = None,
) -> tuple:
    """
    Extract ROI time series using a brain atlas masker.

    Parameters
    ----------
    fmri_img : Nifti1Image or str
        4D resting-state fMRI image.
    atlas_name : str
        Atlas to use: 'schaefer200', 'schaefer100', 'aal'.
    t_r : float
        Repetition time in seconds (needed for bandpass filtering).
    low_pass : float | None
        Low-pass frequency cutoff in Hz.
    high_pass : float | None
        High-pass frequency cutoff in Hz.
    smoothing_fwhm : float
        Spatial smoothing in mm.
    detrend : bool
        Whether to detrend the time series.
    standardize : bool
        Whether to standardize each ROI time series to unit variance.
    confounds : pd.DataFrame | None
        Confound regressors (e.g., motion, WM, CSF signals).

    Returns
    -------
    tuple
        (time_series ndarray shape [n_timepoints, n_rois],
         atlas_labels list of str,
         masker NiftiLabelsMasker)
    """
    if isinstance(fmri_img, str):
        fmri_img = nib.load(fmri_img)

    if atlas_name.startswith("schaefer"):
        n_rois = int(atlas_name.replace("schaefer", "")) if atlas_name != "schaefer200" else 200
        atlas = fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
        atlas_img = atlas.maps
        atlas_labels = atlas.labels.tolist()
    elif atlas_name == "aal":
        atlas = fetch_atlas_aal()
        atlas_img = atlas.maps
        atlas_labels = atlas.labels
    else:
        raise ValueError(f"Unknown atlas: {atlas_name}. Use 'schaefer200', 'schaefer100', or 'aal'.")

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=standardize,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        smoothing_fwhm=smoothing_fwhm,
        verbose=1,
    )

    confound_array = confounds.values if confounds is not None else None
    time_series = masker.fit_transform(fmri_img, confounds=confound_array)

    return time_series, atlas_labels, masker


def compute_fc_matrix(
    time_series: np.ndarray,
    kind: str = "correlation",
) -> np.ndarray:
    """
    Compute a functional connectivity matrix from ROI time series.

    Parameters
    ----------
    time_series : np.ndarray
        Shape [n_timepoints, n_rois].
    kind : str
        Connectivity measure: 'correlation', 'partial correlation',
        'tangent', 'covariance', 'precision'.

    Returns
    -------
    np.ndarray
        Symmetric connectivity matrix, shape [n_rois, n_rois].
    """
    measure = ConnectivityMeasure(kind=kind)
    fc_matrix = measure.fit_transform([time_series])[0]
    np.fill_diagonal(fc_matrix, 0)
    return fc_matrix
```

### 4. ICA Decomposition

```python
from nilearn.decomposition import CanICA


def run_canica(
    fmri_imgs: list,
    n_components: int = 20,
    smoothing_fwhm: float = 6.0,
    threshold: float = 3.0,
    random_state: int = 42,
    n_jobs: int = 1,
):
    """
    Run CanICA group-level spatial ICA decomposition.

    Parameters
    ----------
    fmri_imgs : list
        List of 4D NIfTI images or file paths (one per subject/session).
    n_components : int
        Number of ICA components to extract.
    smoothing_fwhm : float
        Spatial smoothing kernel FWHM in mm.
    threshold : float
        Threshold for component maps (z-score).
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Number of parallel jobs for multi-subject decomposition.

    Returns
    -------
    CanICA
        Fitted CanICA object. Access .components_img_ for component maps.
    """
    canica = CanICA(
        n_components=n_components,
        smoothing_fwhm=smoothing_fwhm,
        threshold=threshold,
        random_state=random_state,
        verbose=1,
        n_jobs=n_jobs,
        memory_level=2,
    )
    canica.fit(fmri_imgs)
    return canica
```

### 5. MVPA SVM Decoding

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from nilearn.decoding import SearchLight
from nilearn.image import get_data
from nilearn.maskers import NiftiMasker


def run_mvpa_svm(
    fmri_imgs: list,
    labels: list,
    mask,
    cv_folds: int = 5,
    kernel: str = "linear",
    C: float = 1.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> dict:
    """
    MVPA classification using SVM with cross-validation.

    Parameters
    ----------
    fmri_imgs : list
        List of 3D NIfTI images (one per trial/sample).
    labels : list
        Class labels corresponding to each image.
    mask : Nifti1Image or str
        Binary brain mask.
    cv_folds : int
        Number of stratified cross-validation folds.
    kernel : str
        SVM kernel: 'linear' (recommended for neuroimaging) or 'rbf'.
    C : float
        SVM regularization parameter.
    n_jobs : int
        Parallel jobs for cross-validation.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Keys: accuracy_mean, accuracy_std, cv_scores, pipeline, masker.
    """
    if isinstance(mask, str):
        mask = nib.load(mask)

    masker = NiftiMasker(
        mask_img=mask,
        standardize=True,
        detrend=False,
        verbose=0,
    )

    # Extract voxel patterns
    X = masker.fit_transform(fmri_imgs)
    y = np.array(labels)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, random_state=random_state)),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        return_train_score=True,
    )

    return {
        "accuracy_mean": cv_results["test_score"].mean(),
        "accuracy_std": cv_results["test_score"].std(),
        "cv_scores": cv_results["test_score"],
        "train_accuracy_mean": cv_results["train_score"].mean(),
        "pipeline": pipeline,
        "masker": masker,
    }
```

## Example 1: Visual Localizer GLM — Face vs. Object Contrast

This example fits a first-level GLM to a visual localizer paradigm and computes
a face-minus-object contrast map, then runs a second-level one-sample t-test
across subjects.

```python
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs

# ------------------------------------------------------------------
# Load example data: Haxby face/object dataset
# (automatically downloaded to ~/nilearn_data/)
# ------------------------------------------------------------------
haxby_dataset = datasets.fetch_haxby(n_subjects=2)
print("Haxby dataset loaded. Functional files:", haxby_dataset.func)

subject_z_maps = []

for subj_idx in range(len(haxby_dataset.func)):
    fmri_file = haxby_dataset.func[subj_idx]
    mask_file = haxby_dataset.mask_vt[subj_idx]
    session_targets = pd.read_csv(
        haxby_dataset.session_target[subj_idx], sep=" "
    )

    # Build events dataframe from labels
    t_r = 2.5
    frame_times = np.arange(len(session_targets)) * t_r

    # Keep only face and house conditions; ignore rest
    condition_mask = session_targets["labels"].isin(["face", "house"])
    onsets = frame_times[condition_mask.values]
    trial_types = session_targets["labels"].values[condition_mask.values]
    events = pd.DataFrame({
        "onset": onsets,
        "duration": np.ones(len(onsets)) * 9.0,
        "trial_type": trial_types,
    })

    # Build motion confounds (use zeros if unavailable)
    n_scans = nib.load(fmri_file).shape[3]
    confounds = pd.DataFrame(
        np.zeros((n_scans, 6)),
        columns=["tx", "ty", "tz", "rx", "ry", "rz"],
    )

    # Fit first-level GLM
    glm = run_first_level_glm(
        fmri_img=fmri_file,
        events_df=events,
        confounds=confounds,
        t_r=t_r,
        smoothing_fwhm=6.0,
        hrf_model="spm",
    )

    # Compute face > house contrast
    contrast_map = glm.compute_contrast(
        contrast_def="face - house",
        output_type="z_score",
    )
    subject_z_maps.append(contrast_map)

    print(f"Subject {subj_idx+1}: contrast map computed, "
          f"shape={contrast_map.shape}")

# ------------------------------------------------------------------
# Second-level: one-sample t-test across subjects
# ------------------------------------------------------------------
group_results = run_second_level_onesample(
    first_level_imgs=subject_z_maps,
    smoothing_fwhm=8.0,
)

print(f"\nGroup z-map shape: {group_results['z_map'].shape}")

# ------------------------------------------------------------------
# Visualize
# ------------------------------------------------------------------
plot_contrast(
    stat_map=group_results["z_map"],
    threshold=2.3,
    title="Group: Face > House",
    display_mode="z",
    output_file="face_vs_house_group.png",
)

# Design matrix inspection
from nilearn.plotting import plot_design_matrix
fig, ax = plt.subplots(figsize=(10, 4))
plot_design_matrix(glm.design_matrices_[0], ax=ax)
ax.set_title("First-Level Design Matrix (last subject)")
plt.tight_layout()
plt.savefig("design_matrix.png", dpi=150)
plt.show()
```

## Example 2: Resting-State Parcellation and Group FC Matrix Comparison

This example extracts ROI time series with the Schaefer-200 atlas, computes
individual and group-average functional connectivity matrices, and compares
FC matrices between two groups.

```python
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from nilearn import datasets, plotting
from nilearn.connectome import ConnectivityMeasure

# ------------------------------------------------------------------
# Load example resting-state data (ADHD200 dataset, first 20 subjects)
# ------------------------------------------------------------------
adhd_dataset = datasets.fetch_adhd(n_subjects=20)
func_imgs = adhd_dataset.func
confounds_list = adhd_dataset.confounds
phenotypic = adhd_dataset.phenotypic

print(f"Loaded {len(func_imgs)} resting-state scans")
print(f"Phenotypic info: {pd.DataFrame(phenotypic).head()}")

# ------------------------------------------------------------------
# Extract ROI time series for each subject
# ------------------------------------------------------------------
t_r = 2.5
all_time_series = []

for i, (img_path, conf_path) in enumerate(zip(func_imgs, confounds_list)):
    print(f"Processing subject {i+1}/{len(func_imgs)}...")

    # Load confounds if available
    if conf_path:
        confounds_df = pd.read_csv(conf_path, sep="\t").fillna(0)
        # Keep motion and tissue signal columns only
        keep_cols = [c for c in confounds_df.columns
                     if any(k in c.lower() for k in
                            ["motion", "white_matter", "csf", "rot", "trans"])]
        confounds_df = confounds_df[keep_cols] if keep_cols else None
    else:
        confounds_df = None

    ts, labels, masker = extract_roi_timeseries(
        fmri_img=img_path,
        atlas_name="schaefer200",
        t_r=t_r,
        low_pass=0.1,
        high_pass=0.01,
        smoothing_fwhm=6.0,
        confounds=confounds_df,
    )
    all_time_series.append(ts)

print(f"Time series shapes: {[ts.shape for ts in all_time_series[:3]]}")

# ------------------------------------------------------------------
# Compute individual FC matrices
# ------------------------------------------------------------------
fc_matrices = []
for ts in all_time_series:
    fc = compute_fc_matrix(ts, kind="correlation")
    fc_matrices.append(fc)

fc_stack = np.stack(fc_matrices, axis=0)  # shape: [n_subjects, n_rois, n_rois]

# ------------------------------------------------------------------
# Split into control vs ADHD groups (based on phenotypic label)
# ------------------------------------------------------------------
pheno_df = pd.DataFrame(phenotypic)
# ADHD200 uses 0=control, 1=ADHD inattentive, 2=ADHD combined
is_control = pheno_df["adhd"].values == 0

ctrl_fc = fc_stack[is_control].mean(axis=0)
adhd_fc = fc_stack[~is_control].mean(axis=0)

# T-test at each edge (uncorrected; apply FDR in practice)
n_rois = ctrl_fc.shape[0]
t_matrix = np.zeros((n_rois, n_rois))
p_matrix = np.ones((n_rois, n_rois))

ctrl_indiv = fc_stack[is_control]
adhd_indiv = fc_stack[~is_control]

for i in range(n_rois):
    for j in range(i + 1, n_rois):
        t_val, p_val = stats.ttest_ind(ctrl_indiv[:, i, j], adhd_indiv[:, i, j])
        t_matrix[i, j] = t_val
        t_matrix[j, i] = t_val
        p_matrix[i, j] = p_val
        p_matrix[j, i] = p_val

sig_edges = (p_matrix < 0.05).sum() // 2
print(f"Significant FC differences (p<0.05, uncorrected): {sig_edges} edges")

# ------------------------------------------------------------------
# Visualize FC matrices
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Show only first 50 ROIs for readability
n_show = 50

im0 = axes[0].imshow(ctrl_fc[:n_show, :n_show], cmap="RdBu_r",
                      vmin=-0.8, vmax=0.8)
axes[0].set_title(f"Control Group FC\n(n={is_control.sum()}, first {n_show} ROIs)")
axes[0].set_xlabel("ROI")
axes[0].set_ylabel("ROI")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(adhd_fc[:n_show, :n_show], cmap="RdBu_r",
                      vmin=-0.8, vmax=0.8)
axes[1].set_title(f"ADHD Group FC\n(n={(~is_control).sum()}, first {n_show} ROIs)")
axes[1].set_xlabel("ROI")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

diff_fc = ctrl_fc - adhd_fc
im2 = axes[2].imshow(diff_fc[:n_show, :n_show], cmap="RdBu_r",
                      vmin=-0.3, vmax=0.3)
axes[2].set_title(f"FC Difference\n(Control - ADHD, first {n_show} ROIs)")
axes[2].set_xlabel("ROI")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

plt.suptitle("Resting-State Functional Connectivity: Schaefer-200 Atlas", fontsize=13)
plt.tight_layout()
plt.savefig("resting_state_fc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# ICA: Run CanICA on all subjects
# ------------------------------------------------------------------
print("\nRunning CanICA...")
canica = run_canica(
    fmri_imgs=func_imgs,
    n_components=20,
    smoothing_fwhm=6.0,
    n_jobs=2,
)

components_img = canica.components_img_
print(f"CanICA complete. Component image shape: {components_img.shape}")

# Visualize first 9 components
plotting.plot_prob_atlas(
    components_img,
    title="CanICA: Resting-State Networks",
    view_type="filled_contours",
)
plt.savefig("canica_components.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# MVPA: Decode ADHD vs. control from mean FC patterns
# ------------------------------------------------------------------
# Vectorize upper triangle of each FC matrix
def fc_to_vector(fc_mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(fc_mat, k=1)
    return fc_mat[idx]

X_fc = np.stack([fc_to_vector(fc) for fc in fc_matrices])
y_labels = ["control" if is_ctl else "adhd"
            for is_ctl in is_control]

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0, random_state=42)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_fc, y_labels, cv=cv, scoring="accuracy")

print(f"\nMVPA (SVM on FC features): {scores.mean():.3f} ± {scores.std():.3f} accuracy")
print(f"Individual fold accuracies: {np.round(scores, 3)}")
```

## Notes and Best Practices

- **Data format**: nilearn expects NIfTI images in MNI152 space (2mm or 1mm). Use fMRIPrep for preprocessing.
- **Confound selection**: Always regress out at least 6 motion parameters, WM, and CSF signals in resting-state analyses.
- **Multiple comparisons**: Apply FDR or cluster-level correction (nilearn threshold_stats_img) before interpreting GLM results.
- **HRF models**: Use 'spm + derivative' for better temporal modeling. Use 'glover' for GE scanners.
- **ICA thresholding**: CanICA's default threshold=3 is reasonable. Use dual-regression for subject-level component maps.
- **MVPA**: Linear SVMs are preferred for high-dimensional neuroimaging; avoid RBF without careful CV-based hyperparameter selection.
- **Atlas choice**: Schaefer-200 provides a good balance between resolution and signal quality. AAL is useful for clinical ROI studies.
- **Memory**: 4D fMRI images can be very large. Use `nilearn.image.index_img` to slice time points and avoid loading full datasets into RAM.
- **SearchLight**: For whole-brain MVPA, use `nilearn.decoding.SearchLight`; it is computationally intensive — parallelize with `n_jobs=-1`.
