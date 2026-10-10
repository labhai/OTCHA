# Dataset preparation

Dataset images, annotation tables, and original split assignments are not included.
Prepare them locally according to the formats below and use paths relative to
`--data_dir` or absolute paths. Keep train/validation/test patient groups disjoint.
The JSON split keys must be `train`, `val`, and `test`, with nonempty lists.

Prepare patient-disjoint splits for your own training. Reproducing the reported
results requires the same preprocessing, labels, and split assignments as the paper;
the original split assignments are not bundled.

## VinDr-Mammo

Official source: [VinDr-Mammo v1.0.0 on PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/).
Follow the dataset page's access instructions to obtain the images and annotations.

The loader consumes `breast-level_annotations.csv` with at least these columns:

| Column | Meaning |
| --- | --- |
| `study_id` | Study identifier |
| `image_id` | Image identifier |
| `laterality` | `L` or `R` |
| `view_position` | `CC` or `MLO` |
| `breast_birads` | BI-RADS value, e.g. `BI-RADS 2` |

The loader maps BI-RADS 1–2 to class 0 and BI-RADS 3–5 to class 1. Unsupported
values are excluded. In `2view` mode each sample is one breast with CC and MLO;
in `4view` mode each sample contains `L-CC`, `L-MLO`, `R-CC`, `R-MLO`, and its label
is the maximum of the four view labels. Incomplete view groups are skipped.

Put readable grayscale images at `images/<study_id>/<image_id>.png` under the data
directory. The loader uses Pillow, so convert source DICOMs to PNG/JPEG first using
the preprocessing chosen for your experiment. The paper additionally uses ROI
cropping and background suppression. Prepare those images before using this
loader to reproduce the paper. The loader does not decode DICOMs or apply DICOM
windowing. Preserve image orientation and identifiers when converting.

`splits.json` accepts either lists of study IDs (see
[the fictional example](../examples/vindr_splits.json)) or lists of objects:

```json
{
  "train": [{"patient_id": "example-train-study", "laterality": "L"}],
  "val": [{"patient_id": "example-validation-study", "laterality": "L"}],
  "test": [{"patient_id": "example-test-study", "laterality": "L"}]
}
```

In the object format, `patient_id` refers to the CSV's `study_id`. Include both
breasts in each split when using four-view training. A simple
list of study IDs includes both breasts automatically.

## CheXpert

Official source: [Stanford CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/).
Use its official download link to obtain the images. The paper uses labels from
[Stanford VisualCheXbert](https://github.com/stanfordmlgroup/VisualCheXbert).

The loader expects a **study-level** `chexpert_label.csv`, rather than a raw
per-image annotation CSV:

| Column | Meaning |
| --- | --- |
| `ID` | `<patient_id>_<study_id>`, with the last underscore separating the study |
| `images` | Python list serialized as a string, containing frontal and lateral paths |
| Label columns | The 13 label columns listed below |

Example `images` value:

```text
['train/patient00001/study1/view1_frontal.jpg', 'train/patient00001/study1/view2_lateral.jpg']
```

Label columns:

```text
Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
Pleural Effusion, Pleural Other, Fracture, Support Devices
```

The implementation predicts 13 outputs, excludes `No Finding`, and uses U-Zeros:
only a label value of 1 maps to positive; uncertain, unannotated, and other values map
to 0. Prepare study-level labels using the policy selected for your experiment.
The paper uses VisualCheXbert labels; use the corresponding study-level label
table when reproducing its results.

Group raw image rows by patient/study, retain one frontal/lateral pair per study,
and serialize the image paths as a list. Filenames containing `frontal` and
`lateral` define view ordering. When neither keyword exists, the loader uses the
first two paths in the list as frontal and lateral, respectively. Studies with
an incomplete pair are skipped.

`splits.json` accepts lists of `ID` strings (see
[the fictional example](../examples/chexpert_splits.json)) or objects containing
`patient_id` and `study`, matching the two parts of `ID`. Keep all studies from
one patient in the same split.

## MURA

Official source: [Stanford MURA](https://stanfordmlgroup.github.io/competitions/mura/).
Use its official download link to obtain the images and study labels.

Prepare `mura_train.csv`, `mura_valid.csv`, and `mura_test.csv`, each with these columns:

| Column | Meaning |
| --- | --- |
| `patient_id` | Integer patient identifier |
| `study` | Integer study number |
| `part` | Body part, e.g. `XR_WRIST` |
| `images` | Serialized Python list of image paths |
| `target` | 0 for normal, 1 for abnormal |

See [the fictional example](../examples/mura_studies.csv). Group source image paths
by patient, study, and body part, and join the study label to each group. For the
provided two-view preset, every row must contain **exactly two** image paths so
that the default DataLoader can stack samples. The paper selects studies with at
least two views and samples two images per study. Prepare and record that selection
in the CSV before training; the loader does not pad or truncate variable-length
studies.

Paths are resolved against `--data_dir`. With `--data_dir data/MURA-v1.1`, a path
should start with `train/` or `valid/`, without another `MURA-v1.1/` prefix. Keep
all examinations from a patient in one split and document how you obtained the
three evaluation partitions from your source data.

## Image transforms

All three loaders read grayscale images, resize to the configured image size,
and expand them to three channels for the supplied backbone. They normalize
with mean 0.5 and standard deviation 0.5. Training also uses horizontal flips
and random affine augmentation. CLAHE is optional and defaults to off.
