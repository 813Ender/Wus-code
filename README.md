# DAMN

A PyTorch multi-task classification model for slice-level `lesion` and slice-level `time`. During evaluation, predictions are aggregated at the case level.

## Setup

Use Python `3.12` and a virtual environment named `wus`:

```bash
python3.12 -m venv wus
source wus/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Note: `torch/torchvision` should match your CUDA setup (if you use a GPU). ImageNet pretrained weights may be downloaded automatically on the first run.

## Data format

Recommended directory layout:

```text
<base_path>/
  train/
    <case_...>/
      DWI/
        *.dcm
      FLAIR/
        *.dcm
  val/
    <case_...>/
      DWI/
        *.dcm
      FLAIR/
        *.dcm
  test/
    <case_...>/
      DWI/
        *.dcm
      FLAIR/
        *.dcm
```

Requirements:

- Each `case` must contain both `DWI/` and `FLAIR/`.
- The same slice must exist in both folders with the same `.dcm` filename (matching is done by filename intersection).
- `lesion_label` is derived from the slice filename: if the filename (case-insensitive) contains `x`, then `lesion_label=1` (lesion-present slice); otherwise `lesion_label=0`.
- `time_label` is derived from the `case` folder name. The code parses a number via `case.split("min")[0].split("_")[-1]`; if the number is `>= 270`, then `time_label=1`, else `time_label=0`.

## Train

```bash
python main.py --base_path ../data/ --output_dir output
```


