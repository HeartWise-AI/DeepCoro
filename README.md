# DeepCoro

Run coronary angiography stenosis inference from DICOM videos and manually
annotated artery views and stenosis boxes. The original anatomic classification
and stenosis detection models (algorithms 1 and 2) are unavailable; annotations
remain the input to algorithms 3–6.

## Install

Use Python 3.10 or newer (the container uses Python 3.11):

```sh
python -m pip install .
# Development checks:
python -m pip install -e '.[test]'
python -m pytest
```

PyTorch and torchvision must be a matching pair. Install the appropriate PyTorch
build for your CUDA setup before installing this package if necessary.
`segmentation-models-pytorch` stays at 0.3.3 to preserve the published checkpoint
architectures; direct dependency ranges replace the old full environment freeze.
Model constructors do not download unrelated ImageNet/Kinetics initialization
weights.

## Inputs

Create an annotation CSV (its filename is unrestricted):

```csv
dicom_path,artery_view,frame,x1,y1,x2,y2
dicom1.dcm,RCA,50,122,214,211,303
dicom1.dcm,RCA,62,195,84,284,173
dicom2.dcm,LCA,21,30,105,51,159
```

Relative DICOM paths resolve against the CSV's directory. Absolute paths also
work. `frame` is a zero-based frame index; boxes use pixel coordinates with
exclusive upper bounds (`x1 < x2`, `y1 < y2`). Artery views must be RCA or LCA.
Each DICOM must contain grayscale angiography frames acquired at 15 FPS and:

- Imager Pixel Spacing, Distance Source To Detector, Distance Source To Patient;
- Recommended Display Frame Rate;
- Study Date and Patient's Birth Date.

Compressed DICOMs may require a pydicom decoding plugin appropriate to their
transfer syntax; these optional codecs are not bundled with the package.

## Run inference

```sh
deepcoro --input-file-path dcm_input/input_file.csv --save-dir results
# Equivalent source/module entry points:
python -m deepcoro --input-file-path dcm_input/input_file.csv --save-dir results
python main.py --input_file_path dcm_input/input_file.csv --save_dir results
```

The CLI chooses CUDA when available, otherwise CPU. Use `--device cpu` to select
CPU explicitly and `--no-videos` to write only the result CSV.

The output directory is created automatically. `df_stenosis.csv` contains
`dicom_path`, `video_path`, `frame`, `box`, `box_resized`, `artery_segment`,
`percent_stenosis`, and `severe_stenosis`. Registered annotated MP4s accompany the
CSV unless disabled. An empty result still produces a CSV with these columns.
The published severe stenosis decision threshold remains 0.23.

## Models on Hugging Face

The original weights are already published at
[heartwise/DeepCoro](https://huggingface.co/heartwise/DeepCoro).
Inference now downloads only the seven segmentation checkpoints and the
stenosis regressor through `huggingface_hub`, using its shared cache. Models
are separate from the Python package and Docker image; no Git LFS clone is
needed. The default snapshot is pinned to
[`13207073d84d6c605ecb9bfa95195458796d7e39`](https://huggingface.co/heartwise/DeepCoro/tree/13207073d84d6c605ecb9bfa95195458796d7e39)
for reproducibility. The eight files total about 1.39 GB; the unrelated
`structure_recon` checkpoint is excluded.

Pre-download weights before an offline run:

```sh
deepcoro-download --cache-dir /path/to/hf-cache
deepcoro --input-file-path dcm_input/input_file.csv --save-dir results \
  --cache-dir /path/to/hf-cache --local-files-only
```

The official repository is gated: accept its access conditions on Hugging Face
and authenticate with `hf auth login` or `HF_TOKEN` before downloading.
`HF_HOME` and standard Hugging Face authentication are respected. Use
`--repo-id namespace/repository --revision COMMIT_OR_TAG` to select another
compatible model repository with the same `models/algorithm4/...` and
`models/algorithm6.pt` layout. The official repository defaults to the pinned
commit; another repository defaults to `main` unless a revision is supplied.

Existing local weights remain supported without any Hub access:

```sh
deepcoro --input-file-path dcm_input/input_file.csv --save-dir results \
  --models-dir /path/to/models
```

`--models-dir` accepts the directory containing `algorithm4/` and `algorithm6.pt`,
or an existing clone containing a `models/` subdirectory. Missing, empty, and
unfetched Git LFS pointer files fail before inference starts.

## Docker

```sh
docker build -t deepcoro_inference .
docker run --rm \
  -v /path/to/dcm_input:/dcm_input:ro \
  -v /path/to/results:/results \
  -v /path/to/hf-cache:/cache/huggingface \
  deepcoro_inference
```

Add `--gpus all` to `docker run` for GPU inference. The persistent cache mount
reuses downloaded weights between containers. CLI arguments after the image
name replace the default arguments, for example:

```sh
docker run --rm \
  -v /path/to/dcm_input:/dcm_input:ro \
  -v /path/to/results:/results \
  -v /path/to/hf-cache:/cache/huggingface \
  deepcoro_inference --input-file-path /dcm_input/annotations.csv \
    --save-dir /results --local-files-only
```

## Code organization

`deepcoro.pipeline` orchestrates inference; `dicom` handles inputs and metadata;
`geometry`, `preprocessing`, and `segments` implement image/segment operations;
`models` owns network construction and checkpoint loading; `hub` resolves
weights; and `cli` provides the command line. `main.py` and `utils.py` retain
compatibility entry points for existing scripts in the source checkout. Only the
`deepcoro` package is installed, keeping generic `utils`/`main` modules out of
site-packages so the neighboring CLIP project can coexist.

### Reference
-------------
Labrecque Langlais, E., Corbin, D., Tastet, O., Hayek, A., Doolub, G., Mrad, S., Tardif, J.-C., Tanguay, J.-F., Marquis-Gravel, G., Tison, G., Kadoury, S., Le, W., Gallo, R., Lesage, F, & Avram, R. Evaluation of Stenoses Using AI Video Models Applied to Coronary Angiographies, 23 November 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3610879/v1]

```
@article{avram2023evaluation,
  title={Evaluation of Stenoses Using AI Video Models Applied to Coronary Angiographies},
  author={Labrecque Langlais, E. and Corbin, D. and Tastet, O. and Hayek, A. and Doolub, G. and Mrad, S. and Tardif, J.-C. and Tanguay, J.-F. and Marquis-Gravel, G. and Tison, G. and Kadoury, S. and Le, W. and Gallo, R. and Lesage, F. and Avram, R.},
  year={2023}
}
```
