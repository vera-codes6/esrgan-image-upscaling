# Real-ESRGAN

Practical super-resolution for real-world images and videos. This repo provides ready-to-use Python scripts for inference on photos and videos, optional face enhancement with GFPGAN, and references for training/finetuning.

- Paper: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (ICCVW 2021)
- Codebase: Built on BasicSR; models and options adapted from the original Real-ESRGAN project

## Features

- High-quality 4x and 2x upscaling for general and anime images
- Tiny general model with adjustable denoise strength (-dn) to avoid over-smoothing
- Video upscaling with audio passthrough and multi-GPU split/merge
- Alpha channel, grayscale, and 16-bit image support
- Optional face enhancement via GFPGAN

## Quick start (Windows PowerShell)

1) Create an environment and install dependencies

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip
# Install PyTorch appropriate for your CUDA/CPU from https://pytorch.org/get-started/locally/
# Example (CUDA/Windows may vary): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python setup.py develop
```

2) Put input files

- Images in `inputs/` (default) or pass a path with `-i`
- Videos in `inputs/video/` (example path)

3) Run image inference

```powershell
# General 4x model
python .\inference_realesrgan.py -i .\inputs -o .\results -n RealESRGAN_x4plus -s 4

# General tiny model with denoise control (0=keep more noise, 1=strong denoise)
python .\inference_realesrgan.py -i .\inputs -o .\results -n realesr-general-x4v3 -dn 0.5

# Anime image model
python .\inference_realesrgan.py -i .\inputs -o .\results -n RealESRGAN_x4plus_anime_6B

# Optional: face enhancement
python .\inference_realesrgan.py -i .\inputs -o .\results -n RealESRGAN_x4plus --face_enhance

# If you have limited VRAM, process with tiles (try 200, 150, 100 ...)
python .\inference_realesrgan.py -i .\inputs -o .\results -n RealESRGAN_x4plus -t 200 --tile_pad 10
```

4) Run video inference

```powershell
# Anime-oriented tiny video model (default)
python .\inference_realesrgan_video.py -i .\inputs\video\clip.mp4 -o .\results -n realesr-animevideov3 -s 2 --fps 30

# General models also work for videos
python .\inference_realesrgan_video.py -i .\inputs\video\clip.mp4 -o .\results -n RealESRGAN_x4plus -s 4
```

Notes

- Pretrained weights are auto-downloaded on first use to `weights/`. You can also place `.pth` files there in advance.
- GPU is used if available. To reduce VRAM usage, increase `--tile` or run with FP32 (`--fp32`).
- The `--outscale` option resizes the final output to an arbitrary scale after model inference.

## Models overview

Image scripts (`inference_realesrgan.py`) support the following model names via `-n/--model_name`:

- RealESRGAN_x4plus (4x RRDB, general images)
- RealESRNet_x4plus (4x RRDB, slightly smoother results)
- RealESRGAN_x2plus (2x RRDB)
- RealESRGAN_x4plus_anime_6B (4x RRDB, compact, anime illustrations)
- realesr-animevideov3 (4x VGG-style, very small, anime/video)
- realesr-general-x4v3 (4x VGG-style tiny general model, supports `-dn` denoise strength)

Video script (`inference_realesrgan_video.py`) supports the same names and will pass audio through when present.

Common flags

- `-i/--input` image file, video file, or folder (default: `inputs`)
- `-o/--output` output folder (default: `results`)
- `-s/--outscale` final upsampling scale (e.g., 2, 3.5, 4)
- `-t/--tile` tile size to avoid OOM; `--tile_pad` padding between tiles
- `--pre_pad` border pre-padding; `--fp32` use full precision
- `--face_enhance` enable GFPGAN face restoration
- `--ext` output extension policy; `--alpha_upsampler` strategy for alpha channels
- `-g/--gpu-id` select which GPU to use (image script); leave unset to auto-pick

## Troubleshooting

- CUDA out of memory: lower `-t` (tile size) or set `--fp32`; you can also reduce `-s` or use `--outscale` for final resizing
- CPU-only: the code will fallback to CPU if CUDA is not available; expect slower inference
- FFmpeg not found (video): ensure `ffmpeg` is on PATH or pass `--ffmpeg_bin` with an absolute path

## Training and finetuning

- See `docs/Training.md` and `options/*.yml` for training/finetuning recipes
- Paired/unpaired datasets are supported (see dataset YAMLs in `tests/data` and `realesrgan/data`)

## Directory structure

- `inference_realesrgan.py` image inference
- `inference_realesrgan_video.py` video inference
- `weights/` pretrained weights (auto-downloaded)
- `docs/` guides, FAQs, model zoo
- `realesrgan/` architectures, datasets, models, utilities

## License and citation

- Code is released under the BSD 3-Clause License (see `LICENSE`)

If this project helps your research or product, please cite the Real-ESRGAN paper.

```bibtex
@inproceedings{wang2021realesrgan,
  title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author    = {Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle = {ICCV Workshops},
  year      = {2021}
}
```

## Acknowledgements

This implementation builds on the original Real-ESRGAN project and the BasicSR toolbox. Thanks to all contributors of the upstream repositories and related libraries (GFPGAN, facexlib, OpenCV, PyTorch, tqdm, Pillow, NumPy).