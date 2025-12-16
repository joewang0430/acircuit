# Project Structure Overview

This repo contains data processing scripts, training code, and models for circuit diagram segmentation and pin heatmap recognition.

- `backend/`
	- `app/`
		- `datasets/`: Dataset loaders and utilities (e.g., tiles dataset for U‑Net training).
		- `models/`: Model definitions (U‑Net for segmentation, lightweight U‑Net for pin heatmaps, classifiers, etc.).
		- `train/`: Training scripts and runners (e.g., `train_unet.py`).
		- `core/`: Core configs or shared assets (e.g., `label_map.json`).
	- `scripts/`: Data preparation tools (unify annotations, overlays, tiling, and split/repartition scripts).
	- `requirements.in`: Top‑level direct Python dependencies (unpinned). Use pip‑tools to compile.
	- `requirements.txt`: Locked dependency versions for reproducible installs.

- `data/`
	- `processed/`: Generated datasets, tiles, masks, and manifests used for training.
	- `raw/` (optional): Original source datasets before unification.

- `runs/`: Training outputs (checkpoints, metrics, logs) organized by experiment name.

- `.venv/`: Project virtual environment (Python) — not committed.

- `.vscode/`, `.idea/`: Editor/IDE settings (optional).

Quick notes
- Use `backend/app/train/train_unet.py` for segmentation training with the tiles manifest.
- Pin heatmap model is in `backend/app/models/pin_unet.py` with helpers for Gaussian target and argmax prediction.
- Data tiling and split manifests are under `data/processed/tiles/annotations/`.

