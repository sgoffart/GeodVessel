# CorSegRec

CorSegRec is a small Python project for vessel-centerline classification and disconnected-vessel reconnection in 3D medical volumes. The repository combines:

- a centerline classifier based on a random-forest model (`CFC`),
- feature extraction from multiscale 3D patches,
- a DPC-based pathfinding workflow to reconnect separated vessel components.

## Project structure

- `train_cfc.py` — trains the centerline classifier from ASOCA-style volumes and labels.
- `run_dpc.py` — loads a segmentation, identifies disconnected components, and runs the DPC reconnection workflow.
- `check.py` — performs a sanity check on the trained model and extracted features.
- `models/cfc_model.py` — lightweight wrapper around `scikit-learn` random forest.
- `dpc/dpc_walk.py` — DPC pathfinding implementation.
- `sampling/` — utilities for extracting training points.
- `utils/` — geometry and patch helpers.

## Requirements

This project targets Python 3.11+.

Install the dependencies with:

```bash
pip install -e .
```

or, if you prefer the requirements file:

```bash
pip install -r requirements.txt
```

## Training the classifier

The training script expects ASOCA-style data in the dataset directory and writes the trained model to the path defined in `train_cfc.py`.

Run:

```bash
python train_cfc.py
```

## Running the DPC reconnection demo

The script `run_dpc.py` loads a segmentation volume, finds candidate disconnected components, and attempts to reconnect them using DPC pathfinding.

Run:

```bash
python run_dpc.py
```

The output includes a Plotly HTML visualization saved under the ASOCA data directory.

## Model sanity check

To validate the saved model and the feature pipeline:

```bash
python check.py
```

## Notes

- The scripts currently use hard-coded dataset paths under the local ASOCA directory. Adjust those paths if you run the project from a different environment.
- The repository is intended as a research-oriented prototype rather than a packaged production tool.
