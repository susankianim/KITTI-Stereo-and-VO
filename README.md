# KITTI Stereo Depth and Visual Odometry

This project implements a complete pipeline for stereo depth estimation and Visual Odometry (VO) using the KITTI dataset. It features dense disparity estimation with multiple cost functions and a feature-based VO system with metric scale recovery.

## Features

### 1. Stereo Depth Estimation
- **Matching Costs**: Implementation of SAD, SSD, and NCC.
- **Support**: Variable window sizes for block matching.
- **Post-processing**: Left-Right consistency check and hole filling.
- **Evaluation**: Bad Pixel Rate (BPR) and Mean Absolute Error (MAE) metrics.

### 2. Visual Odometry
- **Feature Pipeline**: ORB feature extraction and matching.
- **Pose Estimation**: Robust 3D-2D PnP trajectory estimation using RANSAC.
- **Modes**: Support for full metric scale (stereo-based) and monocular scale modes.
- **Analysis**: Full ablation studies on RANSAC impact and stereo vs. monocular scale.

## Installation

Ensure you have [uv](https://github.com/astral-sh/uv) or a standard Python environment installed.

```bash
# Clone the repository
git clone https://github.com/susankianim/KITTI-Stereo-and-VO
cd KITTI-Stereo-and-VO

# Using uv (recommended)
uv venv
uv sync

# Using pip (if requirements.txt exists)
# pip install -r requirements.txt
```

## Running the Project

### Stereo Evaluation
To run the stereo depth estimation pipeline and generate ablation results:
```bash
python run_stereo.py
```
This script evaluates different matching costs and window sizes, saving results to `stereo_results/`.

### Visual Odometry
To run the VO pipeline across multiple sequences:
```bash
python run_vo.py
```
This script processes sequences (defaulting to Seq 03), evaluates cases (Full, No RANSAC, No Stereo Scale), and saves plots to `vo_results/`.

## Project Structure
- `run_stereo.py`: Main entry point for depth estimation and stereo ablation studies.
- `run_vo.py`: Main entry point for visual odometry evaluation and trajectory plotting.
- `stereo_matcher.py`: Core logic for dense block matching and disparity calculation.
- `visual_odometry.py`: Dense-disparity-based VO implementation.
- `visual_odometry_feature.py`: Feature-based VO implementation (adapted for RANSAC and speed).
- `utils.py`: Data loaders, calibration parsing, and evaluation metric implementations (ATE, RPE, BPR).
- `REPORT.md`: Detailed technical report with results, diagrams, and failure analysis.

## Results
- **Stereo Results**: Disparity maps and metrics are stored in `stereo_results/`.
- **VO Results**: Trajectory plots and metric CSVs are stored in `vo_results/`.

For a full breakdown of the results and failure cases, please refer to [REPORT.md](REPORT.md).
