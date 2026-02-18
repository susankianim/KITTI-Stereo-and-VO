# KITTI Stereo Depth and Visual Odometry

This project implements a complete pipeline for dense depth estimation from stereo images and the recovery of camera trajectory using Visual Odometry (VO) on the KITTI dataset.

## Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
# Clone the repository
git clone <repo-url>
cd code

# Create virtual environment and install dependencies
uv venv
uv sync
```

## Dependencies
- `numpy`
- `opencv-python`
- `matplotlib`
- `tqdm`


## Running the Project

### Full Pipeline
To run the standard evaluation on both stereo depth and visual odometry:
```bash
uv run main.py
```
To run visual odometry with more results:
```bash
uv run main_vo.py
```


### Ablation Studies
To reproduce the ablation study results (different matching costs, window sizes, RANSAC impact):
```bash
uv run ablation_study.py
```

### Report Data Generation
To regenerate the 10+ disparity examples and trajectory plots used in the report:
```bash
uv run generate_report_data.py
```

## Project Structure
- `stereo_matcher.py`: Implementation of block matching (SAD, SSD, NCC), L-R consistency, and post-processing.
- `visual_odometry.py`: ORB-based feature matching, 3D-2D PnP pose estimation, and metric scale recovery.
- `utils.py`: Data loaders for KITTI labels, calibrations, and evaluation metrics (BPR, MAE, ATE, RPE).
- `main.py`: Main execution script.
- `output/`: Contains all generated visualizations and results.
