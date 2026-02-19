# Project Report: Stereo Depth and Visual Odometry

## 1. Pipeline Overview

```mermaid
graph TD
    subgraph Phase 1: Stereo Depth
    A1[Left & Right Images] --> B1[Block Matching SAD/SSD/NCC]
    B1 --> C1[Disparity Map]
    C1 --> D1[L-R Consistency Check]
    D1 --> E1[Median Filter & Hole Filling]
    E1 --> F1[Depth Map Z = f*B/d]
    end

    subgraph Phase 2: Visual Odometry
    A2[Frame t & t+1] --> B2[ORB Feature Extraction]
    B2 --> C2[BFMatcher Matching]
    C2 --> D2[Stereo Depth Back-projection 3D]
    D2 --> E2[PnP + RANSAC Pose Estimation]
    E2 --> F2[Trajectory Chaining]
    end

    F1 --> D2
```

## 2. Technical Explanation

### 2.1 Matching Costs
Matching cost functions measure the dissimilarity between a pixel in the left image $I_L(u, v)$ and a potential match in the right image $I_R(u-d, v)$ within a window $W$.

- **SAD (Sum of Absolute Differences)**:
  $$C_{SAD}(u, v, d) = \sum_{(i,j) \in W} |I_L(u+i, v+j) - I_R(u+i-d, v+j)|$$
  Simple and fast, but very sensitive to exposure differences between sensors.

- **SSD (Sum of Squared Differences)**:
  $$C_{SSD}(u, v, d) = \sum_{(i,j) \in W} (I_L(u+i, v+j) - I_R(u+i-d, v+j))^2$$
  Penalizes large errors more heavily than SAD, following a Gaussian noise model.

- **NCC (Normalized Cross-Correlation)**:
  $$C_{NCC}(u, v, d) = \frac{\sum_{(i,j) \in W} (I_L(u+j, v+j) - \bar{I}_L)(I_R(u+i-d, v+j) - \bar{I}_R)}{\sqrt{\sum (I_L - \bar{I}_L)^2 \sum (I_R - \bar{I}_R)^2}}$$
  Invariant to linear intensity changes ($I' = aI + b$), making it the most robust choice for KITTI imagery which often has slight exposure mismatches between the two cameras.

### 2.2 Pose Estimation and RANSAC
The system estimates camera motion using the following pipeline based on current code:

1.  **3D Point Generation**: Using the stereo baseline $B$ and focal length $f$, we triangulate features from the previous frame to create a 3D point cloud $P_{prev}$. Points are discarded if disparity $d=0$ or $d < d_{min}$.
2.  **3D-2D Correspondences**: Features are matched between the previous and current frame using ORB descriptors and a Brute-Force Hamming matcher.
3.  **Essential Matrix Pre-filtering**: Before PnP, `cv2.findEssentialMat` with RANSAC is used to filter out 2D-2D outliers that don't satisfy the epipolar geometry $x'^T E x = 0$.
4.  **PnP Solver**: We solve for $[R|t]$ using `cv2.solvePnPRansac`. This minimizes the reprojection error: 
    $$\min_{R, t} \sum_{i} \|x_i - \pi(R P_i + t)\|^2$$
5.  **RANSAC Role**: Both stages (Essential and PnP) use RANSAC to handle outliers. In `solvePnPRansac`, random subsets of 3 matches are picked to generate hypotheses, and the one with the highest consolidation of inliers (reprojection error < 2.0 or 1.0 depending on config) is selected. This is critical for ignoring dynamic objects like moving cars in the KITTI data.

### 2.3 Calibration Usage
The focal length $f$ (taken from $P[0,0]$) and horizontal baseline $B$ are extracted using:
- $B = \frac{|P1[0,3] - P0[0,3]|}{f}$ (or just $P1[0,3]$ if $P0$ is origin-centered).
- This ensures the 3D points $Z = \frac{f \cdot B}{d}$ are in meters, allowing for metric Visual Odometry.

## 3. Evaluation Results

### 3.1 Stereo Depth (Phase 1)
Results averaged over all available samples in `stereo_results/average_results.csv`:
| Method | Window Size | Bad-pixel Rate (>3px) | MAE (px) |
| :--- | :--- | :--- | :--- |
| **NCC** | 15 | 10.35% | 2.09 |
| **NCC** | 7 | 13.33% | 2.93 |
| **SSD** | 15 | 22.35% | 4.85 |
| **SSD** | 7 | 31.10% | 6.80 |
| **SAD** | 15 | 24.17% | 5.21 |
| **SAD** | 7 | 32.99% | 7.16 |

### 3.2 Visual Odometry (Phase 2)
Full results for Sequences 03 and 04, and "Full" mode for others:

| Sequence | Mode | ATE (m) | RPE (m/f) |
| :--- | :--- | :--- | :--- |
| **03** | **Full** | **8.16** | **0.027** |
| 03 | No RANSAC | 8.42e11 | 1.33e9 |
| 03 | No Stereo Scale | 96.95 | 0.301 |
| **04** | **Full** | **2.43** | **0.061** |
| 04 | No RANSAC | 488.93 | 69.27 |
| 04 | No Stereo Scale | 54.76 | 0.459 |
| **00** | Full | 58.81 | 0.033 |
| **01** | Full | 197.35 | 0.444 |
| **02** | Full | 76.51 | 0.038 |
| **05** | Full | 10.42 | 0.032 |
| **06** | Full | 18.04 | 0.066 |
| **07** | Full | 14.32 | 0.031 |
| **08** | Full | 50.51 | 0.038 |
| **09** | Full | 109.25 | 0.044 |
| **10** | Full | 9.35 | 0.029 |

## 4. Ablation Study Analysis

### 4.1 Depth Ablation (Cost Function and Window Size)
Quantitative comparison of matching costs and window sizes averaged over all data:

| Method | Window | Avg BPR (%) | Avg MAE (px) |
| :--- | :--- | :--- | :--- |
| **NCC** | **15** | **10.35** | **2.09** |
| NCC | 7 | 13.33 | 2.93 |
| SSD | 15 | 22.35 | 4.85 |
| SSD | 7 | 31.10 | 6.80 |
| SAD | 15 | 24.17 | 5.21 |
| SAD | 7 | 32.99 | 7.16 |

- **Observation**: NCC with a $15 \times 15$ window is clearly the winner. The $15 \times 15$ window reduces noise significantly (dropping BPR for NCC from 13.3% to 10.4%). NCC's ability to handle exposure differences is its main advantage over SAD/SSD.

### 4.2 VO Ablation (RANSAC and Scale)
Effect of components on metrics for Sequence 03:

| Config | ATE (m) | Effect |
| :--- | :--- | :--- |
| **Full** | **8.16** | Baseline performance |
| **No RANSAC** | 8.42e11 | Catastrophic failure due to outliers |
| **No Scale** | 96.95 | Loss of metric distance accuracy |

- **RANSAC Analysis**: As shown in the comparison plots, "No RANSAC" often deviates immediately when the car turns or dynamic objects appear.
- **Scale Analysis**: "No Stereo Scale" follows the path shape but "shrinks" or "stretches" the trajectory, illustrating the importance of stereo depth for scale recovery.

| Seq 03 Ablation Comparison (No Scale vs No RANSAC) |
| :---: |
| ![Ablation Scale](vo_results/trajectories/trajectory_03_No_Stereo_Scale.png) ![Ablation RANSAC](vo_results/trajectories/trajectory_03_No_RANSAC.png) |

## 5. Failure Cases

### 5.1 Stereo Depth Failure Cases
The 3 samples with the worst results (highest BPR):

| Sample Index | Method | BPR (%) | MAE (px) | Image Sample |
| :--- | :--- | :--- | :--- | :--- |
| **104** | NCC-15 | 72.35 | 19.99 | ![Worst 104](stereo_results/failue-results/stereo_idx104_NCC_ws15.png) |
| **006** | NCC-15 | 27.41 | 4.18 | ![Worst 006](stereo_results/failue-results/stereo_idx006_NCC_ws15.png) |
| **058** | NCC-15 | 26.29 | 3.51 | ![Worst 058](stereo_results/failue-results/stereo_idx058_NCC_ws15.png) |

**General Analysis of Failure Cases**:
The highest errors observed in samples like 104, 006, and 058 can be attributed to several recurring technical challenges:

- **Insufficient Illumination**: Low-light conditions and underexposed regions, particularly in shadows or during poor weather, lead to a low signal-to-noise ratio. This makes it difficult for the cost functions to find reliable correspondences, resulting in noisy depth maps.
- **Textureless Regions**: The cost functions (SAD, SSD, NCC) are highly sensitive to regions lacking distinctive structural variation, such as the sky or smooth asphalt. In these areas, multiple matches produce near-identical costs, leading to noise-dominated depth as seen in the top failure cases.
- **Occlusion and Boundary Errors**: Fixed-sized windows assume a constant depth for all pixels within the block. At object boundaries or in occluded regions (pixels visible in one camera but not the other), this assumption fails, leading to the "foreground fattening" effect and erroneous depth estimates.

### 5.2 Visual Odometry Failure Cases
Sequences with the worst "Full" mode ATE:

| Sequence | ATE (m) | Trajectory Plot |
| :--- | :--- | :--- |
| **01** | 197.35 | ![Fail 01](vo_results/trajectories/trajectory_01_Full.png) |
| **09** | 109.25 | ![Fail 09](vo_results/trajectories/trajectory_09_Full.png) |
| **02** | 76.51 | ![Fail 02](vo_results/trajectories/trajectory_02_Full.png) |

**VO Failure Analysis**:
- **Sequence 01**: High-speed highway driving. High motion blur and few nearby features lead to poor triangulation.
- **Sequence 09**: Contains many sharp turns. If features are lost during a fast rotation, the pose estimation accumulates significant drift.
- **Sequence 02**: Long distance leading to accumulated drift over time; specifically in areas with heavy foliage (moving leaves act as outliers).

## 6. Visual Results

### 6.1 Stereo Disparity Samples
| Index 000 (NCC ws15) | Index 100 (NCC ws15) |
| :---: | :---: |
| ![Stereo 000](stereo_results/stereo_idx000_NCC_ws15.png) | ![Stereo 100](stereo_results/stereo_idx100_NCC_ws15.png) |

### 6.2 VO Feature Matches
| Seq 03 Frame 1 | Seq 04 Frame 1 |
| :---: | :---: |
| ![Matches 03](vo_results/matches/matches_seq03_frame001.png) | ![Matches 04](vo_results/matches/matches_seq04_frame001.png) |

### 6.3 Trajectories
| Sequence 03 (Full vs Others) | Sequence 04 (Full vs Others) |
| :---: | :---: |
| ![Traj 03](vo_results/trajectories/trajectory_03_Full.png) | ![Traj 04](vo_results/trajectories/trajectory_04_Full.png) |
