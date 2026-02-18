import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from stereo_matcher import StereoMatcher
from visual_odometry import VisualOdometry
from utils import (load_kitti_images, load_kitti_gt, load_calibration, 
                   compute_errors, load_vo_calib, load_poses, 
                   compute_ate, compute_rpe)

def run_phase1():
    print("\n=== Phase 1: Dense Depth from Stereo ===")
    data_dir = 'data_scene_flow'
    calib_dir = 'calib_cam_to_cam'
    output_dir = 'output/stereo'
    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    window_size = 15
    max_disp = 128
    methods = ['SAD', 'SSD', 'NCC']
    # Process all available images (0-20)
    indices = list(range(21))

    matcher = StereoMatcher(window_size=window_size, max_disp=max_disp)

    for method in methods:
        print(f"\nEvaluating Method: {method}")
        total_bpr = 0
        total_mae = 0
        count = 0

        # Create method-specific output dir
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        for idx in tqdm(indices):
            try:
                img_l, img_r, img_color = load_kitti_images(data_dir, index=idx)
                gt = load_kitti_gt(data_dir, index=idx)
                fx, baseline = load_calibration(calib_dir, index=idx)
            except FileNotFoundError:
                continue

            disp_l = matcher.compute_disparity(img_l, img_r, method=method)
            disp_r = matcher.compute_disparity(img_l, img_r, method=method, right_reference=True)
            consistent_disp = matcher.left_right_consistency_check(disp_l, disp_r, threshold=1.0)
            final_disp = matcher.post_process(consistent_disp)

            depth = np.zeros_like(final_disp)
            mask = final_disp > 0
            depth[mask] = (fx * baseline) / final_disp[mask]

            if gt is not None:
                bpr, mae = compute_errors(gt, final_disp)
                total_bpr += bpr
                total_mae += mae
                count += 1
                
                # Save visualizations for every 5th sample to avoid disk clutter, or selective
                if idx % 5 == 0:
                    plt.figure(figsize=(12, 8))
                    plt.subplot(2, 2, 1); plt.title("Left Image"); plt.imshow(img_color)
                    plt.subplot(2, 2, 2); plt.title(f"Disparity ({method})"); plt.imshow(final_disp, cmap='jet')
                    plt.subplot(2, 2, 3); plt.title("Ground Truth Disparity"); plt.imshow(gt, cmap='jet')
                    plt.subplot(2, 2, 4); plt.title("Depth Map"); plt.imshow(np.clip(depth, 0, 80), cmap='magma')
                    plt.tight_layout()
                    plt.savefig(os.path.join(method_dir, f"sample_{idx:06d}.png"))
                    plt.close()

        if count > 0:
            print(f"Results for {method}: Avg BPR: {total_bpr/count:.2f}%, Avg MAE: {total_mae/count:.2f} px")

def run_phase2():
    print("\n=== Phase 2: Stereo Visual Odometry ===")
    sequence_dir = '00'
    poses_path = 'poses/00.txt'
    output_dir = 'output/vo'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(poses_path):
        print("Ground truth poses not found. Skipping evaluation.")
        return

    calib = load_vo_calib(os.path.join(sequence_dir, 'calib.txt'))
    gt_poses = load_poses(poses_path)
    vo = VisualOdometry(calib['P0'], calib['P1'])
    
    img_l_dir = os.path.join(sequence_dir, 'image_0')
    img_r_dir = os.path.join(sequence_dir, 'image_1')
    img_files = sorted(os.listdir(img_l_dir))
    
    num_frames = len(img_files)
    estimated_poses = []
    
    print(f"Processing {num_frames} frames...")
    for i in tqdm(range(num_frames)):
        img_l = cv2.imread(os.path.join(img_l_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(os.path.join(img_r_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None: break
        
        pose, _ = vo.process_frame(img_l, img_r)
        estimated_poses.append(pose)

    gt_subset = gt_poses[:len(estimated_poses)]
    ate = compute_ate(gt_subset, estimated_poses)
    rpe = compute_rpe(gt_subset, estimated_poses)
    
    print(f"VO Results: ATE: {ate:.4f} m, RPE: {rpe:.4f} m/frame")
    
    plt.figure(figsize=(8, 8))
    plt.plot([p[0, 3] for p in gt_subset], [p[2, 3] for p in gt_subset], label='GT')
    plt.plot([p[0, 3] for p in estimated_poses], [p[2, 3] for p in estimated_poses], label='VO', linestyle='--')
    plt.legend(); plt.title("Trajectory Ground Plane"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'trajectory.png'))
    plt.close()

def main():
    run_phase1()
    run_phase2()

if __name__ == "__main__":
    main()
