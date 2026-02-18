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

def generate_stereo_examples():
    print("\n--- Generating 10 Stereo Examples ---")
    data_dir = 'data_scene_flow'
    calib_dir = 'calib_cam_to_cam'
    output_dir = 'output/report/stereo'
    os.makedirs(output_dir, exist_ok=True)
    
    indices = list(range(10))
    matcher = StereoMatcher(window_size=15, max_disp=128) # better results from ablation

    for idx in tqdm(indices):
        img_l, img_r, img_color = load_kitti_images(data_dir, index=idx)
        fx, baseline = load_calibration(calib_dir, index=idx)
        
        disp = matcher.compute_disparity(img_l, img_r, method='SSD')
        disp = matcher.post_process(matcher.left_right_consistency_check(disp, matcher.compute_disparity(img_l, img_r, method='SSD', right_reference=True)))
        
        depth = np.zeros_like(disp)
        mask = disp > 0
        depth[mask] = (fx * baseline) / disp[mask]
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].imshow(img_color); ax[0].set_title("Input"); ax[0].axis('off')
        ax[1].imshow(disp, cmap='jet'); ax[1].set_title("Disparity"); ax[1].axis('off')
        ax[2].imshow(np.clip(depth, 0, 80), cmap='magma'); ax[2].set_title("Depth"); ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"example_{idx:02d}.png"))
        plt.close()

def generate_vo_results():
    print("\n--- Generating VO Results for 2 Sequences ---")
    sequences = ['03', '04']
    output_dir = 'output/report/vo'
    os.makedirs(output_dir, exist_ok=True)
    
    for seq in sequences:
        print(f"Processing Sequence {seq}")
        seq_dir = f'e:/university/master/semester-1/Computer Vision/project/{seq}'
        poses_path = f'poses/{seq}.txt'
        
        if not os.path.exists(poses_path):
            # Try sequence dir for cases where poses are kept there
            poses_path = os.path.join(seq_dir, 'poses.txt')
            if not os.path.exists(poses_path):
                print(f"Poses for {seq} not found. Skipping.")
                continue

        calib = load_vo_calib(os.path.join(seq_dir, 'calib.txt'))
        gt_poses = load_poses(poses_path)
        vo = VisualOdometry(calib['P0'], calib['P1'])
        
        img_l_dir = os.path.join(seq_dir, 'image_0')
        img_r_dir = os.path.join(seq_dir, 'image_1')
        img_files = sorted(os.listdir(img_l_dir))[:100] # Limit for time
        
        est_poses = []
        for i in tqdm(range(len(img_files))):
            img_l = cv2.imread(os.path.join(img_l_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
            img_r = cv2.imread(os.path.join(img_r_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
            pose, debug = vo.process_frame(img_l, img_r)
            est_poses.append(pose)
            
            if i % 10 == 1:
                matches_img = cv2.drawMatches(
                    debug['prev_img'], debug['prev_kp'],   # type: ignore
                    debug['curr_img'], debug['curr_kp'],  # type: ignore
                    [debug['matches'][idx] for idx in debug['inliers'][:100]], None,  # type: ignore
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imwrite(os.path.join(output_dir, f'matches_seq{seq}_frame{i}.png'), matches_img)

        # Plot trajectory
        plt.figure(figsize=(10, 10))
        plt.plot([p[0,3] for p in gt_poses[:len(est_poses)]], [p[2,3] for p in gt_poses[:len(est_poses)]], label='GT')
        plt.plot([p[0,3] for p in est_poses], [p[2,3] for p in est_poses], label='Estimated', linestyle='--')
        plt.legend(); plt.title(f'Trajectory Sequence {seq}'); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'trajectory_{seq}.png'))
        plt.close()

if __name__ == "__main__":
    generate_stereo_examples()
    generate_vo_results()
