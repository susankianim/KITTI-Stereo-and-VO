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

def run_stereo_ablation():
    print("\n--- Stereo Ablation ---")
    data_dir = 'data_scene_flow'
    calib_dir = 'calib_cam_to_cam'
    output_dir = 'output/ablation/stereo'
    os.makedirs(output_dir, exist_ok=True)

    window_sizes = [7, 15]
    methods = ['SAD', 'SSD', 'NCC']
    indices = [0, 5, 10, 15, 20] # 5 samples for ablation

    results = []

    for ws in window_sizes:
        for method in methods:
            print(f"Running WS={ws}, Method={method}")
            matcher = StereoMatcher(window_size=ws, max_disp=128)
            total_bpr = 0
            total_mae = 0
            count = 0
            for idx in indices:
                try:
                    img_l, img_r, _ = load_kitti_images(data_dir, index=idx)
                    gt = load_kitti_gt(data_dir, index=idx)
                    disp_l = matcher.compute_disparity(img_l, img_r, method=method)
                    disp_r = matcher.compute_disparity(img_l, img_r, method=method, right_reference=True)
                    final_disp = matcher.post_process(matcher.left_right_consistency_check(disp_l, disp_r))
                    
                    if gt is not None:
                        bpr, mae = compute_errors(gt, final_disp)
                        total_bpr += bpr
                        total_mae += mae
                        count += 1
                        
                        # Save some examples for report
                        if idx == 0:
                            plt.imsave(os.path.join(output_dir, f"disp_{method}_ws{ws}_idx{idx}.png"), final_disp, cmap='jet')
                except:
                    continue
            
            if count > 0:
                results.append({
                    'ws': ws, 'method': method, 
                    'avg_bpr': total_bpr/count, 'avg_mae': total_mae/count
                })

    print("\nStereo Ablation Results:")
    for r in results:
        print(f"WS: {r['ws']}, Method: {r['method']} -> BPR: {r['avg_bpr']:.2f}%, MAE: {r['avg_mae']:.2f}")

def run_vo_ablation():
    print("\n--- VO Ablation ---")
    sequence_dir = '00'
    poses_path = 'poses/00.txt'
    output_dir = 'output/ablation/vo'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(poses_path):
        return

    calib = load_vo_calib(os.path.join(sequence_dir, 'calib.txt'))
    gt_poses = load_poses(poses_path)
    
    configs = [
        {'ransac': True, 'stereo': True, 'label': 'Full (RANSAC + Stereo Scale)'},
        {'ransac': False, 'stereo': True, 'label': 'No RANSAC'},
        {'ransac': True, 'stereo': False, 'label': 'Monocular Unit Scale'}
    ]

    img_l_dir = os.path.join(sequence_dir, 'image_0')
    img_r_dir = os.path.join(sequence_dir, 'image_1')
    img_files = sorted(os.listdir(img_l_dir))[:50] # Limit to 50 for speed

    all_est_poses = []

    for cfg in configs:
        print(f"Running VO: {cfg['label']}")
        vo = VisualOdometry(calib['P0'], calib['P1'])
        est_poses = []
        for i in tqdm(range(len(img_files))):
            img_l = cv2.imread(os.path.join(img_l_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
            img_r = cv2.imread(os.path.join(img_r_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
            pose, debug = vo.process_frame(img_l, img_r, use_ransac=cfg['ransac'], use_stereo_scale=cfg['stereo'])
            est_poses.append(pose)
            
            # Save feature matches example for the "Full" config
            if cfg['ransac'] and i == 1:
                matches_img = cv2.drawMatches(
                    debug['prev_img'], debug['prev_kp'],   # type: ignore
                    debug['curr_img'], debug['curr_kp'],   # type: ignore
                    [debug['matches'][idx] for idx in debug['inliers']], None,   # type: ignore
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imwrite(os.path.join(output_dir, f'inlier_matches_{i}.png'), matches_img)

        all_est_poses.append(est_poses)
        ate = compute_ate(gt_poses[:len(est_poses)], est_poses)
        print(f"  {cfg['label']} -> ATE: {ate:.4f} m")

    # Plot comparisons
    plt.figure(figsize=(10, 10))
    plt.plot([p[0,3] for p in gt_poses[:50]], [p[2,3] for p in gt_poses[:50]], label='GT', linewidth=2)
    for i, cfg in enumerate(configs):
        plt.plot([p[0,3] for p in all_est_poses[i]], [p[2,3] for p in all_est_poses[i]], label=cfg['label'], linestyle='--')
    plt.legend()
    plt.title('VO Ablation Study - Trajectory Comparison')
    plt.savefig(os.path.join(output_dir, 'vo_ablation.png'))
    plt.close()

if __name__ == "__main__":
    run_stereo_ablation()
    run_vo_ablation()
