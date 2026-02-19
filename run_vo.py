import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from visual_odometry import VisualOdometry
from utils import load_vo_calib, load_poses, compute_ate, compute_rpe

def main():
    # --- Configuration ---
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']  # Example sequences to process
    vo_data_root = 'vo_data'
    output_dir = 'vo_results'
    csv_path = os.path.join(output_dir, 'vo_results.csv')
    
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {'ransac': True, 'stereo': True, 'label': 'Full'},
        {'ransac': False, 'stereo': True, 'label': 'No_RANSAC'},
        {'ransac': True, 'stereo': False, 'label': 'No_Stereo_Scale'}
    ]

    # Initialize CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sequence', 'Case', 'ATE (m)', 'RPE (m/frame)'])

    for seq in sequences:
        print(f"\nProcessing Sequence {seq}...")
        seq_dir = os.path.join(vo_data_root, seq)
        poses_path = os.path.join(vo_data_root, 'poses', f'{seq}.txt')
        
        if not os.path.exists(poses_path):
            print(f"Skipping {seq}: Poses not found at {poses_path}")
            continue

        calib = load_vo_calib(os.path.join(seq_dir, 'calib.txt'))
        gt_poses = load_poses(poses_path)
        
        img_l_dir = os.path.join(seq_dir, 'image_0')
        img_r_dir = os.path.join(seq_dir, 'image_1')
        img_files = sorted(os.listdir(img_l_dir))
        
        # Limit frames for faster execution in this master script if needed, 
        # but the user asked for the "project" to have this structure.
        # I'll process up to 500 frames per sequence to be reasonable.
        num_frames = min(500, len(img_files))
        
        for cfg in configs:
            print(f"  Running Case: {cfg['label']}")
            vo = VisualOdometry(calib['P0'], calib['P1'])
            estimated_poses = []
            
            inlier_images_saved = 0
            
            for i in tqdm(range(num_frames), desc=f"Seq {seq} {cfg['label']}"):
                img_l = cv2.imread(os.path.join(img_l_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
                img_r = cv2.imread(os.path.join(img_r_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
                
                if img_l is None or img_r is None:
                    break
                    
                pose, debug = vo.process_frame(img_l, img_r, use_ransac=cfg['ransac'], use_stereo_scale=cfg['stereo'])
                estimated_poses.append(pose)
                
                # Save first 3 inlier matches for the "Full" case of each sequence
                if cfg['label'] == 'Full' and inlier_images_saved < 3 and debug is not None:
                    save_inlier_matches(debug, seq, i, output_dir)
                    inlier_images_saved += 1

            # Get GT subset for metrics
            gt_subset = gt_poses[:len(estimated_poses)]
            ate = compute_ate(gt_subset, estimated_poses)
            rpe = compute_rpe(gt_subset, estimated_poses, step=1)
            
            # Save to CSV
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([seq, cfg['label'], f"{ate:.4f}", f"{rpe:.4f}"])
            
            # Plot Trajectory
            plot_trajectory(gt_subset, estimated_poses, seq, cfg['label'], output_dir)
            
    print(f"\nVO evaluation complete. Results saved to {csv_path} and plots in {output_dir}/")
    
    # Print summary table
    print("\n--- VO Summary Table ---")
    print(f"{'Seq':<5} | {'Case':<20} | {'ATE (m)':<10} | {'RPE (m/f)':<10}")
    print("-" * 55)
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            print(f"{row[0]:<5} | {row[1]:<20} | {row[2]:<10} | {row[3]:<10}")

def save_inlier_matches(debug, seq, frame_idx, output_dir):
    matches = debug['matches']
    kp1 = debug['prev_kp']
    kp2 = debug['curr_kp']
    inliers = debug['inliers']
    img1 = debug['prev_img']
    img2 = debug['curr_img']
    
    if inliers is not None:
        # Filter matches by inliers
        inlier_matches = [matches[i] for i in inliers]
    else:
        inlier_matches = matches[:50] # Just some matches if no inliers/ransac
        
    out_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # type: ignore
    
    match_dir = os.path.join(output_dir, 'matches')
    os.makedirs(match_dir, exist_ok=True)
    cv2.imwrite(os.path.join(match_dir, f"matches_seq{seq}_frame{frame_idx:03d}.png"), out_img)

def plot_trajectory(gt_poses, est_poses, seq, label, output_dir):
    plt.figure(figsize=(10, 10))
    gt_x = [p[0, 3] for p in gt_poses]
    gt_z = [p[2, 3] for p in gt_poses]
    est_x = [p[0, 3] for p in est_poses]
    est_z = [p[2, 3] for p in est_poses]
    
    plt.plot(gt_x, gt_z, label='Ground Truth', color='blue')
    plt.plot(est_x, est_z, label='Estimated', color='red', linestyle='dashed')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title(f'Trajectory - Seq {seq} ({label})')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    traj_dir = os.path.join(output_dir, 'trajectories')
    os.makedirs(traj_dir, exist_ok=True)
    plt.savefig(os.path.join(traj_dir, f"trajectory_{seq}_{label}.png"))
    plt.close()

if __name__ == "__main__":
    main()
