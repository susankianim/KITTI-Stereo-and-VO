import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from visual_odometry import VisualOdometry
from utils import load_vo_calib, load_poses, compute_ate, compute_rpe

def main():
    # Paths
    sequence_dir = '04'
    poses_path = 'poses/04.txt'
    output_dir = 'output/vo'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Calibration and GT Poses
    calib = load_vo_calib(os.path.join(sequence_dir, 'calib.txt'))
    p0 = calib['P0']
    p1 = calib['P1']
    
    gt_poses = load_poses(poses_path)
    
    # 2. Initialize Visual Odometry
    vo = VisualOdometry(p0, p1)
    
    # Get image files
    img_l_dir = os.path.join(sequence_dir, 'image_0')
    img_r_dir = os.path.join(sequence_dir, 'image_1')
    img_files = sorted(os.listdir(img_l_dir))
    
    num_frames = len(img_files)
    num_to_process = min(1000, num_frames)
    
    estimated_poses = []
    print(f"Processing {num_to_process} frames...")
    
    for i in tqdm(range(num_to_process)):
        # Load grayscale images
        img_l = cv2.imread(os.path.join(img_l_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(os.path.join(img_r_dir, img_files[i]), cv2.IMREAD_GRAYSCALE)
        
        if img_l is None or img_r is None:
            break
            
        current_pose, _ = vo.process_frame(img_l, img_r)
        estimated_poses.append(current_pose)

    # 3. Evaluation
    gt_subset = gt_poses[:num_to_process]
    ate = compute_ate(gt_subset, estimated_poses)
    rpe = compute_rpe(gt_subset, estimated_poses, step=1)
    
    print(f"\nVO Results over {num_to_process} frames:")
    print(f"  ATE: {ate:.4f} m")
    print(f"  RPE (step 1): {rpe:.4f} m/frame")
    
    # 4. Trajectory Plotting
    plt.figure(figsize=(10, 10))
    gt_x = [p[0, 3] for p in gt_subset]
    gt_z = [p[2, 3] for p in gt_subset]
    est_x = [p[0, 3] for p in estimated_poses]
    est_z = [p[2, 3] for p in estimated_poses]
    
    plt.plot(gt_x, gt_z, label='Ground Truth', color='blue')
    plt.plot(est_x, est_z, label='Estimated', color='red', linestyle='dashed')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('KITTI Odometry Sequence 04 - Full Trajectory (Ground Plane)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'trajectory_04.png'))
    plt.show()

if __name__ == "__main__":
    main()
