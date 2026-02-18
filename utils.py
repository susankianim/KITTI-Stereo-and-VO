import numpy as np
import cv2
import os

def load_kitti_images(data_dir, split='training', index=0):
    idx_str = f"{index:06d}_10.png"
    img_l_path = os.path.join(data_dir, split, 'image_2', idx_str)
    img_r_path = os.path.join(data_dir, split, 'image_3', idx_str)
    
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    
    if img_l is None or img_r is None:
        raise FileNotFoundError(f"Images not found for index {index}")
        
    return cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

def load_kitti_gt(data_dir, split='training', index=0):
    idx_str = f"{index:06d}_10.png"
    gt_path = os.path.join(data_dir, split, 'disp_noc_0', idx_str)
    
    # KITTI disparity is saved as uint16, divide by 256.0 to get float
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        return None
    return gt.astype(np.float32) / 256.0

def load_calibration(calib_dir, index=0):
    idx_str = f"{index:06d}.txt"
    calib_path = os.path.join(calib_dir, idx_str)
    
    if not os.path.exists(calib_path):
        # Fallback to 000000.txt if specific one doesn't exist
        calib_path = os.path.join(calib_dir, "000000.txt")
        
    data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    # Skip values that cannot be converted to float (like timestamps)
                    continue
                
    p2 = data['P_rect_02'].reshape(3, 4)
    p3 = data['P_rect_03'].reshape(3, 4)
    
    fx = p2[0, 0]
    # Baseline B = |t_x3 - t_x2| / fx
    # In KITTI, p[0,3] is fx * baseline_offset_from_ref
    # B = |P3[0,3] - P2[0,3]| / fx
    baseline = abs(p3[0, 3] - p2[0, 3]) / fx
    
    return fx, baseline

def compute_errors(gt, pred, threshold=3.0):
    """
    Compute Bad-pixel rate and MAE for disparity.
    Only valid pixels in GT (gt > 0) are considered.
    """
    mask = gt > 0
    if not np.any(mask):
        return 0, 0
    
    error = np.abs(gt[mask] - pred[mask])
    mae = np.mean(error)
    bad_pixels = np.sum(error > threshold)
    bad_pixel_rate = (bad_pixels / np.sum(mask)) * 100.0
    
    return bad_pixel_rate, mae

def load_poses(path):
    """
    Load ground truth poses from KITTI format (12 floats per line).
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            P = np.fromstring(line, sep=' ').reshape(3, 4)
            # Convert to 4x4 matrix
            T = np.eye(4)
            T[:3, :] = P
            poses.append(T)
    return poses

def load_vo_calib(path):
    """
    Load P0, P1, P2, P3 from KITTI odometry calib.txt.
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = np.fromstring(value, sep=' ').reshape(3, 4)
    return data

def compute_ate(gt_poses, pred_poses):
    """
    Compute Absolute Trajectory Error (ATE).
    """
    errors = []
    for gt, pred in zip(gt_poses, pred_poses):
        # Position error
        err = np.linalg.norm(gt[:3, 3] - pred[:3, 3])
        errors.append(err)
    return np.mean(errors)

def compute_rpe(gt_poses, pred_poses, step=1):
    """
    Compute Relative Pose Error (RPE).
    """
    errors = []
    for i in range(len(gt_poses) - step):
        # Relative motion GT
        rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i + step]
        # Relative motion Pred
        rel_pred = np.linalg.inv(pred_poses[i]) @ pred_poses[i + step]
        
        # Error transform
        err_trans = np.linalg.inv(rel_gt) @ rel_pred
        
        # Translation error
        t_err = np.linalg.norm(err_trans[:3, 3])
        errors.append(t_err)
    return np.mean(errors)
