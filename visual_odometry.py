import numpy as np
import cv2
from stereo_matcher import StereoMatcher

class VisualOdometry:
    def __init__(self, P0, P1, window_size=7, max_disp=128):
        self.P0 = P0
        self.P1 = P1
        self.fx = P0[0, 0]
        self.cx = P0[0, 2]
        self.cy = P0[1, 2]
        self.baseline = abs(P1[0, 3] - P0[0, 3]) / self.fx
        
        self.orb = cv2.ORB_create(3000) # type: ignore
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matcher = StereoMatcher(window_size=window_size, max_disp=max_disp)
        
        self.current_pose = np.eye(4)
        self.prev_img_l = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_img_r = None

    def _get_3d_points(self, img_l, img_r, kps):
        """
        Compute 3D points for given keypoints in img_l using stereo matching.
        """
        disp_l = self.matcher.compute_disparity(img_l, img_r, method='SSD')
        disp_l = self.matcher.post_process(disp_l)
        
        pts_3d = []
        valid_idx = []
        
        for i, kp in enumerate(kps):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if v >= disp_l.shape[0] or u >= disp_l.shape[1]:
                continue
                
            d = disp_l[v, u]
            if d <= 0:
                continue
                
            # Z = f * B / d
            z = (self.fx * self.baseline) / d
            # Metric scale: X = (u - cx) * Z / f, Y = (v - cy) * Z / f
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fx
            
            pts_3d.append([x, y, z])
            valid_idx.append(i)
            
        return np.array(pts_3d, dtype=np.float32), valid_idx

    def process_frame(self, img_l, img_r, use_ransac=True, use_stereo_scale=True):
        """
        Process a new stereo frame and update current pose.
        """
        kp_l, des_l = self.orb.detectAndCompute(img_l, None)
        
        if self.prev_img_l is None:
            self.prev_img_l = img_l
            self.prev_kp = kp_l
            self.prev_des = des_l
            self.prev_img_r = img_r
            return self.current_pose, None

        matches = self.bf.match(self.prev_des, des_l) # type: ignore
        matches = sorted(matches, key=lambda x: x.distance)
        
        prev_pts_objs = [self.prev_kp[m.queryIdx] for m in matches] # type: ignore
        curr_pts = np.array([kp_l[m.trainIdx].pt for m in matches], dtype=np.float32)
        
        # pts_3d_prev, valid_idx = self._get_3d_points(self.prev_img_l, self.prev_img_r, prev_pts_objs)
        pts_3d_prev, valid_idx = self._get_3d_points(self.prev_img_l, self.prev_img_r, prev_pts_objs)
        curr_pts_valid = curr_pts[valid_idx]
        matches_valid = [matches[i] for i in valid_idx]
        
        if len(pts_3d_prev) < 10:
            return self.current_pose, None

        K = self.P0[:3, :3]
        dist_coeffs = np.zeros((4, 1))
        
        inlier_mask = None
        if use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d_prev, curr_pts_valid, K, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999, reprojectionError=2.0
            )
            if success and inliers is not None:
                inlier_mask = inliers.flatten()
        else:
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_prev, curr_pts_valid, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            inlier_mask = np.arange(len(pts_3d_prev))
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            if not use_stereo_scale:
                # Normalize translation if not using stereo scale (monocular unit scale)
                norm = np.linalg.norm(tvec)
                if norm > 0:
                    tvec = tvec / norm
            
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.ravel()
            
            # Update world pose
            self.current_pose = self.current_pose @ np.linalg.inv(T_rel)
        
        debug_info = {
            'matches': matches_valid,
            'prev_kp': self.prev_kp,
            'curr_kp': kp_l,
            'inliers': inlier_mask,
            'prev_img': self.prev_img_l,
            'curr_img': img_l
        }

        self.prev_img_l = img_l
        self.prev_kp = kp_l
        self.prev_des = des_l
        self.prev_img_r = img_r
        
        return self.current_pose, debug_info