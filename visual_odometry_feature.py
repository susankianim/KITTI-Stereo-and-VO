import numpy as np
import cv2
from stereo_matcher import StereoMatcher

class VisualOdometryFeature:
    def __init__(self, P0, P1, window_size=7, max_disp=128):
        self.P0 = P0
        self.P1 = P1
        self.fx = P0[0, 0]
        self.cx = P0[0, 2]
        self.cy = P0[1, 2]
        self.baseline = abs(P1[0, 3] - P0[0, 3]) / self.fx
        
        self.orb = cv2.ORB_create(3000) # type: ignore
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.current_pose = np.eye(4)
        self.prev_img_l = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_3d = None

    def _get_disparity(self, img_l, img_r, kps_l, des_l):
        """Matches features between left and right image to get sparsity at time t."""
        kps_r, des_r = self.orb.detectAndCompute(img_r, None)
        matches = self.bf.match(des_l, des_r)
        
        # Epipolar constraint: check vertical alignment and disparity range
        valid_matches = []
        for m in matches:
            pt_l = kps_l[m.queryIdx].pt
            pt_r = kps_r[m.trainIdx].pt
            
            # KITTI is rectified, so y should be similar. Also disparity must be positive.
            if abs(pt_l[1] - pt_r[1]) < 2 and pt_l[0] > pt_r[0]:
                valid_matches.append(m)
        
        disparities = np.zeros(len(kps_l))
        for m in valid_matches:
            pt_l = kps_l[m.queryIdx].pt
            pt_r = kps_r[m.trainIdx].pt
            disparities[m.queryIdx] = pt_l[0] - pt_r[0]
            
        return disparities

    def _get_3d_points_feature_matching(self, img_l, img_r, kps):
        """
        Compute 3D points for given keypoints in img_l using feature matching in the right image.
        """
        # Describe the keypoints in the left image
        _, des_l = self.orb.compute(img_l, kps)

        if des_l is None:
            return np.array([], dtype=np.float32), []

        disparities = self._get_disparity(img_l, img_r, kps, des_l)

        pts_3d = []
        valid_idx = []
        for i, d in enumerate(disparities):
            if d > 0:
                pt_l = kps[i].pt
                z = (self.fx * self.baseline) / d
                x = (pt_l[0] - self.cx) * z / self.fx
                y = (pt_l[1] - self.cy) * z / self.fx
                pts_3d.append([x, y, z])
                valid_idx.append(i)
                    
        return np.array(pts_3d, dtype=np.float32), valid_idx

    def process_frame(self, img_l, img_r, use_ransac=True, use_stereo_scale=True):
        """
        Process a new stereo frame and update current pose.
        """
        kp_l, des_l = self.orb.detectAndCompute(img_l, None)
        
        if self.prev_img_l is None:
            # Re-triangulate for the first frame
            pts_3d, valid_idx = self._get_3d_points_feature_matching(img_l, img_r, kp_l)
            
            self.prev_img_l = img_l
            self.prev_kp = [kp_l[i] for i in valid_idx]
            self.prev_des = des_l[valid_idx]
            self.prev_3d = pts_3d
            return self.current_pose, None

        # Temporal matching (Previous Left to Current Left)
        matches = self.bf.match(self.prev_des, des_l) # type: ignore
        
        idx_prev = [m.queryIdx for m in matches]
        idx_curr = [m.trainIdx for m in matches]
        
        pts_3d_prev = self.prev_3d[idx_prev]
        pts_2d_curr = np.array([kp_l[i].pt for i in idx_curr], dtype=np.float32)
        pts_2d_prev = np.array([self.prev_kp[i].pt for i in idx_prev], dtype=np.float32)
        
        if len(pts_3d_prev) < 10:
            return self.current_pose, None

        K = self.P0[:3, :3]
        dist_coeffs = np.zeros((4, 1))
        
        success = False
        inlier_mask = None
        
        if use_ransac:
            # Step 1: Essential Matrix to filter matching outliers
            E, e_mask = cv2.findEssentialMat(pts_2d_prev, pts_2d_curr, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if e_mask is not None:
                inliers_idx = np.where(e_mask.ravel() == 1)[0]
                pts_3d_filt = pts_3d_prev[inliers_idx]
                pts_2d_filt = pts_2d_curr[inliers_idx]
            else:
                pts_3d_filt, pts_2d_filt, inliers_idx = pts_3d_prev, pts_2d_curr, np.arange(len(pts_3d_prev))

            # Step 2: solvePnPRansac for robust pose
            success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                pts_3d_filt, pts_2d_filt, K, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999, reprojectionError=1.0
            )
            if success and inliers_pnp is not None:
                inlier_mask = inliers_idx[inliers_pnp.flatten()]
        else:
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_prev, pts_2d_curr, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE # type: ignore
            )
            inlier_mask = np.arange(len(pts_3d_prev))
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            if not use_stereo_scale:
                norm = np.linalg.norm(tvec)
                if norm > 0:
                    tvec = tvec / norm
            
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.ravel()
            
            # Update world pose
            self.current_pose = self.current_pose @ np.linalg.inv(T_rel)
            
            # Re-triangulate for next frame
            pts_3d_curr, valid_idx = self._get_3d_points_feature_matching(img_l, img_r, kp_l)
            
            self.prev_3d = pts_3d_curr
            self.prev_kp = [kp_l[i] for i in valid_idx]
            self.prev_des = des_l[valid_idx]
            self.prev_img_l = img_l
        
        debug_info = {
            'matches': matches,
            'prev_kp': self.prev_kp,
            'curr_kp': kp_l,
            'inliers': inlier_mask,
            'prev_img': self.prev_img_l,
            'curr_img': img_l
        }
        
        return self.current_pose, debug_info
