import cv2
import numpy as np

class StereoVO:
    def __init__(self, K, baseline, max_features=1500):
        self.K = K
        self.baseline = baseline
        self.orb = cv2.ORB_create(nfeatures=max_features)
        
        # Matcher for frame-to-frame (temporal) and left-to-right (stereo)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Current camera pose (relative to start)
        self.T_curr = np.eye(4)
        
        # Frame storage
        self.prev_img_l = None
        self.prev_kps_l = None
        self.prev_des_l = None
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

    def process_frame_pair(self, img_l, img_r, use_ransac=True, use_stereo_scale=True):
        """Processes a single stereo pair at time t."""
        kps_l, des_l = self.orb.detectAndCompute(img_l, None)
        
        if self.prev_img_l is None:
            disparity = self._get_disparity(img_l, img_r, kps_l, des_l)
            pts_2d = np.array([kp.pt for kp in kps_l])
            valid = disparity > 0
            
            self.prev_img_l = img_l
            self.prev_kps_l = kps_l
            self.prev_des_l = des_l
            
            f, cx, cy = self.K[0,0], self.K[0,2], self.K[1,2]
            z = (f * self.baseline) / (disparity[valid] + 1e-6)
            x = (pts_2d[valid, 0] - cx) * z / f
            y = (pts_2d[valid, 1] - cy) * z / f
            self.prev_3d = np.stack((x, y, z), axis=-1)
            
            self.prev_kps_l = [kps_l[i] for i, v in enumerate(valid) if v]
            self.prev_des_l = des_l[valid]
            
            return self.T_curr, None

        # Temporal matching
        matches = self.bf.match(self.prev_des_l, des_l)
        idx_prev = [m.queryIdx for m in matches]
        idx_curr = [m.trainIdx for m in matches]
        pts_3d_prev = self.prev_3d[idx_prev]
        pts_2d_curr = np.array([kps_l[i].pt for i in idx_curr])
        pts_2d_prev = np.array([self.prev_kps_l[i].pt for i in idx_prev])
        
        if use_ransac:
            # 1. Estimate Essential Matrix with RANSAC to satisfy project requirement
            # This finds the geometric relationship between 2D-2D correspondences
            E, mask = cv2.findEssentialMat(pts_2d_prev, pts_2d_curr, self.K, 
                                          method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # 2. Use the inliers from the Essential matrix to refine pose with PnP
            # SolvePnP is used to recover the absolute metric scale from 3D-2D matches
            if mask is not None:
                inliers_idx = np.where(mask.ravel() == 1)[0]
                pts_3d_filt = pts_3d_prev[inliers_idx]
                pts_2d_filt = pts_2d_curr[inliers_idx]
            else:
                pts_3d_filt, pts_2d_filt, inliers_idx = pts_3d_prev, pts_2d_curr, np.arange(len(pts_3d_prev))

            retval, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                pts_3d_filt, pts_2d_filt, self.K, distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999, reprojectionError=1.0)
            
            if retval and inliers_pnp is not None:
                inliers = inliers_idx[inliers_pnp.flatten()]
        else:
            retval, rvec, tvec = cv2.solvePnP(pts_3d_prev, pts_2d_curr, self.K, distCoeffs=None)
            inliers = np.arange(len(pts_3d_prev))

        if retval:
            if not use_stereo_scale:
                # Monocular case: normalize translation scale to 1 (arbitrary)
                scale = np.linalg.norm(tvec)
                if scale > 0: tvec /= scale
                
            R, _ = cv2.Rodrigues(rvec)
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.ravel()
            
            if use_ransac:
                matches_mask = np.zeros(len(matches), dtype=np.int32)
                if inliers is not None:
                    matches_mask[inliers.flatten()] = 1
            else:
                matches_mask = np.ones(len(matches), dtype=np.int32)

            debug_info = (self.prev_img_l, img_l, self.prev_kps_l, kps_l, matches, matches_mask)
            self.T_curr = self.T_curr @ np.linalg.inv(T_rel)
            
            # Re-triangulate for next step
            disparity = self._get_disparity(img_l, img_r, kps_l, des_l)
            valid = disparity > 0
            f, cx, cy = self.K[0,0], self.K[0,2], self.K[1,2]
            z = (f * self.baseline) / (disparity[valid] + 1e-6)
            
            pts_2d_valid = np.array([kp.pt for i, kp in enumerate(kps_l) if valid[i]])
            x = (pts_2d_valid[:, 0] - cx) * z / f
            y = (pts_2d_valid[:, 1] - cy) * z / f
            
            self.prev_3d = np.stack((x, y, z), axis=-1)
            self.prev_kps_l = [kps_l[i] for i, v in enumerate(valid) if v]
            self.prev_des_l = des_l[valid]
            self.prev_img_l = img_l
            
            return self.T_curr, debug_info
        
        return self.T_curr, None
