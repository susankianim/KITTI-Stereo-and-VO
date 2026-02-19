import cv2
import numpy as np

class VisualOdometryFeature:
    def __init__(self, P0, P1, max_features=1500):
        """
        Initialize the Visual Odometry with projection matrices.
        P0: Projection matrix for left camera (K [I|0])
        P1: Projection matrix for right camera (K [I|t])
        """
        self.K = P0[:, :3]
        # Calculate baseline from horizontal offset in P1
        # P1[0, 3] = -f * baseline
        self.baseline = abs(P1[0, 3]) / self.K[0, 0]
        
        self.orb = cv2.ORB_create(nfeatures=max_features) # type: ignore
        
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
        if des_r is None or len(des_r) == 0:
            return np.zeros(len(kps_l))
            
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

    def process_frame(self, img_l, img_r, use_ransac=True, use_stereo_scale=True):
        """Processes a single stereo pair at time t."""
        kps_l, des_l = self.orb.detectAndCompute(img_l, None)
        
        if des_l is None or len(des_l) == 0:
            return self.T_curr, None

        if self.prev_img_l is None:
            disparity = self._get_disparity(img_l, img_r, kps_l, des_l)
            pts_2d = np.array([kp.pt for kp in kps_l])
            valid = disparity > 0
            
            if np.sum(valid) == 0:
                # Fallback if no valid disparities found
                self.prev_img_l = img_l
                self.prev_kps_l = kps_l
                self.prev_des_l = des_l
                return self.T_curr, None

            self.prev_img_l = img_l
            
            f, cx, cy = self.K[0,0], self.K[0,2], self.K[1,2]
            z = (f * self.baseline) / (disparity[valid] + 1e-6)
            x = (pts_2d[valid, 0] - cx) * z / f
            y = (pts_2d[valid, 1] - cy) * z / f
            self.prev_3d = np.stack((x, y, z), axis=-1)
            
            self.prev_kps_l = [kps_l[i] for i, v in enumerate(valid) if v]
            self.prev_des_l = des_l[valid]
            
            return self.T_curr, None

        # Temporal matching
        matches = self.bf.match(self.prev_des_l, des_l) # type: ignore
        if len(matches) < 10: # Min matches
            return self.T_curr, None

        idx_prev = [m.queryIdx for m in matches]
        idx_curr = [m.trainIdx for m in matches]
        pts_3d_prev = self.prev_3d[idx_prev] # type: ignore
        pts_2d_curr = np.array([kps_l[i].pt for i in idx_curr])
        pts_2d_prev = np.array([self.prev_kps_l[i].pt for i in idx_prev]) # type: ignore
        
        inliers = None
        if use_ransac:
            # 1. Estimate Essential Matrix with RANSAC to satisfy project requirement
            E, mask = cv2.findEssentialMat(pts_2d_prev, pts_2d_curr, self.K, 
                                          method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # 2. Use the inliers from the Essential matrix to refine pose with PnP
            if mask is not None:
                inliers_idx = np.where(mask.ravel() == 1)[0]
                if len(inliers_idx) < 4:
                     pts_3d_filt, pts_2d_filt, inliers_idx = pts_3d_prev, pts_2d_curr, np.arange(len(pts_3d_prev))
                else:
                    pts_3d_filt = pts_3d_prev[inliers_idx]
                    pts_2d_filt = pts_2d_curr[inliers_idx]
            else:
                pts_3d_filt, pts_2d_filt, inliers_idx = pts_3d_prev, pts_2d_curr, np.arange(len(pts_3d_prev))

            retval, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                pts_3d_filt, pts_2d_filt, self.K, distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999, reprojectionError=1.0)
            
            if retval and inliers_pnp is not None:
                inliers = inliers_idx[inliers_pnp.flatten()] # type: ignore
        else:
            retval, rvec, tvec = cv2.solvePnP(pts_3d_prev, pts_2d_curr, self.K, distCoeffs=None)
            inliers = np.arange(len(pts_3d_prev))

        if retval:
            if not use_stereo_scale:
                # Monocular case: normalize translation scale to 1 (arbitrary)
                # Note: In a real monocular system, you'd want to estimate scale from previous motion or ground plane
                scale = np.linalg.norm(tvec)
                if scale > 0: tvec /= scale
                
            R, _ = cv2.Rodrigues(rvec)
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.ravel()
            
            # Use the inverse of relative transformation to update current camera pose
            # T_curr = T_curr * T_rel^-1
            self.T_curr = self.T_curr @ np.linalg.inv(T_rel)
            
            # Construct debug dictionary as expected by run_vo.py
            debug_info = {
                'prev_img': self.prev_img_l,
                'curr_img': img_l,
                'prev_kp': self.prev_kps_l,
                'curr_kp': [kps_l[i] for i in idx_curr], # Just the matched ones for drawMatches? 
                                                          # Wait, drawMatches expects indices from the full list if kps are full.
                'matches': [cv2.DMatch(_i, _i, 0) for _i in range(len(matches))], # Remapped matches
                'inliers': inliers
            }
            # Actually, run_vo.py does:
            # kp1 = debug['prev_kp']
            # kp2 = debug['curr_kp']
            # matches = debug['matches']
            # inlier_matches = [matches[i] for i in inliers]
            # cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:50], ... )
            
            # Let's adjust debug_info to be exactly what drawMatches needs
            debug_info = {
                'prev_img': self.prev_img_l,
                'curr_img': img_l,
                'prev_kp': self.prev_kps_l,
                'curr_kp': kps_l,
                'matches': matches,
                'inliers': inliers
            }
            
            # Re-triangulate for next step
            disparity = self._get_disparity(img_l, img_r, kps_l, des_l)
            valid = disparity > 0
            
            if np.sum(valid) > 0:
                f, cx, cy = self.K[0,0], self.K[0,2], self.K[1,2]
                z = (f * self.baseline) / (disparity[valid] + 1e-6)
                
                pts_2d_valid = np.array([kp.pt for i, kp in enumerate(kps_l) if valid[i]])
                x = (pts_2d_valid[:, 0] - cx) * z / f
                y = (pts_2d_valid[:, 1] - cy) * z / f
                
                self.prev_3d = np.stack((x, y, z), axis=-1)
                self.prev_kps_l = [kps_l[i] for i, v in enumerate(valid) if v]
                self.prev_des_l = des_l[valid]
                self.prev_img_l = img_l
            else:
                # If no valid triangulation, we might be in trouble for the next frame
                # but we'll let it try.
                self.prev_img_l = img_l
                self.prev_kps_l = kps_l
                self.prev_des_l = des_l
                self.prev_3d = None # This will cause an error next frame if not handled
            
            return self.T_curr, debug_info
        
        return self.T_curr, None
