import numpy as np
import cv2

class StereoMatcher:
    def __init__(self, window_size=7, max_disp=64):
        self.window_size = window_size
        self.max_disp = max_disp
        self.half_window = window_size // 2

    def compute_disparity(self, img_left, img_right, method='SAD', right_reference=False):
        """
        Compute disparity map using block matching.
        If right_reference is True, computes disparity for the right image (searching in left).
        """
        h, w = img_left.shape[:2]
        img_left = img_left.astype(np.float32)
        img_right = img_right.astype(np.float32)

        if method == 'SAD':
            return self._block_matching_sad(img_left, img_right, right_reference)
        elif method == 'SSD':
            return self._block_matching_ssd(img_left, img_right, right_reference)
        elif method == 'NCC':
            return self._block_matching_ncc(img_left, img_right, right_reference)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _block_matching_sad(self, img_l, img_r, right_ref=False):
        h, w = img_l.shape[:2]
        disparity_map = np.zeros((h, w), dtype=np.float32)
        min_costs = np.full((h, w), np.inf, dtype=np.float32)

        for d in range(self.max_disp):
            if not right_ref:
                # Left ref: match L(x, y) with R(x-d, y)
                if d == 0:
                    shifted_r = img_r
                else:
                    shifted_r = np.zeros_like(img_r)
                    shifted_r[:, d:] = img_r[:, :-d]
                diff = np.abs(img_l - shifted_r)
            else:
                # Right ref: match R(x, y) with L(x+d, y)
                if d == 0:
                    shifted_l = img_l
                else:
                    shifted_l = np.zeros_like(img_l)
                    shifted_l[:, :-d] = img_l[:, d:]
                diff = np.abs(img_r - shifted_l)
            
            cost = cv2.boxFilter(diff, -1, (self.window_size, self.window_size), normalize=False)

            mask = cost < min_costs
            min_costs[mask] = cost[mask]
            disparity_map[mask] = d

        return disparity_map

    def _block_matching_ssd(self, img_l, img_r, right_ref=False):
        h, w = img_l.shape[:2]
        disparity_map = np.zeros((h, w), dtype=np.float32)
        min_costs = np.full((h, w), np.inf, dtype=np.float32)

        for d in range(self.max_disp):
            if not right_ref:
                if d == 0:
                    shifted_r = img_r
                else:
                    shifted_r = np.zeros_like(img_r)
                    shifted_r[:, d:] = img_r[:, :-d]
                diff_sq = (img_l - shifted_r) ** 2
            else:
                if d == 0:
                    shifted_l = img_l
                else:
                    shifted_l = np.zeros_like(img_l)
                    shifted_l[:, :-d] = img_l[:, d:]
                diff_sq = (img_r - shifted_l) ** 2

            cost = cv2.boxFilter(diff_sq, -1, (self.window_size, self.window_size), normalize=False)

            mask = cost < min_costs
            min_costs[mask] = cost[mask]
            disparity_map[mask] = d

        return disparity_map

    def _block_matching_ncc(self, img_l, img_r, right_ref=False):
        h, w = img_l.shape[:2]
        disparity_map = np.zeros((h, w), dtype=np.float32)
        max_ncc = np.full((h, w), -1.0, dtype=np.float32)

        if not right_ref:
            ref_img = img_l
            target_img = img_r
        else:
            ref_img = img_r
            target_img = img_l

        mean_ref = cv2.boxFilter(ref_img, -1, (self.window_size, self.window_size))
        sq_ref = cv2.boxFilter(ref_img**2, -1, (self.window_size, self.window_size))
        var_ref = sq_ref - mean_ref**2
        std_ref = np.sqrt(np.maximum(var_ref, 1e-5))

        for d in range(self.max_disp):
            if not right_ref:
                # Match L(x,y) with R(x-d, y)
                if d == 0:
                    shifted_target = target_img
                else:
                    shifted_target = np.zeros_like(target_img)
                    shifted_target[:, d:] = target_img[:, :-d]
            else:
                # Match R(x,y) with L(x+d, y)
                if d == 0:
                    shifted_target = target_img
                else:
                    shifted_target = np.zeros_like(target_img)
                    shifted_target[:, :-d] = target_img[:, d:]

            mean_target = cv2.boxFilter(shifted_target, -1, (self.window_size, self.window_size))
            sq_target = cv2.boxFilter(shifted_target**2, -1, (self.window_size, self.window_size))
            var_target = sq_target - mean_target**2
            std_target = np.sqrt(np.maximum(var_target, 1e-5))

            cov = cv2.boxFilter(ref_img * shifted_target, -1, (self.window_size, self.window_size)) - mean_ref * mean_target
            ncc = cov / (std_ref * std_target)

            mask = ncc > max_ncc
            max_ncc[mask] = ncc[mask]
            disparity_map[mask] = d

        return disparity_map

    def left_right_consistency_check(self, disp_l, disp_r, threshold=1.0):
        h, w = disp_l.shape
        consistent_disp = disp_l.copy()
        
        y, x = np.indices((h, w))
        # For each pixel (x, y) in left image, its match in right image is (x - disp_l(x,y), y)
        x_r = x - disp_l.astype(int)
        
        # Clip x_r to valid range
        x_r = np.clip(x_r, 0, w - 1)
        
        # Get disparity from right map at the matched position
        # disp_r is computed with right as reference, so disp_r(x_r, y) should match disp_l(x,y)
        # Note: If disp_r is computed such that matches are at (x_r + disp_r, y), then:
        matched_disp_r = disp_r[y, x_r]
        
        mask = np.abs(disp_l - matched_disp_r) > threshold
        consistent_disp[mask] = -1 # Mark as invalid
        
        return consistent_disp

    def post_process(self, disparity, median_size=5):
        
        # Hole filling (simple horizontal interpolation)
        processed = self._fill_holes(disparity)
        
        # Median filtering
        processed = cv2.medianBlur(processed.astype(np.float32), median_size)
        
        return processed

    def _fill_holes(self, disp):
        """
        Fill invalid pixels (<= 0) using interpolation.
        """
        h, w = disp.shape
        filled = disp.copy()
        
        # Identify holes
        mask = (disp <= 0).astype(np.uint8)
        
        # Simple row-wise propagation: for each hole, use nearest valid pixel on the left
        # This is a common "simple" strategy in stereo
        for y in range(h):
            last_valid = 0
            # Find first valid pixel in row to initialize last_valid
            for x in range(w):
                if filled[y, x] > 0:
                    last_valid = filled[y, x]
                    break
            
            for x in range(w):
                if filled[y, x] <= 0:
                    filled[y, x] = last_valid
                else:
                    last_valid = filled[y, x]
        
        return filled

    def _fill_holes_nearest(self, disp, invalid_val=0):
        h, w = disp.shape
        filled = disp.copy()
        valid = disp > invalid_val

        for y in range(h):
            for x in range(w):
                if valid[y, x]:
                    continue

                candidates = []

                # ← left
                for xx in range(x-1, -1, -1):
                    if valid[y, xx]:
                        candidates.append((x-xx, disp[y, xx]))
                        break

                # → right
                for xx in range(x+1, w):
                    if valid[y, xx]:
                        candidates.append((xx-x, disp[y, xx]))
                        break

                # # ↑ up
                # for yy in range(y-1, -1, -1):
                #     if valid[yy, x]:
                #         candidates.append((y-yy, disp[yy, x]))
                #         break

                # # ↓ down
                # for yy in range(y+1, h):
                #     if valid[yy, x]:
                #         candidates.append((yy-y, disp[yy, x]))
                #         break

                if candidates:
                    _, val = min(candidates, key=lambda t: t[0])
                    filled[y, x] = val

        return filled