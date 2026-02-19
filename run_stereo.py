import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from stereo_matcher import StereoMatcher
from utils import load_kitti_images, load_kitti_gt, load_calibration, compute_errors

def main():
    # --- Configuration ---
    indices = range(201)  
    methods = ['SAD', 'SSD', 'NCC']
    window_sizes = [7, 15]
    
    data_dir = os.path.join('stereo_data', 'data_scene_flow')
    calib_dir = os.path.join('stereo_data', 'calib_cam_to_cam')
    output_dir = 'stereo_results'
    full_csv_path = os.path.join(output_dir, 'all_results.csv')
    avg_csv_path = os.path.join(output_dir, 'average_results.csv')
    worst_csv_path = os.path.join(output_dir, 'worst_samples.csv')
    
    os.makedirs(output_dir, exist_ok=True)

    # Collect results
    all_results = []

    print(f"Starting Stereo Evaluation on indices: {indices}")
    
    for idx in indices:
        try:
            img_l, img_r, img_color = load_kitti_images(data_dir, index=idx)
            gt = load_kitti_gt(data_dir, index=idx)
            fx, baseline = load_calibration(calib_dir, index=idx)
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            continue

        for method in methods:
            for ws in window_sizes:
                print(f"Processing Index {idx}, Method {method}, WS {ws}...")
                
                matcher = StereoMatcher(window_size=ws, max_disp=128)
                
                # Compute disparity
                disp_l = matcher.compute_disparity(img_l, img_r, method=method)
                disp_r = matcher.compute_disparity(img_l, img_r, method=method, right_reference=True)
                
                # Refinement
                consistent_disp = matcher.left_right_consistency_check(disp_l, disp_r)
                final_disp = matcher.post_process(consistent_disp)
                
                # Compute depth
                depth = np.zeros_like(final_disp)
                mask = final_disp > 0
                depth[mask] = (fx * baseline) / final_disp[mask]
                
                # Metrics
                bpr, mae = 0.0, 0.0
                if gt is not None:
                    bpr, mae = compute_errors(gt, final_disp)
                
                # Store result
                all_results.append({
                    'idx': idx, 
                    'method': method, 
                    'ws': ws, 
                    'bpr': bpr, 
                    'mae': mae
                })
                
                # Visualization
                if idx % 20 == 0:
                    save_visualization(img_color, final_disp, gt, depth, idx, method, ws, output_dir)
    
    # --- Post-processing: Compute Averages and Find Worst Samples ---
    
    # All Results CSV
    with open(full_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'method', 'ws', 'bpr', 'mae'])
        writer.writeheader()
        writer.writerows(all_results)

    # Group by (method, ws)
    settings = {}
    for res in all_results:
        key = (res['method'], res['ws'])
        if key not in settings:
            settings[key] = []
        settings[key].append(res)

    avg_results = []
    
    # 1. Compute averages for all settings
    for (method, ws), res_list in settings.items():
        bprs = [r['bpr'] for r in res_list]
        maes = [r['mae'] for r in res_list]
        avg_results.append({
            'method': method,
            'ws': ws,
            'avg_bpr': np.mean(bprs),
            'avg_mae': np.mean(maes)
        })

    # 2. Find top 10 worst samples specifically for NCC with window size 15
    worst_samples = []
    ncc_15_res = settings.get(('NCC', 15), [])
    if ncc_15_res:
        # Sort by BPR descending and take top 10
        ncc_15_sorted = sorted(ncc_15_res, key=lambda x: x['bpr'], reverse=True)
        top_10_worst = ncc_15_sorted[:10]
        for r in top_10_worst:
            worst_samples.append({
                'method': 'NCC',
                'ws': 15,
                'idx': r['idx'],
                'bpr': r['bpr'],
                'mae': r['mae']
            })

    # Save Average Results
    with open(avg_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'ws', 'avg_bpr', 'avg_mae'])
        writer.writeheader()
        writer.writerows(avg_results)

    # Save Worst Samples (Top 10 NCC 15)
    with open(worst_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'ws', 'idx', 'bpr', 'mae'])
        writer.writeheader()
        writer.writerows(worst_samples)

    print(f"\nEvaluation complete.")
    print(f"All Results: {full_csv_path}")
    print(f"Averages: {avg_csv_path}")
    print(f"Worst NCC 15 Samples: {worst_csv_path}")
    
    # Print Average Summary Table
    print("\n--- Average Results Table ---")
    print(f"{'Method':<10} | {'WS':<5} | {'Avg BPR (%)':<12} | {'Avg MAE (px)':<12}")
    print("-" * 55)
    for r in avg_results:
        print(f"{r['method']:<10} | {r['ws']:<5} | {r['avg_bpr']:<12.2f} | {r['avg_mae']:<12.2f}")

    print("\n--- Top 10 Worst NCC 15 Samples ---")
    print(f"{'Idx':<10} | {'BPR (%)':<10} | {'MAE (px)':<10}")
    print("-" * 35)
    for r in worst_samples:
        print(f"{r['idx']:<10} | {r['bpr']:<10.2f} | {r['mae']:<10.2f}")

def save_visualization(img_color, disp, gt, depth, idx, method, ws, output_dir):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_color)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title(f"Disparity Map ({method}, WS={ws})")
    plt.imshow(disp, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    if gt is not None:
        plt.title("Ground Truth Disparity")
        plt.imshow(gt, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        plt.title("Ground Truth Not Available")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("Depth Map")
    plt.imshow(np.clip(depth, 0, 80), cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    filename = f"stereo_idx{idx:03d}_{method}_ws{ws}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

if __name__ == "__main__":
    main()
