import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.append('/host/d/GitHub_folder/CT_registration_diffusion')
from train_fullCT import UNet3D, SpatialTransformer
from create_fullCT_dataloader import Dataset_FullCT

def load_markers(case_path):
    """Load marker positions from txt files"""
    t00_markers = []
    t50_markers = []
    
    # Read T00 markers
    t00_file = os.path.join(case_path, 'Case_T00_xyz.txt')
    if os.path.exists(t00_file):
        with open(t00_file, 'r') as f:
            for line in f:
                coords = [float(x) for x in line.strip().split()]
                if len(coords) == 3:
                    t00_markers.append(coords)
    
    # Read T50 markers
    t50_file = os.path.join(case_path, 'Case_T50_xyz.txt')
    if os.path.exists(t50_file):
        with open(t50_file, 'r') as f:
            for line in f:
                coords = [float(x) for x in line.strip().split()]
                if len(coords) == 3:
                    t50_markers.append(coords)
    
    return np.array(t00_markers), np.array(t50_markers)

def markers_to_sparse_flow(t00_markers, t50_markers, image_shape=(128, 128, 128)):
    """
    Convert marker pairs to sparse flow field
    t00_markers: [N, 3] - fixed positions (x, y, z)
    t50_markers: [N, 3] - moving positions (x, y, z)
    Returns: sparse flow field [3, D, H, W]
    """
    # Initialize flow field
    flow_marker = np.zeros((3, *image_shape))
    confidence_map = np.zeros(image_shape)
    
    # Calculate displacement for each marker
    displacements = t00_markers - t50_markers  # [N, 3]
    
    # Place displacements at marker locations
    for i, (t50_pos, disp) in enumerate(zip(t50_markers, displacements)):
        # Convert to voxel coordinates (assume normalized [0, 127])
        x, y, z = t50_pos
        
        # Round to nearest voxel
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        
        # Check bounds
        if 0 <= ix < image_shape[2] and 0 <= iy < image_shape[1] and 0 <= iz < image_shape[0]:
            flow_marker[0, iz, iy, ix] = disp[0]  # x displacement
            flow_marker[1, iz, iy, ix] = disp[1]  # y displacement
            flow_marker[2, iz, iy, ix] = disp[2]  # z displacement
            confidence_map[iz, iy, ix] = 1.0
    
    return flow_marker, confidence_map, displacements

def interpolate_flow_at_markers(flow_image, marker_positions):
    """
    Sample flow field at marker positions using trilinear interpolation
    flow_image: [3, D, H, W] - predicted flow from registration
    marker_positions: [N, 3] - marker positions (x, y, z)
    Returns: [N, 3] - interpolated flow vectors
    """
    N = marker_positions.shape[0]
    D, H, W = flow_image.shape[1:]
    
    sampled_flow = np.zeros((N, 3))
    
    for i, pos in enumerate(marker_positions):
        x, y, z = pos
        
        # Clamp to valid range
        x = np.clip(x, 0, W - 1.001)
        y = np.clip(y, 0, H - 1.001)
        z = np.clip(z, 0, D - 1.001)
        
        # Get integer and fractional parts
        x0, y0, z0 = int(x), int(y), int(z)
        x1, y1, z1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1), min(z0 + 1, D - 1)
        
        xd = x - x0
        yd = y - y0
        zd = z - z0
        
        # Trilinear interpolation for each component
        for c in range(3):
            c000 = flow_image[c, z0, y0, x0]
            c001 = flow_image[c, z0, y0, x1]
            c010 = flow_image[c, z0, y1, x0]
            c011 = flow_image[c, z0, y1, x1]
            c100 = flow_image[c, z1, y0, x0]
            c101 = flow_image[c, z1, y0, x1]
            c110 = flow_image[c, z1, y1, x0]
            c111 = flow_image[c, z1, y1, x1]
            
            c00 = c000 * (1 - xd) + c001 * xd
            c01 = c010 * (1 - xd) + c011 * xd
            c10 = c100 * (1 - xd) + c101 * xd
            c11 = c110 * (1 - xd) + c111 * xd
            
            c0 = c00 * (1 - yd) + c01 * yd
            c1 = c10 * (1 - yd) + c11 * yd
            
            sampled_flow[i, c] = c0 * (1 - zd) + c1 * zd
    
    return sampled_flow

def compare_mvf():
    """Main comparison function"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load registration model
    model = UNet3D(in_channels=2, out_channels=3).to(device)
    spatial_transform = SpatialTransformer(size=(128, 128, 128)).to(device)
    
    checkpoint_path = "/host/d/GitHub_folder/CT_registration_diffusion/checkpoints_fullCT/checkpoint_epoch100.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"{'='*80}")
    print(f"MVF COMPARISON: Image Registration vs Ground Truth Markers")
    print(f"{'='*80}\n")
    
    # Load test data
    marker_root = '/host/d/data/4DCT/DIR_LAB_original_0203/markers_final'
    cases = sorted([os.path.join(marker_root, d) for d in os.listdir(marker_root) if d.startswith('Case')])
    test_cases = cases[8:]  # Case9, Case10
    
    test_dataset = Dataset_FullCT(test_cases, target_shape=(128, 128, 128))
    
    all_results = []
    
    for idx, case_path in enumerate(test_cases):
        case_name = f"Case{idx+9}"
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {case_name}")
        print(f"{'='*80}")
        
        # Load markers
        t00_markers, t50_markers = load_markers(case_path)
        
        if len(t00_markers) == 0 or len(t50_markers) == 0:
            print(f"⚠️  No markers found for {case_name}")
            continue
        
        print(f"📍 Found {len(t00_markers)} marker pairs")
        
        # Get ground truth displacement from markers
        gt_displacement = t00_markers - t50_markers  # [N, 3]
        gt_magnitude = np.linalg.norm(gt_displacement, axis=1)  # [N]
        
        print(f"\n📊 Ground Truth (from Markers):")
        print(f"  Mean displacement: {gt_magnitude.mean():.3f} ± {gt_magnitude.std():.3f} voxels")
        print(f"  Max displacement:  {gt_magnitude.max():.3f} voxels")
        print(f"  Min displacement:  {gt_magnitude.min():.3f} voxels")
        
        # Get predicted flow from registration model
        moving, fixed = test_dataset[idx]
        moving_t = moving.unsqueeze(0).to(device)
        fixed_t = fixed.unsqueeze(0).to(device)
        
        with torch.no_grad():
            flow_image = model(moving_t, fixed_t)  # [1, 3, D, H, W]
        
        flow_image_np = flow_image[0].cpu().numpy()  # [3, D, H, W]
        
        # Sample predicted flow at marker positions
        pred_displacement = interpolate_flow_at_markers(flow_image_np, t50_markers)  # [N, 3]
        pred_magnitude = np.linalg.norm(pred_displacement, axis=1)
        
        print(f"\n📊 Predicted (from Registration Model):")
        print(f"  Mean displacement: {pred_magnitude.mean():.3f} ± {pred_magnitude.std():.3f} voxels")
        print(f"  Max displacement:  {pred_magnitude.max():.3f} voxels")
        print(f"  Min displacement:  {pred_magnitude.min():.3f} voxels")
        
        # Calculate errors
        displacement_error = np.linalg.norm(gt_displacement - pred_displacement, axis=1)  # [N]
        
        mean_error = displacement_error.mean()
        std_error = displacement_error.std()
        max_error = displacement_error.max()
        median_error = np.median(displacement_error)
        
        # Calculate TRE (Target Registration Error) - same as displacement error
        tre = displacement_error
        
        print(f"\n🎯 Registration Accuracy:")
        print(f"  Mean TRE:   {mean_error:.3f} ± {std_error:.3f} voxels")
        print(f"  Median TRE: {median_error:.3f} voxels")
        print(f"  Max TRE:    {max_error:.3f} voxels")
        print(f"  TRE < 1.0:  {(tre < 1.0).sum()}/{len(tre)} markers ({(tre < 1.0).mean()*100:.1f}%)")
        print(f"  TRE < 2.0:  {(tre < 2.0).sum()}/{len(tre)} markers ({(tre < 2.0).mean()*100:.1f}%)")
        print(f"  TRE < 3.0:  {(tre < 3.0).sum()}/{len(tre)} markers ({(tre < 3.0).mean()*100:.1f}%)")
        
        # Calculate correlation
        correlation = np.corrcoef(gt_magnitude, pred_magnitude)[0, 1]
        print(f"\n📈 Magnitude Correlation: {correlation:.4f}")
        
        # Store results
        results = {
            'case': case_name,
            'n_markers': len(t00_markers),
            'gt_mean': gt_magnitude.mean(),
            'gt_std': gt_magnitude.std(),
            'pred_mean': pred_magnitude.mean(),
            'pred_std': pred_magnitude.std(),
            'mean_tre': mean_error,
            'std_tre': std_error,
            'median_tre': median_error,
            'max_tre': max_error,
            'tre_below_1': (tre < 1.0).mean() * 100,
            'tre_below_2': (tre < 2.0).mean() * 100,
            'tre_below_3': (tre < 3.0).mean() * 100,
            'correlation': correlation,
            'gt_displacement': gt_displacement,
            'pred_displacement': pred_displacement,
            'tre': tre,
            't50_markers': t50_markers
        }
        all_results.append(results)
        
        # Visualize
        visualize_mvf_comparison(results, flow_image_np, idx)
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    
    overall_tre = np.concatenate([r['tre'] for r in all_results])
    overall_correlation = np.mean([r['correlation'] for r in all_results])
    
    print(f"\n🎯 Overall Registration Accuracy:")
    print(f"  Mean TRE:     {overall_tre.mean():.3f} ± {overall_tre.std():.3f} voxels")
    print(f"  Median TRE:   {np.median(overall_tre):.3f} voxels")
    print(f"  Max TRE:      {overall_tre.max():.3f} voxels")
    print(f"  TRE < 1.0mm:  {(overall_tre < 1.0).mean()*100:.1f}%")
    print(f"  TRE < 2.0mm:  {(overall_tre < 2.0).mean()*100:.1f}%")
    print(f"  Avg Corr:     {overall_correlation:.4f}")
    
    # Save summary
    save_comparison_summary(all_results)
    
    return all_results

def visualize_mvf_comparison(results, flow_image, case_idx):
    """Visualize comparison between ground truth and predicted MVF"""
    
    output_dir = "/host/d/GitHub_folder/CT_registration_diffusion/mvf_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    gt_disp = results['gt_displacement']
    pred_disp = results['pred_displacement']
    tre = results['tre']
    
    # 1. Scatter: GT vs Predicted (X component)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(gt_disp[:, 0], pred_disp[:, 0], alpha=0.6, s=50, c=tre, cmap='hot')
    lim = max(abs(gt_disp[:, 0]).max(), abs(pred_disp[:, 0]).max()) * 1.1
    ax1.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='Perfect match')
    ax1.set_xlabel('Ground Truth (X) [voxels]', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Predicted (X) [voxels]', fontsize=10, fontweight='bold')
    ax1.set_title('X Component', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Scatter: GT vs Predicted (Y component)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(gt_disp[:, 1], pred_disp[:, 1], alpha=0.6, s=50, c=tre, cmap='hot')
    lim = max(abs(gt_disp[:, 1]).max(), abs(pred_disp[:, 1]).max()) * 1.1
    ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='Perfect match')
    ax2.set_xlabel('Ground Truth (Y) [voxels]', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Predicted (Y) [voxels]', fontsize=10, fontweight='bold')
    ax2.set_title('Y Component', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Scatter: GT vs Predicted (Z component)
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(gt_disp[:, 2], pred_disp[:, 2], alpha=0.6, s=50, c=tre, cmap='hot')
    lim = max(abs(gt_disp[:, 2]).max(), abs(pred_disp[:, 2]).max()) * 1.1
    ax3.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='Perfect match')
    ax3.set_xlabel('Ground Truth (Z) [voxels]', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Predicted (Z) [voxels]', fontsize=10, fontweight='bold')
    ax3.set_title('Z Component', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax3, label='TRE (voxels)')
    
    # 4. Magnitude comparison
    ax4 = fig.add_subplot(gs[1, 0])
    gt_mag = np.linalg.norm(gt_disp, axis=1)
    pred_mag = np.linalg.norm(pred_disp, axis=1)
    ax4.scatter(gt_mag, pred_mag, alpha=0.6, s=50, c=tre, cmap='hot')
    lim = max(gt_mag.max(), pred_mag.max()) * 1.1
    ax4.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='Perfect match')
    ax4.set_xlabel('Ground Truth Magnitude [voxels]', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Predicted Magnitude [voxels]', fontsize=10, fontweight='bold')
    ax4.set_title(f'Magnitude (r={results["correlation"]:.3f})', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. TRE histogram
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(tre, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax5.axvline(results['mean_tre'], color='red', linestyle='--', linewidth=2, label=f'Mean: {results["mean_tre"]:.2f}')
    ax5.axvline(results['median_tre'], color='green', linestyle='--', linewidth=2, label=f'Median: {results["median_tre"]:.2f}')
    ax5.set_xlabel('TRE [voxels]', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax5.set_title('Target Registration Error Distribution', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    # 6. Cumulative TRE
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_tre = np.sort(tre)
    cumulative = np.arange(1, len(sorted_tre) + 1) / len(sorted_tre) * 100
    ax6.plot(sorted_tre, cumulative, linewidth=2, color='navy')
    ax6.axvline(1.0, color='red', linestyle='--', alpha=0.7, label=f'1mm: {results["tre_below_1"]:.1f}%')
    ax6.axvline(2.0, color='orange', linestyle='--', alpha=0.7, label=f'2mm: {results["tre_below_2"]:.1f}%')
    ax6.axvline(3.0, color='green', linestyle='--', alpha=0.7, label=f'3mm: {results["tre_below_3"]:.1f}%')
    ax6.set_xlabel('TRE [voxels]', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Cumulative %', fontsize=10, fontweight='bold')
    ax6.set_title('Cumulative TRE Distribution', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Summary text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""
    {'='*100}
    CASE: {results['case']} - MVF Comparison Summary
    {'='*100}
    
    GROUND TRUTH (from Markers):
      • Number of markers:     {results['n_markers']}
      • Mean displacement:     {results['gt_mean']:.3f} ± {results['gt_std']:.3f} voxels
    
    PREDICTED (from Registration Model):
      • Mean displacement:     {results['pred_mean']:.3f} ± {results['pred_std']:.3f} voxels
      • Magnitude correlation: {results['correlation']:.4f}
    
    REGISTRATION ACCURACY:
      • Mean TRE:    {results['mean_tre']:.3f} ± {results['std_tre']:.3f} voxels
      • Median TRE:  {results['median_tre']:.3f} voxels
      • Max TRE:     {results['max_tre']:.3f} voxels
      • TRE < 1mm:   {results['tre_below_1']:.1f}% of markers
      • TRE < 2mm:   {results['tre_below_2']:.1f}% of markers
      • TRE < 3mm:   {results['tre_below_3']:.1f}% of markers
    """
    
    ax7.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(os.path.join(output_dir, f'{results["case"]}_mvf_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def save_comparison_summary(all_results):
    """Save comparison summary to file"""
    output_dir = "/host/d/GitHub_folder/CT_registration_diffusion/mvf_comparison"
    
    with open(os.path.join(output_dir, 'mvf_comparison_summary.txt'), 'w') as f:
        f.write(f"{'='*100}\n")
        f.write(f"MVF COMPARISON: Image Registration vs Ground Truth Markers\n")
        f.write(f"{'='*100}\n\n")
        
        f.write(f"{'Case':<10} {'N_Markers':<12} {'GT_Mean':<12} {'Pred_Mean':<12} {'Mean_TRE':<12} "
               f"{'Median_TRE':<12} {'Max_TRE':<12} {'TRE<1mm%':<12} {'Corr':<10}\n")
        f.write(f"{'-'*100}\n")
        
        for r in all_results:
            f.write(f"{r['case']:<10} {r['n_markers']:<12} {r['gt_mean']:<12.3f} {r['pred_mean']:<12.3f} "
                   f"{r['mean_tre']:<12.3f} {r['median_tre']:<12.3f} {r['max_tre']:<12.3f} "
                   f"{r['tre_below_1']:<12.1f} {r['correlation']:<10.4f}\n")
        
        f.write(f"{'-'*100}\n")
        
        # Overall statistics
        overall_tre = np.concatenate([r['tre'] for r in all_results])
        f.write(f"\nOVERALL STATISTICS:\n")
        f.write(f"  Total markers:  {len(overall_tre)}\n")
        f.write(f"  Mean TRE:       {overall_tre.mean():.3f} ± {overall_tre.std():.3f} voxels\n")
        f.write(f"  Median TRE:     {np.median(overall_tre):.3f} voxels\n")
        f.write(f"  Max TRE:        {overall_tre.max():.3f} voxels\n")
        f.write(f"  TRE < 1.0mm:    {(overall_tre < 1.0).mean()*100:.1f}%\n")
        f.write(f"  TRE < 2.0mm:    {(overall_tre < 2.0).mean()*100:.1f}%\n")
        f.write(f"  Avg Correlation: {np.mean([r['correlation'] for r in all_results]):.4f}\n")

if __name__ == "__main__":
    compare_mvf()
