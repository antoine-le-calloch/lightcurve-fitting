#!/usr/bin/env python3
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Read GP data
gp_data = {}
with open('gp_timescale_parameters.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['object'], row['band'])
        gp_data[key] = {
            'τrise': float(row['rise_time_days']),
            'τfall': float(row['decay_time_days']),
            't0': float(row['t0_days']),
            'peak_mag': float(row['peak_mag']),
            'chi2': float(row['chi2']),
            'fwhm': float(row['fwhm_days']) if row['fwhm_days'] != 'NaN' else np.nan,
            'rise_rate': float(row['rise_rate_mag_per_day']) if row['rise_rate_mag_per_day'] != 'NaN' else np.nan,
            'decay_rate': float(row['decay_rate_mag_per_day']) if row['decay_rate_mag_per_day'] != 'NaN' else np.nan,
        }

# Read Villar data
villar_data = {}
with open('villar_timescale_parameters.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['object'], row['band'])
        rise = float(row['rise_time_days']) if row['rise_time_days'] != 'NaN' else np.nan
        decay = float(row['decay_time_days']) if row['decay_time_days'] != 'NaN' else np.nan
        villar_data[key] = {
            'τrise': rise,
            'τfall': decay,
            't0': float(row['peak_time_days']),
            'chi2': float(row['chi2']),
            'fwhm': float(row['fwhm_days']) if row['fwhm_days'] != 'NaN' else np.nan,
            'rise_rate': float(row['rise_rate_mag_per_day']) if row['rise_rate_mag_per_day'] != 'NaN' else np.nan,
            'decay_rate': float(row['decay_rate_mag_per_day']) if row['decay_rate_mag_per_day'] != 'NaN' else np.nan,
        }

common_keys = set(gp_data.keys()) & set(villar_data.keys())

# Prepare data for plotting
t0_gp = []
t0_vil = []
rise_gp = []
rise_vil = []
fall_gp = []
fall_vil = []
fwhm_gp = []
fwhm_vil = []
rise_rate_gp = []
rise_rate_vil = []
decay_rate_gp = []
decay_rate_vil = []

for obj, band in common_keys:
    gp = gp_data[(obj, band)]
    vil = villar_data[(obj, band)]
    
    # t0 data
    t0_gp.append(gp['t0'])
    t0_vil.append(vil['t0'])
    
    # Rise time data (skip if NaN in either)
    if not np.isnan(gp['τrise']) and not np.isnan(vil['τrise']):
        rise_gp.append(gp['τrise'])
        rise_vil.append(vil['τrise'])
    
    # Decay time data
    if not np.isnan(gp['τfall']) and not np.isnan(vil['τfall']):
        fall_gp.append(gp['τfall'])
        fall_vil.append(vil['τfall'])
    
    # FWHM data
    if not np.isnan(gp['fwhm']) and not np.isnan(vil['fwhm']):
        fwhm_gp.append(gp['fwhm'])
        fwhm_vil.append(vil['fwhm'])
    
    # Rise rate data
    if not np.isnan(gp['rise_rate']) and not np.isnan(vil['rise_rate']):
        rise_rate_gp.append(gp['rise_rate'])
        rise_rate_vil.append(vil['rise_rate'])
    
    # Decay rate data
    if not np.isnan(gp['decay_rate']) and not np.isnan(vil['decay_rate']):
        decay_rate_gp.append(gp['decay_rate'])
        decay_rate_vil.append(vil['decay_rate'])

t0_gp = np.array(t0_gp)
t0_vil = np.array(t0_vil)
rise_gp = np.array(rise_gp)
rise_vil = np.array(rise_vil)
fall_gp = np.array(fall_gp)
fall_vil = np.array(fall_vil)
fwhm_gp = np.array(fwhm_gp)
fwhm_vil = np.array(fwhm_vil)
rise_rate_gp = np.array(rise_rate_gp)
rise_rate_vil = np.array(rise_rate_vil)
decay_rate_gp = np.array(decay_rate_gp)
decay_rate_vil = np.array(decay_rate_vil)

decay_rate_gp = np.array(decay_rate_gp)
decay_rate_vil = np.array(decay_rate_vil)

# Create aligned arrays for rise time vs decay rate (only use pairs where both exist)
rise_decay_gp = []
rise_decay_vil = []
for obj, band in common_keys:
    gp = gp_data[(obj, band)]
    vil = villar_data[(obj, band)]
    if not np.isnan(gp['τrise']) and not np.isnan(gp['decay_rate']) and \
       not np.isnan(vil['τrise']) and not np.isnan(vil['decay_rate']):
        rise_decay_gp.append((gp['τrise'], gp['decay_rate']))
        rise_decay_vil.append((vil['τrise'], vil['decay_rate']))

if rise_decay_gp:
    rise_decay_gp = np.array(rise_decay_gp)
    rise_decay_vil = np.array(rise_decay_vil)
    rise_gp_aligned = rise_decay_gp[:, 0]
    decay_rate_gp_aligned = rise_decay_gp[:, 1]
    rise_vil_aligned = rise_decay_vil[:, 0]
    decay_rate_vil_aligned = rise_decay_vil[:, 1]
else:
    rise_gp_aligned = np.array([])
    decay_rate_gp_aligned = np.array([])
    rise_vil_aligned = np.array([])
    decay_rate_vil_aligned = np.array([])

# Create 3x3 figure with all metrics
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Extended GP vs Villar Comparison: All Metrics', fontsize=16, fontweight='bold')

# Helper function for scatter plots with correlation
def add_scatter(ax, x, y, title, xlabel, ylabel, xlim=None, ylim=None):
    ax.scatter(x, y, alpha=0.5, s=20)
    
    # Set limits first if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Get actual limits after setting
    curr_xlim = ax.get_xlim()
    curr_ylim = ax.get_ylim()
    lims = [min(curr_xlim[0], curr_ylim[0]), max(curr_xlim[1], curr_ylim[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
    
    if len(x) > 2:
        r, _ = stats.pearsonr(x, y)
        ax.text(0.05, 0.95, f'r={r:.3f}\nn={len(x)}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Row 1: Timescales (t0, τrise, τfall)
ax1 = plt.subplot(3, 3, 1)
add_scatter(ax1, t0_gp, t0_vil, 'Peak Time (t0)', 'GP t0 (days)', 'Villar t0 (days)', 
            xlim=(-5, 100), ylim=(-5, 100))

ax2 = plt.subplot(3, 3, 2)
if len(rise_gp) > 0:
    # Filter to reasonable range for better visualization
    rise_mask = (rise_gp < 50) & (rise_vil < 50)
    add_scatter(ax2, rise_gp[rise_mask], rise_vil[rise_mask], 
                'Rise Timescale (τrise)', 'GP τrise (days)', 'Villar τrise (days)',
                xlim=(0, 50), ylim=(0, 50))

ax3 = plt.subplot(3, 3, 3)
if len(fall_gp) > 0:
    # Filter to reasonable range for better visualization
    fall_mask = (fall_gp < 100) & (fall_vil < 200)
    add_scatter(ax3, fall_gp[fall_mask], fall_vil[fall_mask],
                'Decay Timescale (τfall)', 'GP τfall (days)', 'Villar τfall (days)',
                xlim=(0, 100), ylim=(0, 200))

# Row 2: Complementary metrics (FWHM, rise rate, decay rate)
ax4 = plt.subplot(3, 3, 4)
if len(fwhm_gp) > 0:
    # Filter outliers
    fwhm_mask = (fwhm_gp < 20) & (fwhm_vil < 150)
    add_scatter(ax4, fwhm_gp[fwhm_mask], fwhm_vil[fwhm_mask],
                'FWHM Comparison', 'GP FWHM (days)', 'Villar FWHM (days)',
                xlim=(0, 20), ylim=(0, 150))

ax5 = plt.subplot(3, 3, 5)
# Filter outliers for rise rate (negative means brightening)
rise_rate_mask = (rise_rate_gp > -0.5) & (rise_rate_gp < 0.2) & \
                 (rise_rate_vil > -0.5) & (rise_rate_vil < 0.2)
add_scatter(ax5, rise_rate_gp[rise_rate_mask], rise_rate_vil[rise_rate_mask],
            'Rise Rate Comparison', 'GP rise rate (mag/day)', 'Villar rise rate (mag/day)',
            xlim=(-0.5, 0.2), ylim=(-0.5, 0.2))

ax6 = plt.subplot(3, 3, 6)
# Filter outliers for decay rate (positive means fading)
decay_rate_mask = (decay_rate_gp > -0.1) & (decay_rate_gp < 0.3) & \
                  (decay_rate_vil > -0.1) & (decay_rate_vil < 0.3)
add_scatter(ax6, decay_rate_gp[decay_rate_mask], decay_rate_vil[decay_rate_mask],
            'Decay Rate Comparison', 'GP decay rate (mag/day)', 'Villar decay rate (mag/day)',
            xlim=(-0.1, 0.3), ylim=(-0.1, 0.3))

# Row 3: Distribution comparisons
ax7 = plt.subplot(3, 3, 7)
t0_diff = t0_vil - t0_gp
# Filter outliers for histogram
t0_diff_filtered = t0_diff[(t0_diff > -50) & (t0_diff < 50)]
ax7.hist(t0_diff_filtered, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
median_diff = np.median(t0_diff)
ax7.axvline(median_diff, color='red', linestyle='--', linewidth=2, label=f'Median: {median_diff:.2f}d')
ax7.set_xlabel('Villar t0 - GP t0 (days)', fontsize=10)
ax7.set_ylabel('Count', fontsize=10)
ax7.set_title('t0 Difference Distribution', fontsize=11, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(alpha=0.3, axis='y')

ax8 = plt.subplot(3, 3, 8)
# FWHM distributions with reasonable bounds
fwhm_gp_filtered = fwhm_gp[fwhm_gp < 15]
fwhm_vil_filtered = fwhm_vil[fwhm_vil < 120]
ax8.hist(fwhm_gp_filtered, bins=25, alpha=0.7, label=f'GP (med={np.median(fwhm_gp):.1f}d)', 
         edgecolor='black', color='blue')
ax8.hist(fwhm_vil_filtered, bins=25, alpha=0.7, label=f'Villar (med={np.median(fwhm_vil):.1f}d)', 
         edgecolor='black', color='orange')
ax8.set_xlabel('FWHM (days)', fontsize=10)
ax8.set_ylabel('Count', fontsize=10)
ax8.set_title('FWHM Distributions', fontsize=11, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(alpha=0.3, axis='y')

ax9 = plt.subplot(3, 3, 9)
# Decay timescale distributions with reasonable bounds
fall_gp_filtered = fall_gp[fall_gp < 50]
fall_vil_filtered = fall_vil[fall_vil < 150]
ax9.hist(fall_gp_filtered, bins=25, alpha=0.7, label=f'GP (med={np.median(fall_gp):.1f}d)', 
         edgecolor='black', color='blue')
ax9.hist(fall_vil_filtered, bins=25, alpha=0.7, label=f'Villar (med={np.median(fall_vil):.1f}d)', 
         edgecolor='black', color='orange')
ax9.set_xlabel('τfall (days)', fontsize=10)
ax9.set_ylabel('Count', fontsize=10)
ax9.set_title('Decay Timescale Distributions', fontsize=11, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gp_villar_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Comparison plot saved to gp_villar_comparison.png")
plt.close()

# Export comparison data to CSV
with open('gp_villar_comparison.csv', 'w') as f:
    f.write('metric,gp_mean,gp_median,gp_std,villar_mean,villar_median,villar_std,n_pairs\n')
    
    # t0
    f.write(f"t0_days,{np.mean(t0_gp):.3f},{np.median(t0_gp):.3f},{np.std(t0_gp):.3f},{np.mean(t0_vil):.3f},{np.median(t0_vil):.3f},{np.std(t0_vil):.3f},{len(t0_gp)}\n")
    
    # FWHM
    f.write(f"fwhm_days,{np.mean(fwhm_gp):.3f},{np.median(fwhm_gp):.3f},{np.std(fwhm_gp):.3f},{np.mean(fwhm_vil):.3f},{np.median(fwhm_vil):.3f},{np.std(fwhm_vil):.3f},{len(fwhm_gp)}\n")
    
    # Rise rate
    if len(rise_rate_gp) > 0:
        f.write(f"rise_rate_mag_day,{np.mean(rise_rate_gp):.6f},{np.median(rise_rate_gp):.6f},{np.std(rise_rate_gp):.6f},{np.mean(rise_rate_vil):.6f},{np.median(rise_rate_vil):.6f},{np.std(rise_rate_vil):.6f},{len(rise_rate_gp)}\n")
    
    # Decay rate
    if len(decay_rate_gp) > 0:
        f.write(f"decay_rate_mag_day,{np.mean(decay_rate_gp):.6f},{np.median(decay_rate_gp):.6f},{np.std(decay_rate_gp):.6f},{np.mean(decay_rate_vil):.6f},{np.median(decay_rate_vil):.6f},{np.std(decay_rate_vil):.6f},{len(decay_rate_gp)}\n")
    
    # Rise timescale
    if len(rise_gp) > 0:
        f.write(f"rise_time_days,{np.mean(rise_gp):.3f},{np.median(rise_gp):.3f},{np.std(rise_gp):.3f},{np.mean(rise_vil):.3f},{np.median(rise_vil):.3f},{np.std(rise_vil):.3f},{len(rise_gp)}\n")
    
    # Decay timescale
    if len(fall_gp) > 0:
        f.write(f"decay_time_days,{np.mean(fall_gp):.3f},{np.median(fall_gp):.3f},{np.std(fall_gp):.3f},{np.mean(fall_vil):.3f},{np.median(fall_vil):.3f},{np.std(fall_vil):.3f},{len(fall_gp)}\n")

print("✓ Comparison statistics saved to gp_villar_comparison.csv")

# Print summary statistics
print(f"\n{'='*70}")
print("EXTENDED PARAMETER COMPARISON: GP vs VILLAR")
print(f"{'='*70}")

print(f"\n1. t0 (Peak Time): {len(t0_gp)} objects")
print(f"   GP:     mean={np.mean(t0_gp):7.2f}d, median={np.median(t0_gp):7.2f}d, std={np.std(t0_gp):7.2f}d")
print(f"   Villar: mean={np.mean(t0_vil):7.2f}d, median={np.median(t0_vil):7.2f}d, std={np.std(t0_vil):7.2f}d")
print(f"   Difference: median={np.median(t0_vil - t0_gp):+7.2f}d")

print(f"\n2. FWHM (Full Width at Half Max): {len(fwhm_gp)} objects")
print(f"   GP:     mean={np.mean(fwhm_gp):7.2f}d, median={np.median(fwhm_gp):7.2f}d, std={np.std(fwhm_gp):7.2f}d")
print(f"   Villar: mean={np.mean(fwhm_vil):7.2f}d, median={np.median(fwhm_vil):7.2f}d, std={np.std(fwhm_vil):7.2f}d")
if len(fwhm_gp) > 0:
    fwhm_ratio = np.median(fwhm_vil / (fwhm_gp + 1e-8))
    print(f"   Villar/GP ratio: {fwhm_ratio:7.3f}")

print(f"\n3. Rise Rate (mag/day): {len(rise_rate_gp)} objects")
print(f"   GP:     mean={np.mean(rise_rate_gp):8.6f}, median={np.median(rise_rate_gp):8.6f}, std={np.std(rise_rate_gp):8.6f}")
print(f"   Villar: mean={np.mean(rise_rate_vil):8.6f}, median={np.median(rise_rate_vil):8.6f}, std={np.std(rise_rate_vil):8.6f}")

print(f"\n4. Decay Rate (mag/day): {len(decay_rate_gp)} objects")
print(f"   GP:     mean={np.mean(decay_rate_gp):8.6f}, median={np.median(decay_rate_gp):8.6f}, std={np.std(decay_rate_gp):8.6f}")
print(f"   Villar: mean={np.mean(decay_rate_vil):8.6f}, median={np.median(decay_rate_vil):8.6f}, std={np.std(decay_rate_vil):8.6f}")

if len(rise_gp) > 0:
    print(f"\n5. Rise Timescale τrise (days): {len(rise_gp)} objects")
    print(f"   GP:     mean={np.mean(rise_gp):7.2f}d, median={np.median(rise_gp):7.2f}d, std={np.std(rise_gp):7.2f}d")
    print(f"   Villar: mean={np.mean(rise_vil):7.2f}d, median={np.median(rise_vil):7.2f}d, std={np.std(rise_vil):7.2f}d")
    if len(rise_gp) > 2:
        r, p = stats.pearsonr(rise_gp, rise_vil)
        print(f"   Correlation: r={r:.3f} (p={p:.2e})")

if len(fall_gp) > 0:
    print(f"\n6. Decay Timescale τfall (days): {len(fall_gp)} objects")
    print(f"   GP:     mean={np.mean(fall_gp):7.2f}d, median={np.median(fall_gp):7.2f}d, std={np.std(fall_gp):7.2f}d")
    print(f"   Villar: mean={np.mean(fall_vil):7.2f}d, median={np.median(fall_vil):7.2f}d, std={np.std(fall_vil):7.2f}d")
    if len(fall_gp) > 2:
        r, p = stats.pearsonr(fall_gp, fall_vil)
        print(f"   Correlation: r={r:.3f} (p={p:.2e})")

print(f"\n{'='*70}\n")
