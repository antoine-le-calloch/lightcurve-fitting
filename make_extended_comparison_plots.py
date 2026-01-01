#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
print(f"Found {len(common_keys)} common object-band pairs")

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

# For cross-metric plots (where we need both metrics from same object)
fall_fwhm_gp_x = []  # fall values where we also have fwhm
fall_fwhm_gp_y = []  # fwhm values where we also have fall
fall_fwhm_vil_x = []
fall_fwhm_vil_y = []

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
    
    # Matched fall-fwhm pairs for scatter plot
    if (not np.isnan(gp['τfall']) and not np.isnan(gp['fwhm']) and
        not np.isnan(vil['τfall']) and not np.isnan(vil['fwhm'])):
        fall_fwhm_gp_x.append(gp['τfall'])
        fall_fwhm_gp_y.append(gp['fwhm'])
        fall_fwhm_vil_x.append(vil['τfall'])
        fall_fwhm_vil_y.append(vil['fwhm'])

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
fall_fwhm_gp_x = np.array(fall_fwhm_gp_x)
fall_fwhm_gp_y = np.array(fall_fwhm_gp_y)
fall_fwhm_vil_x = np.array(fall_fwhm_vil_x)
fall_fwhm_vil_y = np.array(fall_fwhm_vil_y)

# Create 3x3 figure for all 6 metrics
fig = plt.figure(figsize=(18, 14))

# Define helper function for scatter plots with correlation
def add_scatter(ax, x, y, title, xlabel, ylabel, log_scale=False):
    ax.scatter(x, y, alpha=0.5, s=20)
    
    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
    
    # Compute correlation
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) > 2:
        r, p = stats.pearsonr(x[mask], y[mask])
        ax.text(0.05, 0.95, f'r={r:.3f}\nn={len(x)}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

# Row 1: t0 (peak time)
# Panel 1: t0 scatter
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(t0_gp, t0_vil, alpha=0.5, s=20)
ax1.set_xlim(-5, 100)
ax1.set_ylim(-5, 100)
lims = [-5, 100]
ax1.plot(lims, lims, 'k--', alpha=0.3)
if len(t0_gp) >= 2:
    r, p = stats.pearsonr(t0_gp, t0_vil)
    ax1.text(0.05, 0.95, f'r={r:.3f}\nn={len(t0_gp)}', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax1.text(0.05, 0.95, f'n={len(t0_gp)}', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.set_xlabel('GP t0 (days)')
ax1.set_ylabel('Villar t0 (days)')
ax1.set_title('Peak Time Comparison')

# Panel 2: t0 difference histogram
ax2 = plt.subplot(3, 3, 2)
t0_diff = t0_vil - t0_gp
bins = np.linspace(-20,20,50)
ax2.hist(t0_diff, bins=bins, alpha=0.7, edgecolor='black')
median_diff = np.median(t0_diff)
ax2.axvline(median_diff, color='red', linestyle='--', linewidth=2, label=f'Median: {median_diff:.2f} days')
ax2.set_xlabel('Villar t0 - GP t0 (days)')
ax2.set_ylabel('Count')
ax2.set_title('t0 Difference Distribution')
ax2.set_xlim([-20,20])
ax2.legend()

# Panel 3: t0 vs duration (scatter colored by duration)
ax3 = plt.subplot(3, 3, 3)
# Just use t0 for coloring since arrays have different lengths
scatter = ax3.scatter(t0_gp, t0_vil, c=t0_gp, cmap='viridis', alpha=0.6, s=30)
ax3.set_xlim(-5, 100)
ax3.set_ylim(-5, 100)
ax3.plot(lims, lims, 'k--', alpha=0.3)
ax3.set_xlabel('GP t0 (days)')
ax3.set_ylabel('Villar t0 (days)')
ax3.set_title('t0 Comparison (colored by duration)')
plt.colorbar(scatter, ax=ax3, label='Duration (days)')

# Row 2: FWHM (complementary metric)
# Panel 4: FWHM scatter
ax4 = plt.subplot(3, 3, 4)
add_scatter(ax4, fwhm_gp, fwhm_vil, 'FWHM Comparison', 'GP FWHM (days)', 'Villar FWHM (days)')
ax4.set_xlim([0,100])
ax4.set_ylim([0,100]) 

# Panel 5: FWHM ratio (log scale)
ax5 = plt.subplot(3, 3, 5)
fwhm_ratio = fwhm_vil / (fwhm_gp + 1e-6)
fwhm_ratio = fwhm_ratio[fwhm_ratio > 0]
ax5.hist(np.log10(fwhm_ratio), bins=25, alpha=0.7, edgecolor='black')
ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Villar = GP')
ax5.set_xlabel('log10(Villar FWHM / GP FWHM)')
ax5.set_ylabel('Count')
ax5.set_title('FWHM Ratio Distribution')
ax5.legend()

# Panel 6: FWHM vs τfall (relationship test)
ax6 = plt.subplot(3, 3, 6)
if len(fall_fwhm_gp_x) > 0:
    ax6.scatter(fall_fwhm_gp_x, fall_fwhm_gp_y, alpha=0.5, s=20, label='GP')
    ax6.scatter(fall_fwhm_vil_x, fall_fwhm_vil_y, alpha=0.5, s=20, label='Villar')
    ax6.set_xlabel('τfall (days)')
    ax6.set_ylabel('FWHM (days)')
    ax6.set_title('FWHM vs Decay Timescale')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0,100])
    ax6.set_ylim([0,100]) 

# Row 3: Rise and Decay Rates
# Panel 7: Rise rate scatter
ax7 = plt.subplot(3, 3, 7)
add_scatter(ax7, rise_rate_gp, rise_rate_vil, 'Rise Rate Comparison', 
            'GP rise rate (mag/day)', 'Villar rise rate (mag/day)')
ax7.set_xlim([-0.5,0.5])
ax7.set_ylim([-0.5,0.5]) 

# Panel 8: Decay rate scatter
ax8 = plt.subplot(3, 3, 8)
add_scatter(ax8, decay_rate_gp, decay_rate_vil, 'Decay Rate Comparison',
            'GP decay rate (mag/day)', 'Villar decay rate (mag/day)')
ax8.set_xlim([-0.5,0.5])                   
ax8.set_ylim([-0.5,0.5]) 

# Panel 9: Rise vs Decay Rate (both methods)
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(rise_rate_gp, decay_rate_gp, alpha=0.5, s=20, label='GP')
ax9.scatter(rise_rate_vil, decay_rate_vil, alpha=0.5, s=20, label='Villar')
ax9.set_xlabel('Rise rate (mag/day)')
ax9.set_ylabel('Decay rate (mag/day)')
ax9.set_title('Rise vs Decay Rates')
ax9.set_xlim([-0.5,0.5])
ax9.set_ylim([-0.5,0.5])  
ax9.legend()

plt.tight_layout()
plt.savefig('extended_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Extended comparison plot saved to extended_comparison.png")
plt.close()

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"t0 (peak time): {len(t0_gp)} objects")
print(f"  GP mean: {np.mean(t0_gp):.2f} days, std: {np.std(t0_gp):.2f}")
print(f"  Villar mean: {np.mean(t0_vil):.2f} days, std: {np.std(t0_vil):.2f}")
print(f"\nFWHM: {len(fwhm_gp)} objects")
print(f"  GP mean: {np.mean(fwhm_gp):.2f} days, std: {np.std(fwhm_gp):.2f}")
print(f"  Villar mean: {np.mean(fwhm_vil):.2f} days, std: {np.std(fwhm_vil):.2f}")
print(f"\nRise rate: {len(rise_rate_gp)} objects")
print(f"  GP mean: {np.mean(rise_rate_gp):.6f} mag/day, std: {np.std(rise_rate_gp):.6f}")
print(f"  Villar mean: {np.mean(rise_rate_vil):.6f} mag/day, std: {np.std(rise_rate_vil):.6f}")
print(f"\nDecay rate: {len(decay_rate_gp)} objects")
print(f"  GP mean: {np.mean(decay_rate_gp):.6f} mag/day, std: {np.std(decay_rate_gp):.6f}")
print(f"  Villar mean: {np.mean(decay_rate_vil):.6f} mag/day, std: {np.std(decay_rate_vil):.6f}")
