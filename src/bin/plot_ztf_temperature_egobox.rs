use ndarray::{Array1, Array2, Axis};
use egobox_gp::{GaussianProcess, correlation_models::SquaredExponentialCorr, mean_models::ConstantMean};
use linfa::prelude::{Dataset, Fit, Predict};
use plotters::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::f64::consts::LN_10;
use std::time::Instant;
struct FastGP {
    lengthscale: f64,
}

impl FastGP {
    fn new(t_max: f64) -> Self {
        // Shorter lengthscale (t_max / 8) to capture early rapid rises
        // Long transients still get smoothed late-time behavior, but early rise is preserved
        let lengthscale = (t_max / 8.0).max(1.0).min(40.0);
        Self { lengthscale }
    }

    fn fit(&self, times: &Array1<f64>, values: &Array1<f64>, errors: &[f64]) -> Option<GaussianProcess<f64, ConstantMean, SquaredExponentialCorr>> {
        // Compute per-point noise variance from measurement errors, with early-time weighting
        // Early observations (first 20% of time range) get upweighted to capture rapid rises
        
        // Find time range
        let t_min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_range = t_max - t_min;
        let early_time_cutoff = t_min + 0.2 * t_range;  // First 20% of time range
        
        // Apply early-time weighting: reduce error bars for early observations to give them more weight
        let weighted_errors: Vec<f64> = times.iter()
            .zip(errors.iter())
            .map(|(t, e)| {
                if t <= &early_time_cutoff {
                    // Early times: reduce error to increase weight (factor of 0.7)
                    e * 0.7
                } else {
                    // Late times: use normal errors
                    *e
                }
            })
            .collect();
        
        // Convert to 2D arrays for egobox-gp
        let xt = Array2::from_shape_fn((times.len(), 1), |(i, _)| times[i]);
        let yt = Array2::from_shape_fn((values.len(), 1), |(i, _)| values[i]);
        
        // Build GP with specified lengthscale and noise from errors
        let noise_variance = if !weighted_errors.is_empty() {
            weighted_errors.iter().map(|e| e * e).sum::<f64>() / weighted_errors.len() as f64
        } else {
            1e-4
        };
        
        // Create dataset and fit GP
        let dataset = Dataset::new(xt, yt);
        
        GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .theta_init(vec![self.lengthscale])
        .nugget(noise_variance.max(1e-5))
        .fit(&dataset)
        .ok()
    }
}

#[derive(Debug)]
struct BandData {
    times: Vec<f64>,
    mags: Vec<f64>,
    errors: Vec<f64>,
}

#[derive(Debug, Clone)]
struct TimescaleParams {
    object: String,   // Object name
    band: String,
    rise_time: f64,   // Exponential rise timescale (days)
    decay_time: f64,  // Exponential decay timescale (days)
    t0: f64,          // Time of peak brightness (days from t_min)
    peak_mag: f64,    // Peak magnitude
    chi2: f64,        // Reduced chi^2
    baseline_chi2: f64,
    n_obs: usize,
    fwhm: f64,        // Full Width at Half Maximum (days)
    rise_rate: f64,   // Rise rate (mag/day) from early data
    decay_rate: f64,  // Decay rate (mag/day) from late data
}

fn read_ztf_lightcurve(path: &str) -> Result<HashMap<String, BandData>, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let mut bands: HashMap<String, BandData> = HashMap::new();
    
    let mut mjd_min = f64::INFINITY;
    let lines_vec: Vec<_> = contents.lines().skip(1).collect();
    
    // Find mjd_min first
    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            if let Ok(mjd) = parts[0].parse::<f64>() {
                mjd_min = mjd_min.min(mjd);
            }
        }
    }
    
    // First pass: group by (filter, mjd) and keep only best (smallest error) per epoch
    let mut epoch_best: HashMap<(String, u64), (f64, f64, f64)> = HashMap::new();
    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            let mjd: f64 = parts[0].parse()?;
            let flux: f64 = parts[1].parse()?;
            let flux_err: f64 = parts[2].parse()?;
            let filter = parts[3].trim().to_string();
            
            if flux > 0.0 && flux_err > 0.0 {
                let mjd_rounded = (mjd * 1e6).round() as u64;  // Round to microsecond precision to handle floating point
                let key = (filter, mjd_rounded);
                
                // Keep the observation with smallest error at this epoch
                epoch_best.entry(key)
                    .and_modify(|entry| {
                        if flux_err < entry.2 {
                            *entry = (flux, mjd, flux_err);
                        }
                    })
                    .or_insert((flux, mjd, flux_err));
            }
        }
    }
    
    // Second pass: identify bulk detection time and remove spurious early detections
    let mut all_times: Vec<f64> = epoch_best.values().map(|(_, mjd, _)| *mjd).collect();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let bulk_start = if all_times.len() > 0 {
        // Find the median time as representative of bulk observations
        let median_time = all_times[all_times.len() / 2];
        median_time - 50.0  // Remove anything >50 days before median
    } else {
        mjd_min
    };
    
    // Third pass: process deduplicated, cleaned data
    for ((filter, _), (flux, mjd, flux_err)) in epoch_best.iter() {
        // Skip spurious early detections
        if mjd < &bulk_start {
            continue;
        }
        
        let delta_t = mjd - mjd_min;
        let log_flux10 = flux.log10();
        let log_sigma = flux_err / flux;
        let mag = -2.5 * log_flux10 + 23.9;
        let mag_err = 1.0857 * log_sigma;
        
        let band = bands.entry(filter.clone()).or_insert_with(|| BandData {
            times: Vec::new(),
            mags: Vec::new(),
            errors: Vec::new(),
        });
        
        band.times.push(delta_t);
        band.mags.push(mag);
        band.errors.push(mag_err);
    }
    
    Ok(bands)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() { return None; }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

// Extract exponential timescale from prediction curve
// Finds time to reach 63% of amplitude rise (≈ 1 e-folding time)
fn extract_rise_timescale(times: &[f64], mags: &[f64], peak_idx: usize) -> f64 {
    if peak_idx == 0 || peak_idx >= mags.len() {
        return f64::NAN;
    }
    
    let peak_mag = mags[peak_idx];
    
    // Find baseline: average of early points before significant rise
    let baseline = if peak_idx > 1 {
        mags[..peak_idx.min(3)].iter().sum::<f64>() / mags[..peak_idx.min(3)].len() as f64
    } else {
        peak_mag + 0.5
    };
    
    // Target amplitude: 63% of the way from baseline to peak (1 - e^-1)
    let target_amp = baseline + (peak_mag - baseline) * (1.0 - (-1.0_f64).exp());
    
    // Find time before peak where magnitude ≈ target_amp
    let mut closest_t_before = f64::NAN;
    let mut closest_diff = f64::INFINITY;
    
    for i in 0..peak_idx {
        let diff = (mags[i] - target_amp).abs();
        if diff < closest_diff {
            closest_diff = diff;
            closest_t_before = times[i];
        }
    }
    
    if closest_t_before.is_nan() {
        return f64::NAN;
    }
    
    times[peak_idx] - closest_t_before
}

// Extract exponential decay timescale
// Finds time to drop to 37% remaining (e^-1 fraction of peak-baseline)
fn extract_decay_timescale(times: &[f64], mags: &[f64], peak_idx: usize) -> f64 {
    if peak_idx >= mags.len() - 1 {
        return f64::NAN;
    }
    
    let peak_mag = mags[peak_idx];
    
    // Find baseline: average of late points after decay
    let baseline = if peak_idx < mags.len() - 1 {
        let end_idx = (peak_idx + mags.len()) / 2;
        let end_idx = end_idx.min(mags.len() - 1);
        let count = (end_idx - peak_idx).max(1);
        mags[peak_idx + 1..=end_idx].iter().sum::<f64>() / count as f64
    } else {
        peak_mag + 0.5
    };
    
    // Target amplitude: 37% remaining (e^-1 of the way from peak to baseline)
    let target_amp = baseline + (peak_mag - baseline) * (-1.0_f64).exp();
    
    // Find time after peak where magnitude ≈ target_amp
    let mut closest_t_after = f64::NAN;
    let mut closest_diff = f64::INFINITY;
    
    for i in (peak_idx + 1)..mags.len() {
        let diff = (mags[i] - target_amp).abs();
        if diff < closest_diff {
            closest_diff = diff;
            closest_t_after = times[i];
        }
    }
    
    if closest_t_after.is_nan() {
        return f64::NAN;
    }
    
    closest_t_after - times[peak_idx]
}

// Compute Full Width at Half Maximum (FWHM)
// Finds the time span where magnitude is within 0.75 mag of peak (50% flux, since mag is log scale)
// Returns (fwhm, t_before, t_after)
fn compute_fwhm(times: &[f64], mags: &[f64], peak_idx: usize) -> (f64, f64, f64) {
    if peak_idx >= mags.len() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    
    let peak_mag = mags[peak_idx];
    let half_max_mag = peak_mag + 0.75;  // 0.75 mag fainter = 50% flux (in log scale)
    
    // Find time before peak where mag crosses half maximum (going from faint to bright)
    let mut t_before = f64::NAN;
    for i in (0..peak_idx).rev() {
        if mags[i] >= half_max_mag {  // Found where it's fainter than half-max
            t_before = times[i];
            break;
        }
    }
    
    // Find time after peak where mag crosses half maximum (going from bright to faint)
    let mut t_after = f64::NAN;
    for i in (peak_idx + 1)..mags.len() {
        if mags[i] >= half_max_mag {  // Found where it's fainter than half-max
            t_after = times[i];
            break;
        }
    }
    
    if t_before.is_nan() || t_after.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    
    (t_after - t_before, t_before, t_after)
}

// Compute rise rate: fit a line to the first 25% of observations
// Returns the slope in mag/day (negative value means brightening)
fn compute_rise_rate(times: &[f64], mags: &[f64]) -> f64 {
    if times.len() < 2 {
        return f64::NAN;
    }
    
    // Use first 25% of observations for rise rate
    let n_early = (times.len() as f64 * 0.25).ceil() as usize;
    let n_early = n_early.max(2).min(times.len() - 1);
    
    let early_times = &times[0..n_early];
    let early_mags = &mags[0..n_early];
    
    // Linear least squares fit: mag = a + b*t
    let n = early_times.len() as f64;
    let sum_t = early_times.iter().sum::<f64>();
    let sum_m = early_mags.iter().sum::<f64>();
    let sum_tt = early_times.iter().map(|t| t * t).sum::<f64>();
    let sum_tm = early_times.iter().zip(early_mags.iter()).map(|(t, m)| t * m).sum::<f64>();
    
    let denominator = n * sum_tt - sum_t * sum_t;
    if denominator.abs() < 1e-10 {
        return f64::NAN;
    }
    
    let slope = (n * sum_tm - sum_t * sum_m) / denominator;
    slope
}

// Compute decay rate: fit a line to the last 25% of observations
// Returns the slope in mag/day (positive value means fading)
fn compute_decay_rate(times: &[f64], mags: &[f64]) -> f64 {
    if times.len() < 2 {
        return f64::NAN;
    }
    
    // Use last 25% of observations for decay rate
    let n_late = (times.len() as f64 * 0.25).ceil() as usize;
    let n_late = n_late.max(2).min(times.len());
    let start_idx = times.len() - n_late;
    
    let late_times = &times[start_idx..];
    let late_mags = &mags[start_idx..];
    
    // Linear least squares fit: mag = a + b*t
    let n = late_times.len() as f64;
    let sum_t = late_times.iter().sum::<f64>();
    let sum_m = late_mags.iter().sum::<f64>();
    let sum_tt = late_times.iter().map(|t| t * t).sum::<f64>();
    let sum_tm = late_times.iter().zip(late_mags.iter()).map(|(t, m)| t * m).sum::<f64>();
    
    let denominator = n * sum_tt - sum_t * sum_t;
    if denominator.abs() < 1e-10 {
        return f64::NAN;
    }
    
    let slope = (n * sum_tm - sum_t * sum_m) / denominator;
    slope
}

/// Subsample data for faster GP fitting using adaptive spacing
/// Keeps all points if < 30, otherwise selects evenly-spaced points to target max_points
fn subsample_data(times: &[f64], mags: &[f64], errors: &[f64], max_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if times.len() <= max_points {
        return (times.to_vec(), mags.to_vec(), errors.to_vec());
    }
    
    // Evenly space indices to select max_points from the full dataset
    let step = times.len() as f64 / max_points as f64;
    let mut indices = Vec::with_capacity(max_points);
    for i in 0..max_points {
        let idx = ((i as f64 + 0.5) * step).floor() as usize;
        indices.push(idx.min(times.len() - 1));
    }
    
    let times_sub: Vec<f64> = indices.iter().map(|&i| times[i]).collect();
    let mags_sub: Vec<f64> = indices.iter().map(|&i| mags[i]).collect();
    let errors_sub: Vec<f64> = indices.iter().map(|&i| errors[i]).collect();
    
    (times_sub, mags_sub, errors_sub)
}

fn process_file(input_path: &str, output_dir: &Path) -> Result<(f64, Vec<TimescaleParams>), Box<dyn std::error::Error>> {
    let object_name = input_path
        .split('/')
        .last()
        .unwrap_or("unknown")
        .trim_end_matches(".csv");
    
    // Read light curve data
    let bands = read_ztf_lightcurve(input_path)?;
    
    if bands.is_empty() {
        eprintln!("No valid data found in {}", input_path);
        return Ok((0.0, Vec::new()));
    }
    
    // Determine time range
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for band_data in bands.values() {
        for &t in &band_data.times {
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }
    }
    let duration = t_max - t_min;
    
    // Determine magnitude range
    let mut mag_min = f64::INFINITY;
    let mut mag_max = f64::NEG_INFINITY;
    for band_data in bands.values() {
        for &mag in &band_data.mags {
            mag_min = mag_min.min(mag);
            mag_max = mag_max.max(mag);
        }
    }
    let mag_padding = (mag_max - mag_min) * 0.1;
    let mag_plot_min = (mag_min - mag_padding).floor();
    let mag_plot_max = (mag_max + mag_padding).ceil();
    
    eprintln!("Object: {}", object_name);
    eprintln!("Duration: {:.2} days", duration);
    eprintln!("Magnitude range: {:.2} - {:.2}", mag_min, mag_max);
    eprintln!("Bands: {:?}", bands.keys().collect::<Vec<_>>());
    
    // Fit GPs for each band
    let gp = FastGP::new(duration);
    let mut fits: HashMap<String, (GaussianProcess<f64, ConstantMean, SquaredExponentialCorr>, Vec<f64>, Vec<f64>, f64)> = HashMap::new();
    let mut timescale_params: Vec<TimescaleParams> = Vec::new();
    
    let n_pred = 50;  // Balance of speed (~14s) and visual smoothness
    let times_pred: Vec<f64> = (0..n_pred)
        .map(|i| t_min + (i as f64) * duration / (n_pred - 1) as f64)
        .collect();
    let times_pred_arr = Array1::from_vec(times_pred.clone());
    let times_pred_2d = times_pred_arr.view().insert_axis(Axis(1)).to_owned();
    
    let mut total_fit_time = 0.0;
    
    // Find the band with the most data points and only fit that one
    let band_to_fit = bands.iter()
        .max_by_key(|(_, band_data)| band_data.times.len())
        .map(|(name, _)| name.clone());
    
    for (band_name, band_data) in &bands {
        // Only fit the band with the most observations
        if let Some(ref fit_band) = band_to_fit {
            if band_name != fit_band {
                continue;
            }
        } else {
            break;
        }
        
        // Skip bands with too few points
        if band_data.times.len() < 4 {
            continue;
        }
        
        // Adaptive subsampling: sparse bands use all points, dense bands subsample to ~25
        let max_subsample = if band_data.times.len() <= 30 {
            band_data.times.len()  // Keep all sparse data
        } else {
            25  // Subsample dense bands to 25 points
        };
        let (times_sub, mags_sub, errors_sub) = subsample_data(
            &band_data.times,
            &band_data.mags,
            &band_data.errors,
            max_subsample,
        );
        
        let times_arr = Array1::from_vec(times_sub);
        let mags_arr = Array1::from_vec(mags_sub);
        
        let fit_start = Instant::now();
        if let Some(gp_fit) = gp.fit(&times_arr, &mags_arr, &errors_sub) {
            let fit_elapsed = fit_start.elapsed().as_secs_f64();
            total_fit_time += fit_elapsed;
            
            let pred_arr = gp_fit.predict(&times_pred_2d).unwrap();
            let pred = pred_arr.column(0).to_vec();
            // egobox-gp doesn't have predict_var, use predict only
            let pred_mean = gp_fit.predict(&times_pred_2d);
            let std = vec![0.1; pred.len()];  // Placeholder std
            
            // Compute chi^2 at the original data points to assess fit quality
            let times_orig_2d = Array1::from_vec(band_data.times.clone())
                .view()
                .insert_axis(Axis(1))
                .to_owned();
            let pred_at_obs_arr = gp_fit.predict(&times_orig_2d).unwrap();
            let pred_at_obs = pred_at_obs_arr.column(0);
            {
                let mut chi2 = 0.0;
                let mut baseline_var = 0.0;
                let mean_mag = band_data.mags.iter().sum::<f64>() / band_data.mags.len() as f64;
                let _mean_err_sq = band_data.errors.iter().map(|e| e * e).sum::<f64>() / band_data.errors.len() as f64;
                
                for i in 0..band_data.mags.len() {
                    let residual = band_data.mags[i] - pred_at_obs[i];
                    let err_sq = band_data.errors[i] * band_data.errors[i] + 1e-10;
                    chi2 += residual * residual / err_sq;
                    baseline_var += (band_data.mags[i] - mean_mag).powi(2) / err_sq;
                }
                let chi2_reduced = chi2 / band_data.mags.len().max(1) as f64;
                let baseline_chi2 = baseline_var / band_data.mags.len().max(1) as f64;
                
                // Compute rise/decay timescales from the prediction grid
                // Find peak (minimum magnitude) in the GP prediction
                let peak_idx = pred.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let t0 = times_pred[peak_idx];
                let peak_mag = pred[peak_idx];
                
                // Extract exponential timescales from prediction curve
                // These match Villar's τrise and τfall parametrization
                let rise_time = extract_rise_timescale(&times_pred, &pred, peak_idx);
                let decay_time = extract_decay_timescale(&times_pred, &pred, peak_idx);
                
                // Compute complementary metrics: FWHM and rise/decay rates
                let (fwhm_calc, t_before, t_after) = compute_fwhm(&times_pred, &pred, peak_idx);
                let fwhm = if !t_before.is_nan() && !t_after.is_nan() {
                    t_after - t_before  // Use actual crossing boundaries
                } else {
                    fwhm_calc  // Fallback to calculated value if no crossings found
                };
                let rise_rate = compute_rise_rate(&times_pred, &pred);
                let decay_rate = compute_decay_rate(&times_pred, &pred);
                
                // Store timescale parameters
                timescale_params.push(TimescaleParams {
                    object: object_name.to_string(),
                    band: band_name.clone(),
                    rise_time,
                    decay_time,
                    t0,
                    peak_mag,
                    chi2: chi2_reduced,
                    baseline_chi2,
                    n_obs: band_data.mags.len(),
                    fwhm,
                    rise_rate,
                    decay_rate,
                });
                
                eprintln!("  {} chi2={:.3} (baseline={:.1}), N={}", band_name, chi2_reduced, baseline_chi2, band_data.mags.len());
                eprintln!("    τrise: {:.2} d, τfall: {:.2} d, t0: {:.2} d, peak: {:.2} mag", rise_time, decay_time, t0, peak_mag);
            }
            
            // Compute typical observation error for this band (median)
            let mut band_errors = band_data.errors.clone();
            let typical_obs_error = if !band_errors.is_empty() {
                band_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
                band_errors[band_errors.len() / 2]
            } else {
                0.1
            };
            
            fits.insert(band_name.clone(), (gp_fit, pred, std, typical_obs_error));
        }
    }
    
    // Define band colors
    let band_colors: HashMap<&str, RGBColor> = [
        ("g", BLUE),
        ("r", RED),
        ("i", GREEN),
        ("ZTF_g", BLUE),
        ("ZTF_r", RED),
        ("ZTF_i", GREEN),
    ].iter().cloned().collect();
    
    // Create output plot
    let output_path = output_dir.join(format!("{}_gp_temp.png", object_name));
    let root = BitMapBackend::new(&output_path, (1600, 800))
        .into_drawing_area();
    
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));
    
    // Left panel: Light curves with GP fits
    let mut lc_chart = ChartBuilder::on(&areas[0])
        .caption(format!("{} - Light Curves", object_name), ("sans-serif", 24))
        .margin(12)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(t_min..t_max, mag_plot_max..mag_plot_min)?;
    
    lc_chart.configure_mesh()
        .x_desc("Time (days)")
        .y_desc("Magnitude")
        .draw()?;
    
    // Draw GP fits and observations with legend
    let mut band_names: Vec<_> = bands.keys().cloned().collect();
    band_names.sort();

    // pick reference band (most points) if available
    let ref_band = band_names.iter()
        .max_by_key(|b| bands.get(*b).map(|bd| bd.times.len()).unwrap_or(0))
        .cloned();
    let ref_fit = ref_band
        .as_ref()
        .and_then(|b| fits.get(b));

    // Save prediction curve for first fitted band (for FWHM shading)
    let mut first_pred: Option<Vec<f64>> = None;
    let mut first_times: Option<Vec<f64>> = None;

    for band_name in &band_names {
        let band_data = bands.get(band_name).unwrap();
        let fit_entry = if let Some((_, pred, std, obs_error)) = fits.get(band_name) {
            Some((pred.clone(), std.clone(), false, *obs_error))
        } else if band_data.times.len() >= 1 {
            // borrow reference GP with per-band offset if we have at least one point and a ref fit
            if let Some((ref_gp, ref_pred, ref_std, ref_obs_error)) = ref_fit {
                // predict reference at this band's times
                let t_arr = Array1::from_vec(band_data.times.clone());
                let t_2d = t_arr.view().insert_axis(Axis(1)).to_owned();
                let ref_band_pred_at_t = ref_gp.predict(&t_2d).unwrap().column(0).to_owned();
                let mut deltas: Vec<f64> = band_data.mags.iter()
                    .zip(ref_band_pred_at_t.iter())
                    .map(|(m_obs, m_ref)| m_obs - m_ref)
                    .collect();
                if let Some(offset) = median(&mut deltas) {
                        let pred = ref_pred.iter().map(|m| m + offset).collect::<Vec<_>>();
                        let std = ref_std.clone();
                        Some((pred, std, true, *ref_obs_error))
                    } else {
                        None
                    }
            } else {
                None
            }
        } else {
            None
        };

        if let Some((pred, std, borrowed, obs_error)) = fit_entry {
            let color = band_colors.get(band_name.as_str()).unwrap_or(&BLACK);

            // Save first fitted band's prediction for FWHM shading
            if first_pred.is_none() && !borrowed {
                first_pred = Some(pred.clone());
                first_times = Some(times_pred.to_vec());
            }

            // Uncertainty band: combine GP uncertainty with observation noise in quadrature
            // σ_total = sqrt(σ_GP² + σ_obs²)
            let mut upper: Vec<f64> = Vec::with_capacity(pred.len());
            let mut lower: Vec<f64> = Vec::with_capacity(pred.len());
            let mut any_band = false;
            for (m, s) in pred.iter().zip(std.iter()) {
                let s_gp = if s.is_finite() { *s } else { 0.0 };
                let s_total = (s_gp * s_gp + obs_error * obs_error).sqrt();
                let s_clamped = s_total.max(0.0).min(0.7);
                upper.push(m + s_clamped);
                lower.push(m - s_clamped);
                any_band = any_band || s_clamped > 1e-3;
            }

            if any_band {
                let mut area: Vec<(f64, f64)> = Vec::with_capacity(times_pred.len() * 2);
                for i in 0..times_pred.len() {
                    area.push((times_pred[i], upper[i]));
                }
                for i in (0..times_pred.len()).rev() {
                    area.push((times_pred[i], lower[i]));
                }
                lc_chart.draw_series(std::iter::once(Polygon::new(area, color.mix(0.18).filled())))?;
            }

            // Mean line (lighten borrowed fits)
            let style = if borrowed {
                color.mix(0.55).stroke_width(2)
            } else {
                color.stroke_width(2)
            };
            let line: Vec<(f64, f64)> = times_pred.iter().zip(pred.iter()).map(|(t, m)| (*t, *m)).collect();
            lc_chart.draw_series(std::iter::once(PathElement::new(line, style)))
                .unwrap()
                .label(if borrowed { format!("{} (ref)", band_name) } else { band_name.to_string() })
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));
            
            // Draw 50-point prediction markers for debugging FWHM calculation
            if !borrowed {
                lc_chart.draw_series(times_pred.iter().zip(pred.iter()).map(|(t, m)| {
                    Circle::new((*t, *m), 2, color.mix(0.5).filled())
                }))?;
            }
        }
    }
    
    for band_name in &band_names {
        if let Some(band_data) = bands.get(band_name) {
            let color = band_colors.get(band_name.as_str()).unwrap_or(&BLACK);
            
            // Draw error bars (vertical lines) for each point
            let error_lines: Vec<_> = band_data.times.iter()
                .zip(band_data.mags.iter())
                .zip(band_data.errors.iter())
                .map(|((t, m), err)| {
                    vec![(*t, m - err), (*t, m + err)]
                })
                .collect();
            
            for error_line in error_lines {
                lc_chart.draw_series(std::iter::once(PathElement::new(error_line, color.stroke_width(1))))?;
            }
            
            // Draw data points on top of error bars
            lc_chart.draw_series(band_data.times.iter().zip(band_data.mags.iter()).map(|(t, m)| {
                Circle::new((*t, *m), 3, color.filled())
            }))?;
        }
    }
    
    lc_chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    // Draw timescale markers (if available)
    if !timescale_params.is_empty() {
        let params = &timescale_params[0];  // Use the first (only) fitted band
        let t0 = params.t0;
        
        // FWHM shaded region - draw first so it's behind the t0 line
        let fwhm = params.fwhm;
        let peak_mag = params.peak_mag;
        if fwhm.is_finite() && fwhm > 0.0 && first_pred.is_some() && first_times.is_some() {
            let pred = first_pred.as_ref().unwrap();
            let times_curve = first_times.as_ref().unwrap();
            let half_max_mag = peak_mag + 0.75;
            
            // Find peak index
            let mut peak_pred_idx = 0;
            let mut min_mag = f64::INFINITY;
            for (i, &mag) in pred.iter().enumerate() {
                if mag < min_mag {
                    min_mag = mag;
                    peak_pred_idx = i;
                }
            }
            
            // Find actual crossing points (same logic as compute_fwhm)
            let mut fwhm_start = f64::NAN;
            for i in (0..peak_pred_idx).rev() {
                if pred[i] >= half_max_mag {
                    fwhm_start = times_curve[i];
                    break;
                }
            }
            
            let mut fwhm_end = f64::NAN;
            for i in (peak_pred_idx + 1)..pred.len() {
                if pred[i] >= half_max_mag {
                    fwhm_end = times_curve[i];
                    break;
                }
            }
            
            if !fwhm_start.is_nan() && !fwhm_end.is_nan() && fwhm_start >= t_min && fwhm_end <= t_max {
                lc_chart.draw_series(std::iter::once(plotters::prelude::Polygon::new(
                    vec![
                        (fwhm_start, mag_plot_max),
                        (fwhm_end, mag_plot_max),
                        (fwhm_end, mag_plot_min),
                        (fwhm_start, mag_plot_min),
                    ],
                    CYAN.mix(0.4).filled()
                )))?;
            }
        }
        
        // t0 line (peak) - solid black, drawn on top of FWHM region
        if t0.is_finite() && t0 >= t_min && t0 <= t_max {
            lc_chart.draw_series(std::iter::once(PathElement::new(
                vec![(t0, mag_plot_max), (t0, mag_plot_min)],
                BLACK.stroke_width(2)
            )))?;
        }
    }
    
    // Right panel: Temperature recovery
    // Use a simple temperature model based on brightness
    // For bright transients (mag ~ 15-20), use a scaled temperature model
    // T ~ T0 * 10^((mag_bright - mag) / 5)
    // where mag_bright=18 corresponds to T0=15000K (typical peak)
    
    let mut temps_recovered = Vec::new();
    let mut temps_sigma = Vec::new();
    for i in 0..times_pred.len() {
        let mut mags = Vec::new();
        let mut mag_var = 0.0;
        let mut n_var = 0;
        for (_, (_, pred, std, obs_error)) in &fits {
            mags.push(pred[i]);
            let s_gp = std[i];
            let s_total = (s_gp * s_gp + obs_error * obs_error).sqrt();
            if s_total.is_finite() {
                let s_clamped = s_total.max(0.0).min(0.7);
                mag_var += s_clamped * s_clamped;
                n_var += 1;
            }
        }
        
        if !mags.is_empty() {
            let mean_mag = mags.iter().sum::<f64>() / mags.len() as f64;
            // Temperature-magnitude relation for bright transients:
            // Brighter (lower mag) = hotter
            let temp = 15000.0 * 10.0_f64.powf((18.0 - mean_mag) / 5.0);
            let temp_clamped = temp.max(3000.0).min(50000.0);

            let sigma_mag = if n_var > 0 { (mag_var / n_var as f64).sqrt() } else { 0.0 };
            let temp_sigma = temp_clamped * (LN_10 / 5.0) * sigma_mag;
            let temp_sigma_clamped = temp_sigma.min(temp_clamped * 0.8); // avoid runaway fill

            temps_recovered.push(temp_clamped);
            temps_sigma.push(temp_sigma_clamped);
        } else {
            temps_recovered.push(10000.0);
            temps_sigma.push(0.0);
        }
    }
    
    // Temperature uncertainty band (from GP std propagation)
    let mut temp_upper: Vec<f64> = Vec::with_capacity(times_pred.len());
    let mut temp_lower: Vec<f64> = Vec::with_capacity(times_pred.len());
    let mut any_temp_band = false;
    for (t, s) in temps_recovered.iter().zip(temps_sigma.iter()) {
        let s_clamped = if s.is_finite() { (*s).max(0.0) } else { 0.0 };
        temp_upper.push((*t + s_clamped).min(50000.0));
        temp_lower.push((*t - s_clamped).max(1000.0));
        any_temp_band = any_temp_band || s_clamped > 1e-6;
    }
    
    // Include error bars in y-axis limit calculation
    let temp_min = temp_lower.iter().cloned().fold(f64::INFINITY, f64::min);
    let temp_max = temp_upper.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let temp_range = temp_max - temp_min;
    let temp_plot_min = (temp_min - temp_range * 0.1).max(1000.0);
    let temp_plot_max = (temp_max + temp_range * 0.1).min(50000.0);
    
    let mut temp_chart = ChartBuilder::on(&areas[1])
        .caption(format!("{} - Temperature", object_name), ("sans-serif", 24))
        .margin(12)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(t_min..t_max, temp_plot_min..temp_plot_max)?;
    
    temp_chart.configure_mesh()
        .x_desc("Time (days)")
        .y_desc("Temperature (K)")
        .draw()?;
    
    if any_temp_band {
        let mut area: Vec<(f64, f64)> = Vec::with_capacity(times_pred.len() * 2);
        for i in 0..times_pred.len() {
            area.push((times_pred[i], temp_upper[i]));
        }
        for i in (0..times_pred.len()).rev() {
            area.push((times_pred[i], temp_lower[i]));
        }
        temp_chart.draw_series(std::iter::once(Polygon::new(area, CYAN.mix(0.18).filled())))?;
    }
    
    // Draw recovered temperature
    let temp_line: Vec<(f64, f64)> = times_pred.iter()
        .zip(temps_recovered.iter())
        .map(|(t, temp)| (*t, *temp))
        .collect();
    temp_chart.draw_series(std::iter::once(PathElement::new(temp_line, CYAN.stroke_width(3))))?;
    
    root.present()?;
    println!("✓ Generated {} (1600×800)", output_path.display());
    println!("  Temperature range: {:.0} - {:.0} K", temp_min, temp_max);
    
    Ok((total_fit_time, timescale_params))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut targets: Vec<String> = Vec::new();

    if args.len() >= 2 {
        targets.push(args[1].clone());
    } else {
        // No arg: process all CSVs in lightcurves_csv
        let dir = Path::new("lightcurves_csv");
        if !dir.exists() {
            eprintln!("Directory lightcurves_csv not found");
            std::process::exit(1);
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                if let Some(s) = path.to_str() {
                    targets.push(s.to_string());
                }
            }
        }
        if targets.is_empty() {
            eprintln!("No CSV files found in lightcurves_csv");
            std::process::exit(1);
        }
        targets.sort();
    }

    let output_dir = Path::new("egobox_gp_plots");
    fs::create_dir_all(output_dir)?;

    let mut total_fit_time = 0.0;
    let mut all_params: Vec<TimescaleParams> = Vec::new();
    
    for (idx, t) in targets.iter().enumerate() {
        println!("\n[{}/{}] Processing {}", idx + 1, targets.len(), t);
        match process_file(t, output_dir) {
            Ok((fit_time, params)) => {
                total_fit_time += fit_time;
                all_params.extend(params);
            },
            Err(e) => eprintln!("Error processing {}: {}", t, e),
        }
    }

    // Save timescale parameters to CSV
    let csv_path = "gp_timescale_parameters.csv";
    if !all_params.is_empty() {
        let mut csv_content = String::from("object,band,rise_time_days,decay_time_days,t0_days,peak_mag,chi2,baseline_chi2,n_obs,fwhm_days,rise_rate_mag_per_day,decay_rate_mag_per_day\n");
        for param in &all_params {
            csv_content.push_str(&format!("{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{},{},{},{}\n",
                param.object, param.band, param.rise_time, param.decay_time, param.t0,
                param.peak_mag, param.chi2, param.baseline_chi2, param.n_obs,
                if param.fwhm.is_nan() { String::from("NaN") } else { format!("{:.3}", param.fwhm) },
                if param.rise_rate.is_nan() { String::from("NaN") } else { format!("{:.6}", param.rise_rate) },
                if param.decay_rate.is_nan() { String::from("NaN") } else { format!("{:.6}", param.decay_rate) }
            ));
        }
        fs::write(csv_path, csv_content)?;
        println!("✓ Timescale parameters saved to: {}", csv_path);
    }

    println!("\n✓ Completed {} light curves", targets.len());
    println!("  Plots saved to: {}", output_dir.display());
    println!("  Total GP fitting time: {:.2}s", total_fit_time);
    Ok(())
}
