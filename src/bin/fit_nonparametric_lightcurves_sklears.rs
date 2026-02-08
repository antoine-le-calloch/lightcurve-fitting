use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_gaussian_process::{
    GaussianProcessRegressor, GprTrained,
    kernels::{ConstantKernel, RBF, WhiteKernel, ProductKernel, SumKernel},
};
use sklears_core::traits::{Untrained, Fit, Predict};
use plotters::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::f64::consts::LN_10;
use std::time::Instant;
use lightcurve_fiting::lightcurve_common::{BandData, read_ztf_lightcurve, median, extract_rise_timescale, extract_decay_timescale, compute_fwhm, compute_rise_rate, compute_decay_rate};

struct FastGP {
    base: GaussianProcessRegressor<Untrained>,
}

impl FastGP {
    fn new(t_max: f64) -> Self {
        // Use a smaller amplitude and shorter lengthscale to reduce large uncertainty
        let amp = 0.2; // kernel amplitude (reduced)
        let cst: Box<dyn sklears_gaussian_process::Kernel> = Box::new(ConstantKernel::new(amp));
        // even shorter lengthscale to keep covariance local and reduce overall variance
        let lengthscale = (t_max / 16.0).max(0.3).min(12.0);
        let rbf: Box<dyn sklears_gaussian_process::Kernel> = Box::new(RBF::new(lengthscale));
        let prod = Box::new(ProductKernel::new(vec![cst, rbf]));
        // keep a very small white-noise term; primary noise handled by alpha
        let noise = 1e-10;
        let white = Box::new(WhiteKernel::new(noise));
        let kernel = SumKernel::new(vec![prod, white]);

        // reduce alpha (regularization) to lower predictive variance
        let base = GaussianProcessRegressor::new()
            .kernel(Box::new(kernel))
            .alpha(1e-10)
            .normalize_y(true);

        Self { base }
    }

    fn fit(&self, times: &Array1<f64>, values: &Array1<f64>, errors: &[f64]) -> Option<GaussianProcessRegressor<GprTrained>> {
        // Early-time weighting similar to egobox version
        let t_min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_range = t_max - t_min;
        let early_time_cutoff = t_min + 0.2 * t_range;

        let weighted_errors: Vec<f64> = times.iter()
            .zip(errors.iter())
            .map(|(t, e)| if t <= &early_time_cutoff { *e * 0.7 } else { *e })
            .collect();

        let avg_error_var = if !weighted_errors.is_empty() {
            weighted_errors.iter().map(|e| e * e).sum::<f64>() / weighted_errors.len() as f64
        } else { 1e-4 };

        let alpha_with_errors = avg_error_var.max(1e-5);
        let gp_with_alpha = self.base.clone().alpha(alpha_with_errors);
        let xt = times.view().insert_axis(Axis(1)).to_owned();
        gp_with_alpha.fit(&xt, values).ok()
    }
}

// BandData struct imported from lightcurve_common

#[derive(Debug, Clone)]
struct TimescaleParams {
    object: String,
    band: String,
    rise_time: f64,
    decay_time: f64,
    t0: f64,
    peak_mag: f64,
    chi2: f64,
    baseline_chi2: f64,
    n_obs: usize,
    fwhm: f64,
    rise_rate: f64,
    decay_rate: f64,
    gp_dfdt_now: f64,
    gp_dfdt_next: f64,
    gp_d2fdt2_now: f64,
    gp_predicted_mag_1d: f64,
    gp_predicted_mag_2d: f64,
    gp_time_to_peak: f64,
    gp_extrap_slope: f64,
    gp_T_peak: f64,
    gp_T_now: f64,
    gp_dTdt_peak: f64,
    gp_dTdt_now: f64,
    gp_sigma_f: f64,
    gp_peak_to_peak: f64,
    gp_snr_max: f64,
    gp_dfdt_max: f64,
    gp_dfdt_min: f64,
    gp_frac_of_peak: f64,
    gp_post_var_mean: f64,
    gp_post_var_max: f64,
    gp_skewness: f64,
    gp_kurtosis: f64,
    gp_n_inflections: f64,
    gp_amp: f64,
    gp_lengthscale: f64,
    gp_alpha: f64,
}

#[derive(Debug, Clone)]
struct PredictiveFeatures {
    gp_dfdt_now: f64,
    gp_dfdt_next: f64,
    gp_d2fdt2_now: f64,
    gp_predicted_mag_1d: f64,
    gp_predicted_mag_2d: f64,
    gp_time_to_peak: f64,
    gp_extrap_slope: f64,
    gp_T_peak: f64,
    gp_T_now: f64,
    gp_dTdt_peak: f64,
    gp_dTdt_now: f64,
    gp_sigma_f: f64,
    gp_peak_to_peak: f64,
    gp_snr_max: f64,
    gp_dfdt_max: f64,
    gp_dfdt_min: f64,
    gp_frac_of_peak: f64,
    gp_post_var_mean: f64,
    gp_post_var_max: f64,
    gp_skewness: f64,
    gp_kurtosis: f64,
    gp_n_inflections: f64,
}

fn subsample_data(times: &[f64], mags: &[f64], errors: &[f64], max_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if times.len() <= max_points { return (times.to_vec(), mags.to_vec(), errors.to_vec()); }
    let step = times.len() as f64 / max_points as f64; let mut indices = Vec::with_capacity(max_points);
    for i in 0..max_points { let idx = ((i as f64 + 0.5) * step).floor() as usize; indices.push(idx.min(times.len() - 1)); }
    let times_sub: Vec<f64> = indices.iter().map(|&i| times[i]).collect(); let mags_sub: Vec<f64> = indices.iter().map(|&i| mags[i]).collect(); let errors_sub: Vec<f64> = indices.iter().map(|&i| errors[i]).collect();
    (times_sub, mags_sub, errors_sub)
}

// Helper: linear interpolate pred array defined on times_pred
fn interp_pred_at(times_pred: &[f64], pred: &[f64], t: f64) -> f64 {
    if pred.is_empty() || times_pred.is_empty() { return f64::NAN; }
    if t <= times_pred[0] { return pred[0]; }
    if t >= *times_pred.last().unwrap() { return *pred.last().unwrap(); }
    // find interval
    let mut i = 0usize;
    while i + 1 < times_pred.len() && times_pred[i+1] < t { i += 1; }
    let t0 = times_pred[i]; let t1 = times_pred[i+1]; let y0 = pred[i]; let y1 = pred[i+1];
    let w = (t - t0) / (t1 - t0);
    y0 * (1.0 - w) + y1 * w
}

fn compute_predictive_features(
    t_last: f64,
    t0: f64,
    temps: &[f64],
    times_pred: &[f64],
    pred: &[f64],
    std: &[f64],
    obs_mags: &[f64],
    obs_errors: &[f64],
) -> PredictiveFeatures {
    let dt = 1.0;
    // sample around t_last using interpolation
    let f_m1 = interp_pred_at(times_pred, pred, t_last - dt);
    let f_0  = interp_pred_at(times_pred, pred, t_last);
    let f_p1 = interp_pred_at(times_pred, pred, t_last + dt);
    let f_p2 = interp_pred_at(times_pred, pred, t_last + 2.0*dt);
    let f_p3 = interp_pred_at(times_pred, pred, t_last + 3.0*dt);

    // temperature features same as original using temps and times_pred
    let (gp_T_peak, gp_T_now, gp_dTdt_peak, gp_dTdt_now) = if !temps.is_empty() && !times_pred.is_empty() {
        let (t_peak_idx, t_peak_val) = temps.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, &v)| (i, v)).unwrap_or((0, f64::NAN));
        let t_now = temps.last().copied().unwrap_or(f64::NAN);
        let dt_temp = if times_pred.len() > 1 { times_pred[1] - times_pred[0] } else { 1.0 };
        let dtdt_peak = if t_peak_idx > 0 && t_peak_idx < temps.len() - 1 { (temps[t_peak_idx + 1] - temps[t_peak_idx - 1]) / (2.0 * dt_temp) } else { f64::NAN };
        let dtdt_now = if temps.len() > 1 && times_pred.len() > 1 { let n = temps.len(); let dt_temp = times_pred[n-1] - times_pred[n-2]; (temps[n-1] - temps[n-2]) / dt_temp } else { f64::NAN };
        (t_peak_val, t_now, dtdt_peak, dtdt_now)
    } else { (f64::NAN, f64::NAN, f64::NAN, f64::NAN) };

    // Variability strength
    let gp_sigma_f = if !pred.is_empty() { let mean = pred.iter().sum::<f64>() / pred.len() as f64; (pred.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / pred.len() as f64).sqrt() } else { f64::NAN };

    let gp_peak_to_peak = if !pred.is_empty() { pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - pred.iter().cloned().fold(f64::INFINITY, f64::min) } else { f64::NAN };

    let gp_snr_max = if !obs_errors.is_empty() { obs_mags.iter().zip(obs_errors.iter()).map(|(mag, err)| mag.abs() / err).fold(f64::NEG_INFINITY, f64::max) } else { f64::NAN };

    // Derivative features on prediction grid
    let gp_dfdt_max = if pred.len() > 1 && times_pred.len() > 1 { let dt_grid = times_pred[1] - times_pred[0]; (0..pred.len()-1).map(|i| (pred[i+1] - pred[i]) / dt_grid).fold(f64::NEG_INFINITY, f64::max) } else { f64::NAN };
    let gp_dfdt_min = if pred.len() > 1 && times_pred.len() > 1 { let dt_grid = times_pred[1] - times_pred[0]; (0..pred.len()-1).map(|i| (pred[i+1] - pred[i]) / dt_grid).fold(f64::INFINITY, f64::min) } else { f64::NAN };

    let gp_frac_of_peak = if !pred.is_empty() { let peak_mag = pred.iter().cloned().fold(f64::INFINITY, f64::min); let last_mag = pred.last().copied().unwrap_or(f64::NAN); last_mag / peak_mag } else { f64::NAN };

    let gp_post_var_mean = if !std.is_empty() { std.iter().map(|s| s * s).sum::<f64>() / std.len() as f64 } else { f64::NAN };
    let gp_post_var_max = if !std.is_empty() { std.iter().map(|s| s * s).fold(f64::NEG_INFINITY, f64::max) } else { f64::NAN };

    let (gp_skewness, gp_kurtosis) = if !pred.is_empty() && pred.len() > 3 { let mean = pred.iter().sum::<f64>() / pred.len() as f64; let variance = pred.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / pred.len() as f64; let std_dev = variance.sqrt(); if std_dev > 1e-10 { let skew = pred.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / pred.len() as f64; let kurt = pred.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / pred.len() as f64 - 3.0; (skew, kurt) } else { (f64::NAN, f64::NAN) } } else { (f64::NAN, f64::NAN) };

    // Inflection points from second derivative sign changes
    let gp_n_inflections = if pred.len() > 2 && times_pred.len() > 2 {
        let dt_grid = times_pred[1] - times_pred[0]; let mut d2: Vec<f64> = Vec::with_capacity(pred.len().saturating_sub(2));
        for i in 1..(pred.len()-1) { let v = (pred[i+1] - 2.0*pred[i] + pred[i-1]) / (dt_grid * dt_grid); d2.push(v); }
        let eps = 1e-6_f64; let mut count = 0usize; for i in 0..(d2.len().saturating_sub(1)) { let a = d2[i]; let b = d2[i+1]; if a.is_finite() && b.is_finite() { if a.abs() > eps && b.abs() > eps && (a * b) < 0.0 { count += 1; } } } count as f64
    } else { f64::NAN };

    // small local derivatives around t_last
    let gp_dfdt_now_local = if f_0.is_finite() && f_m1.is_finite() { (f_0 - f_m1) / dt } else { f64::NAN };
    let gp_dfdt_next_local = if f_p1.is_finite() && f_0.is_finite() { (f_p1 - f_0) / dt } else { f64::NAN };
    let gp_d2fdt2_now_local = if f_p1.is_finite() && f_0.is_finite() && f_m1.is_finite() { (f_p1 - 2.0*f_0 + f_m1) / (dt*dt) } else { f64::NAN };

    PredictiveFeatures {
        gp_dfdt_now: gp_dfdt_now_local,
        gp_dfdt_next: gp_dfdt_next_local,
        gp_d2fdt2_now: gp_d2fdt2_now_local,
        gp_predicted_mag_1d: f_p1,
        gp_predicted_mag_2d: f_p2,
        gp_time_to_peak: t0 - t_last,
        gp_extrap_slope: (f_p3 - f_p2) / dt,
        gp_T_peak,
        gp_T_now,
        gp_dTdt_peak,
        gp_dTdt_now,
        gp_sigma_f,
        gp_peak_to_peak,
        gp_snr_max,
        gp_dfdt_max,
        gp_dfdt_min,
        gp_frac_of_peak,
        gp_post_var_mean,
        gp_post_var_max,
        gp_skewness,
        gp_kurtosis,
        gp_n_inflections,
    }
}

fn process_file(input_path: &str, output_dir: &Path, do_plot: bool) -> Result<(f64, Vec<TimescaleParams>), Box<dyn std::error::Error>> {
    let object_name = input_path.split('/').last().unwrap_or("unknown").trim_end_matches(".csv");
    let bands = read_ztf_lightcurve(input_path, true)?;;
    if bands.is_empty() { eprintln!("No valid data found in {}", input_path); return Ok((0.0, Vec::new())); }

    let mut t_min = f64::INFINITY; let mut t_max = f64::NEG_INFINITY;
    for band_data in bands.values() { for &t in &band_data.times { t_min = t_min.min(t); t_max = t_max.max(t); } }
    let duration = t_max - t_min;
    let mut mag_min = f64::INFINITY; let mut mag_max = f64::NEG_INFINITY; for band_data in bands.values() { for &mag in &band_data.mags { mag_min = mag_min.min(mag); mag_max = mag_max.max(mag); } }
    let mag_padding = (mag_max - mag_min) * 0.1; let mag_plot_min = (mag_min - mag_padding).floor(); let mag_plot_max = (mag_max + mag_padding).ceil();

    eprintln!("Object: {}", object_name);
    eprintln!("Duration: {:.2} days", duration);
    eprintln!("Magnitude range: {:.2} - {:.2}", mag_min, mag_max);
    eprintln!("Bands: {:?}", bands.keys().collect::<Vec<_>>());

    let gp = FastGP::new(duration);
    let mut fits: HashMap<String, (GaussianProcessRegressor<GprTrained>, Vec<f64>, Vec<f64>, f64)> = HashMap::new();
    let mut timescale_params: Vec<TimescaleParams> = Vec::new();

    let n_pred = 50;
    let times_pred: Vec<f64> = (0..n_pred).map(|i| t_min + (i as f64) * duration / (n_pred - 1) as f64).collect();
    let times_pred_arr = Array1::from_vec(times_pred.clone());
    let times_pred_2d = times_pred_arr.view().insert_axis(Axis(1)).to_owned();

    let mut total_fit_time = 0.0;

    let band_to_fit = bands.iter().max_by_key(|(_, band_data)| band_data.times.len()).map(|(name, _)| name.clone());
    if let Some(ref_name) = &band_to_fit {
        eprintln!("Reference band chosen for borrowing: {}", ref_name);
    }

    let min_points_for_independent_fit = 5;
    for (band_name, band_data) in &bands {
        if band_data.times.len() < min_points_for_independent_fit { continue; }
        let max_subsample = if band_data.times.len() <= 30 { band_data.times.len() } else { 25 };
        let (times_sub, mags_sub, errors_sub) = subsample_data(&band_data.times, &band_data.mags, &band_data.errors, max_subsample);
        let times_arr = Array1::from_vec(times_sub);
        let mags_arr = Array1::from_vec(mags_sub);

        let fit_start = Instant::now();

        // Per-band hyperparameter grid search (amplitude, lengthscale factor, alpha)
        let xt_sub = times_arr.view().insert_axis(Axis(1)).to_owned();
        let mut best_gp_fit: Option<GaussianProcessRegressor<GprTrained>> = None;
        let mut best_rms = f64::INFINITY;
        let mut best_params: Option<(f64, f64, f64)> = None; // (amp, lengthscale, alpha)

        let amp_candidates = vec![0.05, 0.1, 0.2, 0.4];
        let ls_factors = vec![4.0, 6.0, 8.0, 12.0, 16.0, 24.0];
        let avg_error_var = if !errors_sub.is_empty() { errors_sub.iter().map(|e| e*e).sum::<f64>() / errors_sub.len() as f64 } else { 1e-4 };
        // compute median dt to avoid lengthscales smaller than data sampling
        let mut dt_vec: Vec<f64> = Vec::new();
        for w in 1..times_arr.len() { dt_vec.push(times_arr[w] - times_arr[w-1]); }
        dt_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_dt = if !dt_vec.is_empty() { dt_vec[dt_vec.len()/2] } else { 1.0 };
        let min_lengthscale = (median_dt * 2.0).max(0.1);
        // choose alphas anchored to observed measurement variance to avoid extreme under-regularization
        let alpha_candidates = vec![avg_error_var.max(1e-6), avg_error_var.max(1e-4)];

        for &amp in &amp_candidates {
            for &factor in &ls_factors {
                let lengthscale = (duration / factor).max(0.1);
                if lengthscale < min_lengthscale { continue; }
                for &alpha in &alpha_candidates {
                    let cst: Box<dyn sklears_gaussian_process::Kernel> = Box::new(ConstantKernel::new(amp));
                    let rbf: Box<dyn sklears_gaussian_process::Kernel> = Box::new(RBF::new(lengthscale));
                    let prod = Box::new(ProductKernel::new(vec![cst, rbf]));
                    let white = Box::new(WhiteKernel::new(1e-10));
                    let kernel = SumKernel::new(vec![prod, white]);

                    let gp_candidate = GaussianProcessRegressor::new()
                        .kernel(Box::new(kernel))
                        .alpha(alpha)
                        .normalize_y(true);

                    if let Ok(trained) = gp_candidate.fit(&xt_sub, &mags_arr) {
                        if let Ok(pred_at_obs) = trained.predict(&xt_sub) {
                            let mut residuals_sq = 0.0f64;
                            for i in 0..mags_arr.len() {
                                let residual = mags_arr[i] - pred_at_obs[i];
                                residuals_sq += residual * residual;
                            }
                            let rms = (residuals_sq / mags_arr.len() as f64).sqrt();

                            // compute mean predictive std at observed points to penalize overfitting
                            let mut mean_pred_std = 0.0f64;
                            if let Ok((pred_std_obs, _)) = trained.predict_with_std(&xt_sub) {
                                let v = pred_std_obs.to_vec();
                                let mut ssum = 0.0f64; let mut scnt = 0usize;
                                for s in v.iter() { if s.is_finite() { ssum += *s; scnt += 1; } }
                                if scnt > 0 { mean_pred_std = ssum / scnt as f64; }
                            }

                            // reject candidates with absurd extrapolated peak magnitudes
                            let mut pred_grid_min = f64::INFINITY;
                            if let Ok(pred_grid) = trained.predict(&times_pred_2d) {
                                for v in pred_grid.iter() { pred_grid_min = pred_grid_min.min(*v); }
                            }
                            let obs_min = mags_arr.iter().cloned().fold(f64::INFINITY, f64::min);
                            if pred_grid_min.is_finite() {
                                // require predicted peak to be within ~6 mag of observed range
                                if (pred_grid_min - obs_min).abs() > 6.0 {
                                    continue;
                                }
                            }

                            // score combines fit residuals and predictive uncertainty (penalize extremely low variance)
                            let penalty_coef = 0.6_f64;
                            let score = rms + penalty_coef * mean_pred_std;
                            if score.is_finite() && score < best_rms {
                                best_rms = score;
                                best_gp_fit = Some(trained.clone());
                                best_params = Some((amp, lengthscale, alpha));
                            }
                        }
                    }
                }
            }
        }

        let fit_elapsed = fit_start.elapsed().as_secs_f64(); total_fit_time += fit_elapsed;

        if best_gp_fit.is_none() {
            // fallback to previous API if grid search failed
            if let Some(gp_fit_fallback) = gp.fit(&times_arr, &mags_arr, &errors_sub) {
                best_gp_fit = Some(gp_fit_fallback);
            }
        }

        if let Some(gp_fit) = best_gp_fit {
            if let Some((amp_ch, ls_ch, alpha_ch)) = best_params {
                eprintln!("  Selected GP params for {}: amp={:.3}, ls={:.3}, alpha={:.3}", band_name, amp_ch, ls_ch, alpha_ch);
            }
            // predictions
            let pred = gp_fit.predict(&times_pred_2d).ok().unwrap().to_vec();
            let (pred_std, _) = gp_fit.predict_with_std(&times_pred_2d).ok().unwrap();
            let mut std = pred_std.to_vec();

            // compute chi2
            let times_orig_2d = Array1::from_vec(band_data.times.clone()).view().insert_axis(Axis(1)).to_owned();
            if let Ok(pred_at_obs) = gp_fit.predict(&times_orig_2d) {
                let mut chi2 = 0.0; let mut baseline_var = 0.0; let mean_mag = band_data.mags.iter().sum::<f64>() / band_data.mags.len() as f64; let _mean_err_sq = band_data.errors.iter().map(|e| e*e).sum::<f64>() / band_data.errors.len() as f64;
                // compute residuals and chi2
                let mut residuals_sq = 0.0f64;
                for i in 0..band_data.mags.len() {
                    let residual = band_data.mags[i] - pred_at_obs[i];
                    let err_sq = band_data.errors[i] * band_data.errors[i] + 1e-10;
                    residuals_sq += residual * residual;
                    chi2 += residual * residual / err_sq;
                    baseline_var += (band_data.mags[i] - mean_mag).powi(2) / err_sq;
                }
                let rms_residual = (residuals_sq / band_data.mags.len() as f64).sqrt();

                // Scale predicted std to match observed residuals at data locations
                        if let Ok((pred_std_obs, _)) = gp_fit.predict_with_std(&times_orig_2d) {
                    let pred_std_obs_vec = pred_std_obs.to_vec();
                    let mut sum = 0.0f64; let mut cnt = 0usize;
                    for s in pred_std_obs_vec.iter() { if s.is_finite() { sum += *s; cnt += 1; } }
                    if cnt > 0 {
                        let mean_pred_std_obs = sum / cnt as f64;
                                if mean_pred_std_obs > 1e-12 && rms_residual.is_finite() {
                                    // compute scale to match RMS residuals, but clamp to avoid extreme upscaling
                                    let mut scale = (rms_residual / mean_pred_std_obs).max(0.05).min(5.0);
                                    // apply conservative shrink to overall uncertainty
                                    scale *= 0.6_f64;
                                    for v in std.iter_mut() { *v = (*v) * scale; }
                                }
                    }
                }
                let chi2_reduced = chi2 / band_data.mags.len().max(1) as f64; let baseline_chi2 = baseline_var / band_data.mags.len().max(1) as f64;
                let peak_idx = pred.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0);
                let t0 = times_pred[peak_idx]; let peak_mag = pred[peak_idx];
                let rise_time = extract_rise_timescale(&times_pred, &pred, peak_idx); let decay_time = extract_decay_timescale(&times_pred, &pred, peak_idx);
                let (fwhm_calc, t_before, t_after) = compute_fwhm(&times_pred, &pred, peak_idx); let fwhm = if !t_before.is_nan() && !t_after.is_nan() { t_after - t_before } else { fwhm_calc };
                let rise_rate = compute_rise_rate(&times_pred, &pred); let decay_rate = compute_decay_rate(&times_pred, &pred);

                let t_last = *band_data.times.last().unwrap();
                let predictive = compute_predictive_features(t_last, t0, &[], &times_pred, &pred, &std, &band_data.mags, &band_data.errors);

                timescale_params.push(TimescaleParams {
                    object: object_name.to_string(), band: band_name.clone(), rise_time, decay_time, t0, peak_mag, chi2: chi2_reduced, baseline_chi2, n_obs: band_data.mags.len(), fwhm, rise_rate, decay_rate,
                    gp_dfdt_now: predictive.gp_dfdt_now, gp_dfdt_next: predictive.gp_dfdt_next, gp_d2fdt2_now: predictive.gp_d2fdt2_now,
                    gp_predicted_mag_1d: predictive.gp_predicted_mag_1d, gp_predicted_mag_2d: predictive.gp_predicted_mag_2d, gp_time_to_peak: predictive.gp_time_to_peak, gp_extrap_slope: predictive.gp_extrap_slope,
                    gp_T_peak: predictive.gp_T_peak, gp_T_now: predictive.gp_T_now, gp_dTdt_peak: predictive.gp_dTdt_peak, gp_dTdt_now: predictive.gp_dTdt_now,
                    gp_sigma_f: predictive.gp_sigma_f, gp_peak_to_peak: predictive.gp_peak_to_peak, gp_snr_max: predictive.gp_snr_max, gp_dfdt_max: predictive.gp_dfdt_max, gp_dfdt_min: predictive.gp_dfdt_min,
                    gp_frac_of_peak: predictive.gp_frac_of_peak, gp_post_var_mean: predictive.gp_post_var_mean, gp_post_var_max: predictive.gp_post_var_max, gp_skewness: predictive.gp_skewness, gp_kurtosis: predictive.gp_kurtosis,
                    gp_n_inflections: predictive.gp_n_inflections,
                    gp_amp: best_params.map_or(f64::NAN, |p| p.0),
                    gp_lengthscale: best_params.map_or(f64::NAN, |p| p.1),
                    gp_alpha: best_params.map_or(f64::NAN, |p| p.2),
                });

                eprintln!("  {} chi2={:.3} (baseline={:.1}), N={}", band_name, chi2_reduced, baseline_chi2, band_data.mags.len());
                eprintln!("    τrise: {:.2} d, τfall: {:.2} d, t0: {:.2} d, peak: {:.2} mag", rise_time, decay_time, t0, peak_mag);
            }

            let mut band_errors = band_data.errors.clone(); let typical_obs_error = if !band_errors.is_empty() { band_errors.sort_by(|a,b| a.partial_cmp(b).unwrap()); band_errors[band_errors.len()/2] } else { 0.1 };
            fits.insert(band_name.clone(), (gp_fit, pred, std, typical_obs_error));
        }
    }

    if do_plot {
        // Create plots (reuse color map)
        let band_colors: HashMap<&str, RGBColor> = [("g", BLUE), ("r", RED), ("i", GREEN), ("ZTF_g", BLUE), ("ZTF_r", RED), ("ZTF_i", GREEN)].iter().cloned().collect();
        let output_path = output_dir.join(format!("{}_gp_temp_sklears.png", object_name));
        let root = BitMapBackend::new(&output_path, (1600, 800)).into_drawing_area();
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
        let ref_fit = ref_band.as_ref().and_then(|b| fits.get(b));

        // Save prediction curve for first fitted band (for FWHM shading)
        let mut first_pred: Option<Vec<f64>> = None;
        let mut first_times: Option<Vec<f64>> = None;

        for band_name in &band_names {
            let band_data = bands.get(band_name).unwrap();
            let fit_entry = if let Some((_, pred, std, obs_error)) = fits.get(band_name) {
                Some((pred.clone(), std.clone(), false, *obs_error))
            } else if band_data.times.len() >= 1 && band_data.times.len() < 5 {
                if let Some((ref_gp, ref_pred, ref_std, ref_obs_error)) = ref_fit {
                    let t_arr = Array1::from_vec(band_data.times.clone());
                    let t_2d = t_arr.view().insert_axis(Axis(1)).to_owned();
                    let ref_band_pred_at_t = ref_gp.predict(&t_2d).ok().unwrap().to_vec();
                    let mut deltas: Vec<f64> = band_data.mags.iter().zip(ref_band_pred_at_t.iter()).map(|(m_obs, m_ref)| m_obs - m_ref).collect();
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
                    for i in 0..times_pred.len() { area.push((times_pred[i], upper[i])); }
                    for i in (0..times_pred.len()).rev() { area.push((times_pred[i], lower[i])); }
                    lc_chart.draw_series(std::iter::once(Polygon::new(area, color.mix(0.18).filled())))?;
                }

                // Mean line (lighten borrowed fits)
                let style = if borrowed { color.mix(0.55).stroke_width(2) } else { color.stroke_width(2) };
                let line: Vec<(f64, f64)> = times_pred.iter().zip(pred.iter()).map(|(t, m)| (*t, *m)).collect();
                lc_chart.draw_series(std::iter::once(PathElement::new(line, style))).unwrap()
                    .label(if borrowed { format!("{} (ref)", band_name) } else { band_name.to_string() })
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));

                // Draw 50-point prediction markers for debugging FWHM calculation
                if !borrowed {
                    lc_chart.draw_series(times_pred.iter().zip(pred.iter()).map(|(t, m)| { Circle::new((*t, *m), 2, color.mix(0.5).filled()) }))?;
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
                    .map(|((t, m), err)| { vec![(*t, m - err), (*t, m + err)] })
                    .collect();

                for error_line in error_lines {
                    lc_chart.draw_series(std::iter::once(PathElement::new(error_line, color.stroke_width(1))))?;
                }

                // Draw data points on top of error bars
                lc_chart.draw_series(band_data.times.iter().zip(band_data.mags.iter()).map(|(t, m)| { Circle::new((*t, *m), 3, color.filled()) }))?;
            }
        }

        lc_chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        // Draw timescale markers (if available)
        if !timescale_params.is_empty() {
            let params = &timescale_params[0];
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
                    if mag < min_mag { min_mag = mag; peak_pred_idx = i; }
                }

                // Find actual crossing points
                let mut fwhm_start = f64::NAN;
                for i in (0..peak_pred_idx).rev() {
                    if pred[i] >= half_max_mag { fwhm_start = times_curve[i]; break; }
                }

                let mut fwhm_end = f64::NAN;
                for i in (peak_pred_idx + 1)..pred.len() {
                    if pred[i] >= half_max_mag { fwhm_end = times_curve[i]; break; }
                }

                if !fwhm_start.is_nan() && !fwhm_end.is_nan() && fwhm_start >= t_min && fwhm_end <= t_max {
                    lc_chart.draw_series(std::iter::once(plotters::prelude::Polygon::new(
                        vec![(fwhm_start, mag_plot_max), (fwhm_end, mag_plot_max), (fwhm_end, mag_plot_min), (fwhm_start, mag_plot_min)],
                        CYAN.mix(0.4).filled()
                    )))?;
                }
            }

            // t0 line (peak)
            if t0.is_finite() && t0 >= t_min && t0 <= t_max {
                lc_chart.draw_series(std::iter::once(PathElement::new(vec![(t0, mag_plot_max), (t0, mag_plot_min)], BLACK.stroke_width(2))))?;
            }
        }

        // Right panel: Temperature recovery
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
                let temp = 15000.0 * 10.0_f64.powf((18.0 - mean_mag) / 5.0);
                let temp_clamped = temp.max(3000.0).min(50000.0);

                let sigma_mag = if n_var > 0 { (mag_var / n_var as f64).sqrt() } else { 0.0 };
                let temp_sigma = temp_clamped * (LN_10 / 5.0) * sigma_mag;
                let temp_sigma_clamped = temp_sigma.min(temp_clamped * 0.8);
                temps_recovered.push(temp_clamped);
                temps_sigma.push(temp_sigma_clamped);
            } else {
                temps_recovered.push(10000.0);
                temps_sigma.push(0.0);
            }
        }

        // Temperature uncertainty band
        let mut temp_upper: Vec<f64> = Vec::with_capacity(times_pred.len());
        let mut temp_lower: Vec<f64> = Vec::with_capacity(times_pred.len());
        let mut any_temp_band = false;
        for (t, s) in temps_recovered.iter().zip(temps_sigma.iter()) {
            let s_clamped = if s.is_finite() { (*s).max(0.0) } else { 0.0 };
            temp_upper.push((*t + s_clamped).min(50000.0));
            temp_lower.push((*t - s_clamped).max(1000.0));
            any_temp_band = any_temp_band || s_clamped > 1e-6;
        }

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
            for i in 0..times_pred.len() { area.push((times_pred[i], temp_upper[i])); }
            for i in (0..times_pred.len()).rev() { area.push((times_pred[i], temp_lower[i])); }
            temp_chart.draw_series(std::iter::once(Polygon::new(area, CYAN.mix(0.18).filled())))?;
        }

        let temp_line: Vec<(f64, f64)> = times_pred.iter().zip(temps_recovered.iter()).map(|(t, temp)| (*t, *temp)).collect();
        temp_chart.draw_series(std::iter::once(PathElement::new(temp_line, CYAN.stroke_width(3))))?;

        root.present()?;
        println!("✓ Generated {} (1600×800)", output_path.display());
        println!("  Temperature range: {:.0} - {:.0} K", temp_min, temp_max);
    } // end if do_plot


    eprintln!("  Completed {} bands, GP fitting: {:.4}s", bands.len(), total_fit_time);
    Ok((total_fit_time, timescale_params))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let args: Vec<String> = std::env::args().collect();
    let do_plot = !args.iter().any(|a| a == "--no-plot");

    let mut targets: Vec<String> = Vec::new();
    for arg in args.iter().skip(1) {
        if arg.starts_with("--") { continue; }
        let p = Path::new(arg);
        if p.is_dir() {
            for entry in fs::read_dir(p)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                    if let Some(s) = path.to_str() { targets.push(s.to_string()); }
                }
            }
        } else {
            targets.push(arg.clone());
        }
    }
    if targets.is_empty() {
        let dir = Path::new("lightcurves_csv");
        if !dir.exists() { eprintln!("Directory lightcurves_csv not found"); std::process::exit(1); }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                if let Some(s) = path.to_str() { targets.push(s.to_string()); }
            }
        }
        if targets.is_empty() { eprintln!("No CSV files found in lightcurves_csv"); std::process::exit(1); }
    }
    targets.sort();

    let output_dir = Path::new("egobox_gp_plots");
    fs::create_dir_all(output_dir)?;

    let n_targets = targets.len();
    let progress = AtomicUsize::new(0);
    let total_start = Instant::now();

    let file_results: Vec<(String, Result<(f64, Vec<TimescaleParams>), String>)> = targets
        .par_iter()
        .map(|t| {
            let result = process_file(t, output_dir, do_plot);
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            eprint!("\r[{}/{}] {}", done, n_targets, t);
            (t.clone(), result.map_err(|e| e.to_string()))
        })
        .collect();

    let total_elapsed = total_start.elapsed().as_secs_f64();
    eprintln!();

    let mut total_fit_time = 0.0;
    let mut all_params: Vec<TimescaleParams> = Vec::new();
    let mut n_success = 0usize;
    for (t, result) in file_results {
        match result {
            Ok((fit_time, params)) => {
                total_fit_time += fit_time;
                if !params.is_empty() { n_success += 1; }
                all_params.extend(params);
            }
            Err(e) => eprintln!("Error processing {}: {}", t, e),
        }
    }

    // Write global CSV
    let csv_path = "gp_timescale_parameters_sklears.csv";
    if !all_params.is_empty() {
        let mut csv_content = String::from("object,band,rise_time_days,decay_time_days,t0_days,peak_mag,chi2,baseline_chi2,n_obs,fwhm_days,rise_rate_mag_per_day,decay_rate_mag_per_day,gp_dfdt_now,gp_dfdt_next,gp_d2fdt2_now,gp_predicted_mag_1d,gp_predicted_mag_2d,gp_time_to_peak,gp_extrap_slope,gp_T_peak,gp_T_now,gp_dTdt_peak,gp_dTdt_now,gp_sigma_f,gp_peak_to_peak,gp_snr_max,gp_dfdt_max,gp_dfdt_min,gp_frac_of_peak,gp_post_var_mean,gp_post_var_max,gp_skewness,gp_kurtosis,gp_n_inflections,gp_amp,gp_lengthscale,gp_alpha\n");
        for p in &all_params {
            fn fmt(v: f64, prec: usize) -> String { if v.is_nan() { "NaN".to_string() } else { format!("{:.prec$}", v, prec = prec) } }
            csv_content.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                p.object, p.band, fmt(p.rise_time,3), fmt(p.decay_time,3), fmt(p.t0,3), fmt(p.peak_mag,3),
                fmt(p.chi2,3), fmt(p.baseline_chi2,1), p.n_obs, fmt(p.fwhm,3), fmt(p.rise_rate,6), fmt(p.decay_rate,6),
                fmt(p.gp_dfdt_now,6), fmt(p.gp_dfdt_next,6), fmt(p.gp_d2fdt2_now,6),
                fmt(p.gp_predicted_mag_1d,6), fmt(p.gp_predicted_mag_2d,6), fmt(p.gp_time_to_peak,6), fmt(p.gp_extrap_slope,6),
                fmt(p.gp_T_peak,1), fmt(p.gp_T_now,1), fmt(p.gp_dTdt_peak,3), fmt(p.gp_dTdt_now,3),
                fmt(p.gp_sigma_f,6), fmt(p.gp_peak_to_peak,6), fmt(p.gp_snr_max,3),
                fmt(p.gp_dfdt_max,6), fmt(p.gp_dfdt_min,6), fmt(p.gp_frac_of_peak,6),
                fmt(p.gp_post_var_mean,6), fmt(p.gp_post_var_max,6), fmt(p.gp_skewness,6), fmt(p.gp_kurtosis,6),
                fmt(p.gp_n_inflections,0), fmt(p.gp_amp,6), fmt(p.gp_lengthscale,6), fmt(p.gp_alpha,6),
            ));
        }
        fs::write(csv_path, csv_content)?;
    }

    println!("\n=== Throughput Analysis ===");
    println!("  Objects processed: {}/{}", n_success, n_targets);
    println!("  Bands fitted:     {}", all_params.len());
    println!("  Total wall time:   {:.2}s", total_elapsed);
    println!("  GP fit time (sum): {:.2}s", total_fit_time);
    if n_success > 0 {
        println!("  Objects/sec (wall): {:.1}", n_success as f64 / total_elapsed);
        println!("  Objects/sec (fit):  {:.1}", n_success as f64 / total_fit_time);
    }
    println!("  CSV: {}", csv_path);
    Ok(())
}
