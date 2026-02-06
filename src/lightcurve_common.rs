use std::collections::HashMap;
use std::fs;

#[derive(Debug)]
pub struct BandData {
    pub times: Vec<f32>,
    pub mags: Vec<f32>,
    pub errors: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TimescaleParams {
    pub object: String,
    pub band: String,
    pub method: String,
    pub rise_time: f32,
    pub decay_time: f32,
    pub t0: f32,
    pub peak_mag: f32,
    pub chi2: f32,
    pub baseline_chi2: f32,
    pub n_obs: usize,
    pub fwhm: f32,
    pub rise_rate: f32,
    pub decay_rate: f32,
    pub gp_dfdt_now: f32,
    pub gp_dfdt_next: f32,
    pub gp_d2fdt2_now: f32,
    pub gp_predicted_mag_1d: f32,
    pub gp_predicted_mag_2d: f32,
    pub gp_time_to_peak: f32,
    pub gp_extrap_slope: f32,
    pub gp_T_peak: f32,
    pub gp_T_now: f32,
    pub gp_dTdt_peak: f32,
    pub gp_dTdt_now: f32,
    pub gp_sigma_f: f32,
    pub gp_peak_to_peak: f32,
    pub gp_snr_max: f32,
    pub gp_dfdt_max: f32,
    pub gp_dfdt_min: f32,
    pub gp_frac_of_peak: f32,
    pub gp_post_var_mean: f32,
    pub gp_post_var_max: f32,
    pub gp_skewness: f32,
    pub gp_kurtosis: f32,
    pub gp_n_inflections: f32,
}

pub fn read_ztf_lightcurve(path: &str, convert_to_mag: bool) -> Result<HashMap<String, BandData>, Box<dyn std::error::Error + Send + Sync>> {
    let contents = fs::read_to_string(path)?;
    let mut bands: HashMap<String, BandData> = HashMap::new();

    let mut mjd_min = f32::INFINITY;
    let lines_vec: Vec<_> = contents.lines().skip(1).collect();
    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            if let Ok(mjd) = parts[0].parse::<f32>() { mjd_min = mjd_min.min(mjd); }
        }
    }

    let mut epoch_best: HashMap<(String, u64), (f32, f32, f32)> = HashMap::new();
    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            let mjd: f32 = parts[0].parse()?;
            let flux: f32 = parts[1].parse()?;
            let flux_err: f32 = parts[2].parse()?;
            let filter = parts[3].trim().to_string();
            if flux > 0.0 && flux_err > 0.0 {
                let mjd_rounded = (mjd * 1e6).round() as u64;
                let key = (filter, mjd_rounded);
                epoch_best.entry(key)
                    .and_modify(|entry| { if flux_err < entry.2 { *entry = (flux, mjd, flux_err); } })
                    .or_insert((flux, mjd, flux_err));
            }
        }
    }

    let mut all_times: Vec<f32> = epoch_best.values().map(|(_, mjd, _)| *mjd).collect();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bulk_start = if all_times.len() > 0 { all_times[all_times.len() / 2] - 50.0 } else { mjd_min };

    for ((filter, _), (flux, mjd, flux_err)) in epoch_best.iter() {
        if mjd < &bulk_start { continue; }
        let delta_t = mjd - mjd_min;
        
        let (value, error) = if convert_to_mag {
            // Convert flux to magnitude
            let log_flux10 = flux.log10();
            let log_sigma = flux_err / flux;
            let mag = -2.5 * log_flux10 + 23.9;
            let mag_err = 1.0857 * log_sigma;
            (mag, mag_err)
        } else {
            // Keep as flux
            (*flux, *flux_err)
        };
        
        let band = bands.entry(filter.clone()).or_insert_with(|| BandData { times: Vec::new(), mags: Vec::new(), errors: Vec::new() });
        band.times.push(delta_t);
        band.mags.push(value);
        band.errors.push(error);
    }

    Ok(bands)
}

pub fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() { return None; }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 { Some((values[mid - 1] + values[mid]) / 2.0) } else { Some(values[mid]) }
}

pub fn extract_rise_timescale(times: &[f32], mags: &[f32], peak_idx: usize) -> f32 {
    if peak_idx == 0 || peak_idx >= mags.len() { return f32::NAN; }
    let peak_mag = mags[peak_idx];
    let baseline = if peak_idx > 1 { mags[..peak_idx.min(3)].iter().sum::<f32>() / mags[..peak_idx.min(3)].len() as f32 } else { peak_mag + 0.5 };
    let target_amp = baseline + (peak_mag - baseline) * (1.0 - (-1.0_f32).exp());
    let mut closest_t_before = f32::NAN; let mut closest_diff = f32::INFINITY;
    for i in 0..peak_idx { let diff = (mags[i] - target_amp).abs(); if diff < closest_diff { closest_diff = diff; closest_t_before = times[i]; } }
    if closest_t_before.is_nan() { return f32::NAN; }
    times[peak_idx] - closest_t_before
}

pub fn extract_decay_timescale(times: &[f32], mags: &[f32], peak_idx: usize) -> f32 {
    if peak_idx >= mags.len() - 1 { return f32::NAN; }
    let peak_mag = mags[peak_idx];
    let baseline = if peak_idx < mags.len() - 1 {
        let end_idx = (peak_idx + mags.len()) / 2;
        let end_idx = end_idx.min(mags.len() - 1);
        let count = (end_idx - peak_idx).max(1);
        mags[peak_idx + 1..=end_idx].iter().sum::<f32>() / count as f32
    } else { peak_mag + 0.5 };
    let target_amp = baseline + (peak_mag - baseline) * (-1.0_f32).exp();
    let mut closest_t_after = f32::NAN; let mut closest_diff = f32::INFINITY;
    for i in (peak_idx + 1)..mags.len() { let diff = (mags[i] - target_amp).abs(); if diff < closest_diff { closest_diff = diff; closest_t_after = times[i]; } }
    if closest_t_after.is_nan() { return f32::NAN; }
    closest_t_after - times[peak_idx]
}

pub fn compute_fwhm(times: &[f32], mags: &[f32], peak_idx: usize) -> (f32, f32, f32) {
    if peak_idx >= mags.len() { return (f32::NAN, f32::NAN, f32::NAN); }
    let peak_mag = mags[peak_idx];
    let half_max_mag = peak_mag + 0.75;
    let mut t_before = f32::NAN;
    for i in (0..peak_idx).rev() { if mags[i] >= half_max_mag { t_before = times[i]; break; } }
    let mut t_after = f32::NAN;
    for i in (peak_idx + 1)..mags.len() { if mags[i] >= half_max_mag { t_after = times[i]; break; } }
    if t_before.is_nan() || t_after.is_nan() { return (f32::NAN, f32::NAN, f32::NAN); }
    (t_after - t_before, t_before, t_after)
}

pub fn compute_rise_rate(times: &[f32], mags: &[f32]) -> f32 {
    if times.len() < 2 { return f32::NAN; }
    let n_early = (times.len() as f32 * 0.25).ceil() as usize;
    let n_early = n_early.max(2).min(times.len() - 1);
    let early_times = &times[0..n_early]; let early_mags = &mags[0..n_early];
    let n = early_times.len() as f32; let sum_t = early_times.iter().sum::<f32>(); let sum_m = early_mags.iter().sum::<f32>();
    let sum_tt = early_times.iter().map(|t| t * t).sum::<f32>(); let sum_tm = early_times.iter().zip(early_mags.iter()).map(|(t,m)| t*m).sum::<f32>();
    let denominator = n * sum_tt - sum_t * sum_t; if denominator.abs() < 1e-10 { return f32::NAN; }
    (n * sum_tm - sum_t * sum_m) / denominator
}

pub fn compute_decay_rate(times: &[f32], mags: &[f32]) -> f32 {
    if times.len() < 2 { return f32::NAN; }
    let n_late = (times.len() as f32 * 0.25).ceil() as usize;
    let n_late = n_late.max(2).min(times.len());
    let start_idx = times.len() - n_late;
    let late_times = &times[start_idx..]; let late_mags = &mags[start_idx..];
    let n = late_times.len() as f32; let sum_t = late_times.iter().sum::<f32>(); let sum_m = late_mags.iter().sum::<f32>();
    let sum_tt = late_times.iter().map(|t| t * t).sum::<f32>(); let sum_tm = late_times.iter().zip(late_mags.iter()).map(|(t,m)| t*m).sum::<f32>();
    let denominator = n * sum_tt - sum_t * sum_t; if denominator.abs() < 1e-10 { return f32::NAN; }
    (n * sum_tm - sum_t * sum_m) / denominator
}
