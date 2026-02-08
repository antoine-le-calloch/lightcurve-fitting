use std::collections::HashMap;
use std::fs;

#[derive(Debug)]
pub struct BandData {
    pub times: Vec<f64>,
    pub mags: Vec<f64>,
    pub errors: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TimescaleParams {
    pub object: String,
    pub band: String,
    pub method: String,
    pub rise_time: f64,
    pub decay_time: f64,
    pub t0: f64,
    pub peak_mag: f64,
    pub chi2: f64,
    pub baseline_chi2: f64,
    pub n_obs: usize,
    pub fwhm: f64,
    pub rise_rate: f64,
    pub decay_rate: f64,
    pub gp_dfdt_now: f64,
    pub gp_dfdt_next: f64,
    pub gp_d2fdt2_now: f64,
    pub gp_predicted_mag_1d: f64,
    pub gp_predicted_mag_2d: f64,
    pub gp_time_to_peak: f64,
    pub gp_extrap_slope: f64,
    pub gp_T_peak: f64,
    pub gp_T_now: f64,
    pub gp_dTdt_peak: f64,
    pub gp_dTdt_now: f64,
    pub gp_sigma_f: f64,
    pub gp_peak_to_peak: f64,
    pub gp_snr_max: f64,
    pub gp_dfdt_max: f64,
    pub gp_dfdt_min: f64,
    pub gp_frac_of_peak: f64,
    pub gp_post_var_mean: f64,
    pub gp_post_var_max: f64,
    pub gp_skewness: f64,
    pub gp_kurtosis: f64,
    pub gp_n_inflections: f64,
}

pub fn read_ztf_lightcurve(path: &str, convert_to_mag: bool) -> Result<HashMap<String, BandData>, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let mut bands: HashMap<String, BandData> = HashMap::new();

    let mut mjd_min = f64::INFINITY;
    let lines_vec: Vec<_> = contents.lines().skip(1).collect();
    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            if let Ok(mjd) = parts[0].parse::<f64>() { mjd_min = mjd_min.min(mjd); }
        }
    }

    let mut epoch_best: HashMap<(String, u64), (f64, f64, f64)> = HashMap::new();
    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            let mjd: f64 = parts[0].parse()?;
            let flux: f64 = parts[1].parse()?;
            let flux_err: f64 = parts[2].parse()?;
            let filter = parts[3].trim().to_string();
            if flux_err > 0.0 && (flux > 0.0 || !convert_to_mag) {
                let mjd_rounded = (mjd * 1e6).round() as u64;
                let key = (filter, mjd_rounded);
                epoch_best.entry(key)
                    .and_modify(|entry| { if flux_err < entry.2 { *entry = (flux, mjd, flux_err); } })
                    .or_insert((flux, mjd, flux_err));
            }
        }
    }

    let mut all_times: Vec<f64> = epoch_best.values().map(|(_, mjd, _)| *mjd).collect();
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

/// Read an LSST-format lightcurve CSV file.
///
/// Supports two sub-formats (auto-detected from header):
///   1. night,band,expMidptMJD,mag,magerr,flux_corrected,flux_corrected_err,fluxlim,magLim
///   2. night,band,expMidptMJD,psfDiffFlux,psfDiffFluxErr,fluxlim,flux_corrected,flux_corrected_err
///
/// Uses flux_corrected / flux_corrected_err (nJy) as the primary data columns.
/// When convert_to_mag is true and format 1 has valid mag/magerr, those are used directly;
/// otherwise flux_corrected is converted with ZP = 31.4 (AB mag for nJy).
pub fn read_lsst_lightcurve(path: &str, convert_to_mag: bool) -> Result<HashMap<String, BandData>, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let mut lines_iter = contents.lines();

    // Parse header to find column indices
    let header = lines_iter.next().ok_or("Empty file")?;
    let cols: Vec<&str> = header.split(',').collect();
    let col_idx = |name: &str| cols.iter().position(|c| *c == name);

    let idx_mjd = col_idx("expMidptMJD").ok_or("Missing expMidptMJD column")?;
    let idx_band = col_idx("band").ok_or("Missing band column")?;
    let idx_flux = col_idx("flux_corrected");
    let idx_flux_err = col_idx("flux_corrected_err");
    let idx_mag = col_idx("mag");
    let idx_magerr = col_idx("magerr");

    let lines_vec: Vec<&str> = lines_iter.collect();

    // First pass: find minimum MJD
    let mut mjd_min = f64::INFINITY;
    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() > idx_mjd {
            if let Ok(mjd) = parts[idx_mjd].parse::<f64>() {
                mjd_min = mjd_min.min(mjd);
            }
        }
    }

    // Dedup: keep lowest-error observation per (band, MJD_rounded)
    // value stored: (flux_or_mag, mjd, error, band, has_direct_mag)
    let mut epoch_best: HashMap<(String, u64), (f64, f64, f64, bool)> = HashMap::new();

    for line in &lines_vec {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() <= idx_mjd || parts.len() <= idx_band {
            continue;
        }

        let mjd: f64 = match parts[idx_mjd].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let band = parts[idx_band].trim().to_string();

        // Try to get flux_corrected and flux_corrected_err
        let flux_val = idx_flux.and_then(|i| parts.get(i)).and_then(|s| s.parse::<f64>().ok());
        let flux_err_val = idx_flux_err.and_then(|i| parts.get(i)).and_then(|s| s.parse::<f64>().ok());

        // Try to get direct mag/magerr (format 1)
        let mag_val = idx_mag.and_then(|i| parts.get(i)).and_then(|s| s.parse::<f64>().ok());
        let magerr_val = idx_magerr.and_then(|i| parts.get(i)).and_then(|s| s.parse::<f64>().ok());

        // Decide which value to use
        let (value, error, has_direct_mag) = if convert_to_mag {
            // Prefer direct mag if available and valid
            if let (Some(mag), Some(magerr)) = (mag_val, magerr_val) {
                if magerr > 0.0 && mag.is_finite() {
                    (mag, magerr, true)
                } else if let (Some(flux), Some(ferr)) = (flux_val, flux_err_val) {
                    if flux > 0.0 && ferr > 0.0 {
                        let mag = -2.5 * flux.log10() + 31.4;
                        let mag_err = 1.0857 * (ferr / flux);
                        (mag, mag_err, false)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else if let (Some(flux), Some(ferr)) = (flux_val, flux_err_val) {
                if flux > 0.0 && ferr > 0.0 {
                    let mag = -2.5 * flux.log10() + 31.4;
                    let mag_err = 1.0857 * (ferr / flux);
                    (mag, mag_err, false)
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else {
            // Flux mode: use flux_corrected
            if let (Some(flux), Some(ferr)) = (flux_val, flux_err_val) {
                if ferr > 0.0 && (flux > 0.0 || !convert_to_mag) {
                    (flux, ferr, false)
                } else {
                    continue;
                }
            } else {
                continue;
            }
        };

        let mjd_rounded = (mjd * 1e6).round() as u64;
        let key = (band, mjd_rounded);
        epoch_best
            .entry(key)
            .and_modify(|entry| {
                if error < entry.2 {
                    *entry = (value, mjd, error, has_direct_mag);
                }
            })
            .or_insert((value, mjd, error, has_direct_mag));
    }

    // Bulk-start filter: discard early observations before median_time - 50 days
    let mut all_times: Vec<f64> = epoch_best.values().map(|(_, mjd, _, _)| *mjd).collect();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bulk_start = if !all_times.is_empty() {
        all_times[all_times.len() / 2] - 50.0
    } else {
        mjd_min
    };

    let mut bands: HashMap<String, BandData> = HashMap::new();
    for ((filter, _), (value, mjd, error, _)) in &epoch_best {
        if mjd < &bulk_start {
            continue;
        }
        let delta_t = mjd - mjd_min;
        let band = bands.entry(filter.clone()).or_insert_with(|| BandData {
            times: Vec::new(),
            mags: Vec::new(),
            errors: Vec::new(),
        });
        band.times.push(delta_t);
        band.mags.push(*value);
        band.errors.push(*error);
    }

    Ok(bands)
}

/// Auto-detect lightcurve format and read accordingly.
///
/// Checks the CSV header to determine whether the file is ZTF format
/// (mjd,flux,flux_err,filter) or LSST format (night,band,expMidptMJD,...).
pub fn read_lightcurve_auto(path: &str, convert_to_mag: bool) -> Result<HashMap<String, BandData>, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let header = contents.lines().next().unwrap_or("");

    if header.starts_with("night,band,") || header.contains("expMidptMJD") {
        drop(contents); // release before re-reading in sub-function
        read_lsst_lightcurve(path, convert_to_mag)
    } else {
        drop(contents);
        read_ztf_lightcurve(path, convert_to_mag)
    }
}

/// Detect the flux zero-point for a lightcurve file.
/// Returns 31.4 for LSST (nJy) files, 23.9 for ZTF (μJy) files.
pub fn detect_flux_zeropoint(path: &str) -> f64 {
    if let Ok(contents) = fs::read_to_string(path) {
        let header = contents.lines().next().unwrap_or("");
        if header.starts_with("night,band,") || header.contains("expMidptMJD") {
            return 31.4; // LSST nJy
        }
    }
    23.9 // ZTF μJy
}

pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() { return None; }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 { Some((values[mid - 1] + values[mid]) / 2.0) } else { Some(values[mid]) }
}

pub fn extract_rise_timescale(times: &[f64], mags: &[f64], peak_idx: usize) -> f64 {
    if peak_idx == 0 || peak_idx >= mags.len() { return f64::NAN; }
    let peak_mag = mags[peak_idx];
    let baseline = if peak_idx > 1 { mags[..peak_idx.min(3)].iter().sum::<f64>() / mags[..peak_idx.min(3)].len() as f64 } else { peak_mag + 0.5 };
    let target_amp = baseline + (peak_mag - baseline) * (1.0 - (-1.0_f64).exp());
    let mut closest_t_before = f64::NAN; let mut closest_diff = f64::INFINITY;
    for i in 0..peak_idx { let diff = (mags[i] - target_amp).abs(); if diff < closest_diff { closest_diff = diff; closest_t_before = times[i]; } }
    if closest_t_before.is_nan() { return f64::NAN; }
    times[peak_idx] - closest_t_before
}

pub fn extract_decay_timescale(times: &[f64], mags: &[f64], peak_idx: usize) -> f64 {
    if peak_idx >= mags.len() - 1 { return f64::NAN; }
    let peak_mag = mags[peak_idx];
    let baseline = if peak_idx < mags.len() - 1 {
        let end_idx = (peak_idx + mags.len()) / 2;
        let end_idx = end_idx.min(mags.len() - 1);
        let count = (end_idx - peak_idx).max(1);
        mags[peak_idx + 1..=end_idx].iter().sum::<f64>() / count as f64
    } else { peak_mag + 0.5 };
    let target_amp = baseline + (peak_mag - baseline) * (-1.0_f64).exp();
    let mut closest_t_after = f64::NAN; let mut closest_diff = f64::INFINITY;
    for i in (peak_idx + 1)..mags.len() { let diff = (mags[i] - target_amp).abs(); if diff < closest_diff { closest_diff = diff; closest_t_after = times[i]; } }
    if closest_t_after.is_nan() { return f64::NAN; }
    closest_t_after - times[peak_idx]
}

pub fn compute_fwhm(times: &[f64], mags: &[f64], peak_idx: usize) -> (f64, f64, f64) {
    if peak_idx >= mags.len() { return (f64::NAN, f64::NAN, f64::NAN); }
    let peak_mag = mags[peak_idx];
    let half_max_mag = peak_mag + 0.75;
    let mut t_before = f64::NAN;
    for i in (0..peak_idx).rev() { if mags[i] >= half_max_mag { t_before = times[i]; break; } }
    let mut t_after = f64::NAN;
    for i in (peak_idx + 1)..mags.len() { if mags[i] >= half_max_mag { t_after = times[i]; break; } }
    if t_before.is_nan() || t_after.is_nan() { return (f64::NAN, f64::NAN, f64::NAN); }
    (t_after - t_before, t_before, t_after)
}

pub fn compute_rise_rate(times: &[f64], mags: &[f64]) -> f64 {
    if times.len() < 2 { return f64::NAN; }
    let n_early = (times.len() as f64 * 0.25).ceil() as usize;
    let n_early = n_early.max(2).min(times.len() - 1);
    let early_times = &times[0..n_early]; let early_mags = &mags[0..n_early];
    let n = early_times.len() as f64; let sum_t = early_times.iter().sum::<f64>(); let sum_m = early_mags.iter().sum::<f64>();
    let sum_tt = early_times.iter().map(|t| t * t).sum::<f64>(); let sum_tm = early_times.iter().zip(early_mags.iter()).map(|(t,m)| t*m).sum::<f64>();
    let denominator = n * sum_tt - sum_t * sum_t; if denominator.abs() < 1e-10 { return f64::NAN; }
    (n * sum_tm - sum_t * sum_m) / denominator
}

pub fn compute_decay_rate(times: &[f64], mags: &[f64]) -> f64 {
    if times.len() < 2 { return f64::NAN; }
    let n_late = (times.len() as f64 * 0.25).ceil() as usize;
    let n_late = n_late.max(2).min(times.len());
    let start_idx = times.len() - n_late;
    let late_times = &times[start_idx..]; let late_mags = &mags[start_idx..];
    let n = late_times.len() as f64; let sum_t = late_times.iter().sum::<f64>(); let sum_m = late_mags.iter().sum::<f64>();
    let sum_tt = late_times.iter().map(|t| t * t).sum::<f64>(); let sum_tm = late_times.iter().zip(late_mags.iter()).map(|(t,m)| t*m).sum::<f64>();
    let denominator = n * sum_tt - sum_t * sum_t; if denominator.abs() < 1e-10 { return f64::NAN; }
    (n * sum_tm - sum_t * sum_m) / denominator
}
