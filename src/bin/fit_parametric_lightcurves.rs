use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;
use rayon::prelude::*;

use argmin::core::{CostFunction, Error as ArgminError, Executor, State};
use argmin::solver::particleswarm::ParticleSwarm;
use plotters::prelude::*;

use lightcurve_fiting::lightcurve_common::read_ztf_lightcurve;
use sklears_core::plugin::testing_prelude;

// Zeropoint consistent with GP plotter
const ZP: f64 = 23.9;

#[derive(Clone, Copy, Debug, PartialEq)]
enum ModelVariant {
    Full,
    DecayOnly,
    FastDecay,
    PowerLaw,
    Bazin,
}

#[derive(Clone, Debug)]
struct VillarTimescaleParams {
    band: String,
    variant: String,
    rise_time: f64,      // tau_rise for Full, NaN for PowerLaw
    decay_time: f64,     // tau_fall
    peak_time: f64,      // t0
    peak_mag: f64,       // Peak magnitude (brightest, minimum value)
    chi2: f64,
    n_obs: usize,
    fwhm: f64,           // Full Width at Half Maximum (days)
    rise_rate: f64,      // Rise rate (mag/day)
    decay_rate: f64,     // Decay rate (mag/day)
    // Power law parameters (NaN for Villar models)
    powerlaw_amplitude: f64,  // a in power law: a * (t - t0)^(-alpha)
    powerlaw_index: f64,      // alpha in power law
}

#[derive(Clone)]
struct BandFitData<'a> {
    times: &'a [f64],
    flux: Vec<f64>,
    flux_err: Vec<f64>,
    noise_frac_median: f64,
    peak_flux_obs: f64,
    // Precompute weights outside of cost loop
    weights: Vec<f64>
}

#[derive(Clone)]
struct SingleBandVillarCost <'a>{
    band: &'a BandFitData <'a>,
    variant: ModelVariant,
}

impl <'a> CostFunction for SingleBandVillarCost <'a>{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        match self.variant {
            ModelVariant::PowerLaw => {
                let a = p[0].exp();
                let sigma_extra = p[3].exp();

                if !a.is_finite() || !sigma_extra.is_finite() {
                    return Ok(1e99);
                }

                let alpha = p[1];
                let t0 = p[2];
                let sigma_penalty = (sigma_extra / 0.1).powi(2);

                // move this up here under the assumption total_chi2 << 1e6
                if t0 < -100.0 || t0 > 50.0 || alpha < 0.0 || alpha > 5.0 {
                    return Ok(1e6 + sigma_penalty);
                }

                let mut total_chi2 = 0.0;
                for ((&time, &flux), &weight) in self.band.times.iter().zip(&self.band.flux).zip(&self.band.weights) {
                    let model = powerlaw_flux(a, alpha, t0, time);
                    let diff = model - flux;
                    // Use only observational errors in chi2, not sigma_extra
                    total_chi2 += diff * diff * weight;
                    
                }

                let n = self.band.times.len().max(1) as f64;
                // Add penalty for large sigma_extra to prevent overfitting
                Ok(total_chi2 / n + sigma_penalty)
            }

            ModelVariant::Bazin => {
                // Bazin: [log_a, b, t0, log_tau_rise, log_tau_fall]
                let a = p[0].exp();
                let inv_tau_rise: f64 =1.0 / p[3].exp();
                let inv_tau_fall: f64 =1.0 / p[4].exp();

                if !a.is_finite() || !inv_tau_rise.is_finite() || !inv_tau_fall.is_finite() {
                    return Ok(1e99);
                }

                let b = p[1];
                let t0 = p[2];

                if t0 < -100.0 || t0 > 100.0 || inv_tau_rise > 1e6 || inv_tau_rise < 1e-4 || inv_tau_fall > 1e6 || inv_tau_fall < 1e-4 {
                    return Ok(1e6)
                }

                let mut total_chi2 = 0.0;
                for ((&time, &flux), &weight) in self.band.times.iter().zip(&self.band.flux).zip(&self.band.weights) {
                    let model = bazin_flux(a, b, t0, inv_tau_rise, inv_tau_fall, time);
                    let diff = model - flux;
                    total_chi2 += diff * diff * weight;
                }

                let n = self.band.times.len().max(1) as f64;
                Ok(total_chi2 / n)
            }
            _ => {
                let a = p[0].exp();
                let gamma = p[2].exp();
                let inv_tau_rise: f64 =1.0 / p[4].exp();
                let inv_tau_fall: f64 =1.0 / p[5].exp();
                let sigma_extra = p[6].exp();

                if !a.is_finite() || !gamma.is_finite() || !inv_tau_rise.is_finite() || !inv_tau_fall.is_finite() || !sigma_extra.is_finite() {
                    return Ok(1e99);
                }

                let t0 = p[3];

                if t0 < -100.0 || t0 > 100.0 || inv_tau_rise > 1e6 || inv_tau_rise < 1e-4 || inv_tau_fall > 1e6 || inv_tau_fall < 1e-4 {
                    return Ok(1e6)
                }

                let beta = p[1];
                let mut total_chi2 = 0.0;
                for ((&time, &flux), &weight) in self.band.times.iter().zip(&self.band.flux).zip(&self.band.weights) {
                    let model = match self.variant {
                        ModelVariant::Full => villar_flux(a, beta, gamma, t0, inv_tau_rise, inv_tau_fall, time),
                        ModelVariant::DecayOnly | ModelVariant::FastDecay => villar_flux_decay(a, beta, gamma, t0, inv_tau_fall, time),
                        ModelVariant::PowerLaw | ModelVariant::Bazin => unreachable!(),
                    };
                    let diff = model - flux;
                    total_chi2 += diff * diff * weight;
                }

                // Add penalty for large sigma_extra to prevent overfitting
                let sigma_penalty = (sigma_extra / 0.1).powi(2);
                let n = self.band.times.len().max(1) as f64;
                Ok(total_chi2 / n + sigma_penalty)
            }
        }
    }
}

fn pso_bounds(base: Option<&[f64]>, variant: ModelVariant) -> (Vec<f64>, Vec<f64>) {
    if matches!(variant, ModelVariant::Bazin) {
        // Bazin model: [a, b (baseline), t0, tau_rise, tau_fall]
        let lower = vec![-0.3, -1.0, -100.0, 1e-8, 1e-8];
        let upper = vec![0.5, 1.0, 30.0, 3.5, 3.5];
        return (lower, upper);
    }
    if matches!(variant, ModelVariant::PowerLaw) {
        let mut lower = vec![-0.5, 0.5, -10.0, -4.0];
        let mut upper = vec![0.8, 4.0, 30.0, -1.5];
        if let Some(b) = base {
            let span = vec![0.7, 0.5, 5.0, 2.0];
            for i in 0..b.len() {
                lower[i] = (b[i] - span[i]).max(lower[i]);
                upper[i] = (b[i] + span[i]).min(upper[i]);
            }
        }
        return (lower, upper);
    }

    let mut lower = vec![-0.5, 0.0, -8.0, -5.0, -5.0, -5.0, -4.0];
    let mut upper = vec![0.8, 0.08, 4.0, 100.0, 6.5, 7.5, -1.5];  // Increased tau_fall max from 6.5 to 7.5 (~1800 days)
    if matches!(variant, ModelVariant::FastDecay) {
        upper[1] = 0.02;
        upper[2] = 2.0;
        upper[5] = 5.0;  // Increased from 3.0 to 5.0 (~150 days for fast decay)
        lower[5] = -4.0;
    }
    if let Some(b) = base {
        let span = vec![0.7, 0.01, 1.0, 10.0, 1.0, 1.0, 2.0];
        for i in 0..b.len() {
            lower[i] = (b[i] - span[i]).max(lower[i]);
            upper[i] = (b[i] + span[i]).min(upper[i]);
        }
    }
    (lower, upper)
}

#[inline]
pub fn villar_flux(a: f64, beta: f64, gamma: f64, t0: f64, inv_tau_rise: f64, inv_tau_fall: f64, t: f64) -> f64 {
    let phase = t - t0;
    let sigmoid = 1.0 / (1.0 + (-phase * inv_tau_rise).exp());
    let piece = if phase < gamma {
        1.0 - beta * phase
    } else {
        (1.0 - beta * gamma) * ((gamma - phase) * inv_tau_fall).exp()
    };
    a * sigmoid * piece
}

#[inline]
pub fn villar_flux_decay(a: f64, beta: f64, gamma: f64, t0: f64, inv_tau_fall: f64, t: f64) -> f64 {
    let phase = t - t0;
    let piece = if phase < gamma {
        1.0 - beta * phase
    } else {
        (1.0 - beta * gamma) * ((gamma - phase) * inv_tau_fall).exp()
    };
    a * piece
}

#[inline]
pub fn powerlaw_flux(a: f64, alpha: f64, t0: f64, t: f64) -> f64 {
    let phase = t - t0;
    // Power law should apply for all observed times (phase > 0)
    // If phase <= 0, return a very large flux (invisible/not observed)
    if phase <= 0.0 {
        f64::INFINITY  // Model predicts no flux before t0 (explosion time)
    } else {
        a * phase.powf(-alpha)
    }
}

#[inline]
pub fn bazin_flux(a: f64, b: f64, t0: f64, inv_tau_rise: f64, inv_tau_fall: f64, t: f64) -> f64 {
    let dt = t - t0;
    let num = (-(dt) * inv_tau_fall).exp();
    let den = 1.0 + (-(dt) * inv_tau_rise).exp();
    a * (num / den) + b
}

#[inline]
fn flux_to_mag(flux: f64) -> f64 {
    -2.5 * flux.log10() + ZP
}

fn median(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() { return None; }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        Some((xs[mid - 1] + xs[mid]) / 2.0)
    } else {
        Some(xs[mid])
    }
}

fn get_band_color(band: &str) -> RGBColor {
    match band {
        "g" | "ZTF_g" => BLUE,
        "r" | "ZTF_r" => RED,
        "i" | "ZTF_i" => GREEN,
        _ => BLACK,
    }
}

// Compute Full Width at Half Maximum (FWHM) in magnitude space
// Finds the time span where magnitude is within 0.75 mag of peak (50% flux)
fn compute_fwhm(times: &[f64], mags: &[f64]) -> (f64, f64, f64) {
    if times.is_empty() || mags.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    
    // Find peak (minimum magnitude)
    let peak_mag = mags.iter().copied().fold(f64::INFINITY, f64::min);
    let half_max_mag = peak_mag + 0.75;  // 0.75 mag fainter = 50% flux
    
    // Find time before peak where mag crosses half maximum (going from faint to bright)
    let mut t_before = f64::NAN;
    for (t, m) in times.iter().zip(mags.iter()) {
        if m >= &half_max_mag {  // Found where it's fainter than half-max
            t_before = *t;
            break;
        }
    }
    
    // Find time after peak where mag crosses half maximum (going from bright to faint)
    let mut t_after = f64::NAN;
    for (t, m) in times.iter().zip(mags.iter()).rev() {
        if m >= &half_max_mag {  // Found where it's fainter than half-max
            t_after = *t;
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

// Adapter function to convert BandData to the tuple format used by parametric fitting
// Parametric fitting needs flux values (not magnitudes), so we pass convert_to_mag=false
fn read_lightcurve(path: &str) -> Result<HashMap<String, (Vec<f64>, Vec<f64>, Vec<f64>)>, Box<dyn std::error::Error + Send + Sync>> {
    let bands = read_ztf_lightcurve(path, false)?;
    let result = bands
        .into_iter()
        .map(|(filter, bd)| (filter, (bd.times, bd.mags, bd.errors)))
        .collect();
    Ok(result)
}
struct BandPlot <'a> {
    times_obs: &'a [f64],
    mags_obs: &'a [f64],
    mag_errors: Vec<f64>,
    times_pred: &'a [f64],
    mags_model: Vec<f64>,
    mags_upper: Vec<f64>,
    mags_lower: Vec<f64>,
    label: &'a String,
    chi2: f64,
    legend_label: String,
}


struct RefFit {
    params: Vec<f64>,
    variant: ModelVariant,
}

fn fit_band(data: &BandFitData, times_pred: &[f64], ref_fit: Option<&RefFit>, force_variant: Option<ModelVariant>) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, String, String, Vec<f64>, VillarTimescaleParams) {
    let run_fit = |base: Option<&[f64]>, variant: ModelVariant, iters: u64, particles: usize| {
        let (lower, upper) = pso_bounds(base, variant);
        let solver = ParticleSwarm::new((lower, upper), particles);
        let problem = SingleBandVillarCost { band: data, variant };
        let res = Executor::new(problem, solver)
            .configure(|state| state.max_iters(iters))
            .run()
            .expect("PSO failed");
        let state = res.state();
        let best = state.get_best_param().unwrap();
        let chi2 = state.get_cost();
        (best.position.clone(), chi2)
    };

    // If sparse band (< 50 points) and reference fit available, constrain to reference variant
    let is_sparse = data.times.len() < 50;
    let use_ref_variant = is_sparse && ref_fit.is_some();

    // If force_variant is specified, only try that variant (for non-reference bands)
    let try_variants = if let Some(forced) = force_variant {
        vec![forced]
    } else {
        vec![ModelVariant::Full, ModelVariant::FastDecay, ModelVariant::PowerLaw, ModelVariant::Bazin]
    };

    let (params_full, chi2_full) = if try_variants.contains(&ModelVariant::Full) {
        let base = if use_ref_variant { Some(ref_fit.unwrap().params.as_slice()) } else { None };
        run_fit(base, ModelVariant::Full, 100, 60)
    } else {
        (vec![], f64::INFINITY)
    };
    
    let (params_fast, chi2_fast) = if try_variants.contains(&ModelVariant::FastDecay) {
        let base = if use_ref_variant { Some(ref_fit.unwrap().params.as_slice()) } else { None };
        run_fit(base, ModelVariant::FastDecay, 100, 60)
    } else {
        (vec![], f64::INFINITY)
    };
    
    let (params_power, chi2_power) = if try_variants.contains(&ModelVariant::PowerLaw) {
        let base = if use_ref_variant { Some(ref_fit.unwrap().params.as_slice()) } else { None };
        run_fit(base, ModelVariant::PowerLaw, 80, 50)
    } else {
        (vec![], f64::INFINITY)
    };

    let (params_bazin, chi2_bazin) = if try_variants.contains(&ModelVariant::Bazin) {
        let base = if use_ref_variant { Some(ref_fit.unwrap().params.as_slice()) } else { None };
        run_fit(base, ModelVariant::Bazin, 100, 60)
    } else {
        (vec![], f64::INFINITY)
    };

    let (params, variant, chi2_best) = {
        let mut best = (params_full, ModelVariant::Full, chi2_full);
        if chi2_fast < best.2 { best = (params_fast, ModelVariant::FastDecay, chi2_fast); }
        if chi2_power < best.2 { best = (params_power, ModelVariant::PowerLaw, chi2_power); }
        if chi2_bazin < best.2 { best = (params_bazin, ModelVariant::Bazin, chi2_bazin); }
        best
    };

    // If fitting completely failed, return NaN values
    if params.is_empty() {
        let nan_vec = vec![f64::NAN; times_pred.len()];
        let failed_params = VillarTimescaleParams {
            band: String::from("unknown"),
            variant: String::from("NoFit"),
            rise_time: f64::NAN,
            decay_time: f64::NAN,
            peak_time: f64::NAN,
            peak_mag: f64::NAN,
            chi2: chi2_best,
            n_obs: data.times.len(),
            fwhm: f64::NAN,
            rise_rate: f64::NAN,
            decay_rate: f64::NAN,
            powerlaw_amplitude: f64::NAN,
            powerlaw_index: f64::NAN,
        };
        return (nan_vec.clone(), nan_vec.clone(), nan_vec, chi2_best, String::from("NoFit"), String::from("chi2=NaN"), vec![f64::NAN; 7], failed_params);
    }

    let sigma_extra = match variant {
        ModelVariant::PowerLaw => params[3].exp(),
        ModelVariant::Bazin => 0.0,  // Bazin has no sigma_extra parameter
        _ => params[6].exp(),
    };

    let param_summary = match variant {
        ModelVariant::Full => {
            let beta = params[1];
            let gamma = params[2].exp();
            let t0 = params[3];
            let tau_rise = params[4].exp();
            let tau_fall = params[5].exp();
            format!("Full t0={:.2}, tr={:.2}, tf={:.2}, beta={:.3}, gam={:.2}", t0, tau_rise, tau_fall, beta, gamma)
        }
        ModelVariant::DecayOnly | ModelVariant::FastDecay => {
            let beta = params[1];
            let gamma = params[2].exp();
            let t0 = params[3];
            let tau_fall = params[5].exp();
            format!("Fast t0={:.2}, tf={:.2}, beta={:.3}, gam={:.2}", t0, tau_fall, beta, gamma)
        }
        ModelVariant::PowerLaw => {
            let alpha = params[1];
            let t0 = params[2];
            format!("PL t0={:.2}, alpha={:.3}", t0, alpha)
        }
        ModelVariant::Bazin => {
            let b = params[1];
            let t0 = params[2];
            let tau_rise = params[3].exp();
            let tau_fall = params[4].exp();
            format!("Bazin t0={:.2}, tr={:.2}, tf={:.2}, b={:.3}", t0, tau_rise, tau_fall, b)
        }
    };

    let (eval_model, flux_model):(Box<dyn Fn(f64) -> f64>, Vec<f64>) = match variant {
        ModelVariant::Full => {
            let a = params[0].exp();
            let beta = params[1];
            let gamma = params[2].exp();
            let t0 = params[3];
            let inv_tau_rise: f64 =1.0 / params[4].exp();
            let inv_tau_fall: f64 =1.0 / params[5].exp();
            let eval = move |t: f64| villar_flux(a, beta, gamma, t0, inv_tau_rise, inv_tau_fall, t);
            let flux = times_pred.iter().map(|&t| eval(t)).collect();
            (Box::new(eval), flux)
        }
        ModelVariant::DecayOnly | ModelVariant::FastDecay => {
            let a = params[0].exp();
            let beta = params[1];
            let gamma = params[2].exp();
            let t0 = params[3];
            let inv_tau_fall: f64 =1.0 / params[5].exp();
            let eval = move |t: f64| villar_flux_decay(a, beta, gamma, t0, inv_tau_fall, t);
            let flux = times_pred.iter().map(|&t| eval(t)).collect();
            (Box::new(eval), flux)
        }
        ModelVariant::PowerLaw => {
            let a = params[0].exp();
            let alpha = params[1];
            let t0 = params[2];
            let eval = move |t: f64| powerlaw_flux(a, alpha, t0, t);
            let flux = times_pred.iter().map(|&t| eval(t)).collect();
            (Box::new(eval), flux)
        }
        ModelVariant::Bazin => {
            let a = params[0].exp();
            let b = params[1];
            let t0 = params[2];
            let inv_tau_rise: f64 =1.0 / params[3].exp();
            let inv_tau_fall: f64 =1.0 / params[4].exp();
            let eval = move |t: f64| bazin_flux(a, b, t0, inv_tau_rise, inv_tau_fall, t);
            let flux = times_pred.iter().map(|&t| eval(t)).collect();
            (Box::new(eval), flux)
        }
    };

    // Weighted scale fit in normalized space to avoid forcing model to the observed peak
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..data.times.len() {
        let m = eval_model(data.times[i]);
        let y = data.flux[i];
        let var = data.flux_err[i] * data.flux_err[i] + sigma_extra * sigma_extra + 1e-10;
        let w = 1.0 / var;
        num += m * y * w;
        den += m * m * w;
    }
    let scale_norm = if den > 0.0 { num / den } else { 1.0 };
    let flux_scale = scale_norm * data.peak_flux_obs;

    let vec_length = flux_model.len();
    let mut mags = Vec::with_capacity(vec_length);
    let mut mags_upper = Vec::with_capacity(vec_length);
    let mut mags_lower = Vec::with_capacity(vec_length);

    let frac_sigma = (sigma_extra).hypot(data.noise_frac_median);
    let sigma_mag = 1.0857 * frac_sigma;
    let sigma_mag_clamped = sigma_mag.min(0.35);
    for f in &flux_model {
        let f_scaled = f * flux_scale; // back to observed flux units
            let m = flux_to_mag(f_scaled.max(1e-12));
            // sigma_extra is in normalized flux units (fraction of peak); combine as fractional scatter
        mags.push(m);
        mags_upper.push(m + sigma_mag_clamped);
        mags_lower.push(m - sigma_mag_clamped);
    }

    // Extract timescale parameters
    let (rise_time, decay_time, peak_time) = match variant {
        ModelVariant::Full => {
            let tau_rise = params[4].exp();
            let tau_fall = params[5].exp();
            let t0 = params[3];
            (tau_rise, tau_fall, t0)
        }
        ModelVariant::DecayOnly | ModelVariant::FastDecay => {
            let tau_fall = params[5].exp();
            let t0 = params[3];
            (f64::NAN, tau_fall, t0)  // No rise time for decay-only models
        }
        ModelVariant::PowerLaw => {
            let t0 = params[2];
            (f64::NAN, f64::NAN, t0)  // No rise/decay times for power-law
        }
        ModelVariant::Bazin => {
            let tau_rise = params[3].exp();
            let tau_fall = params[4].exp();
            let t0 = params[2];
            (tau_rise, tau_fall, t0)
        }
    };
    
    // Extract power law parameters
    let (powerlaw_amplitude, powerlaw_index) = match variant {
        ModelVariant::PowerLaw => {
            let a = params[0].exp();
            let alpha = params[1];
            (a, alpha)
        }
        _ => (f64::NAN, f64::NAN),  // NaN for non-power-law models
    };
    
    // Compute complementary metrics: FWHM and rise/decay rates
    let peak_mag = mags.iter().cloned().fold(f64::INFINITY, f64::min);
    let (fwhm_calc, t_before, t_after) = compute_fwhm(times_pred, &mags);
    let fwhm = if !t_before.is_nan() && !t_after.is_nan() {
        t_after - t_before  // Use actual crossing boundaries
    } else {
        fwhm_calc  // Fallback to calculated value if no crossings found
    };
    let rise_rate = compute_rise_rate(times_pred, &mags);
    let decay_rate = compute_decay_rate(times_pred, &mags);
    
    let variant_str_for_params = format!("{:?}", variant);
    let timescale_params = VillarTimescaleParams {
        band: String::new(),  // Will be set by caller
        variant: variant_str_for_params.clone(),
        rise_time,
        decay_time,
        peak_time,
        peak_mag,
        chi2: chi2_best,
        n_obs: data.times.len(),
        fwhm,
        rise_rate,
        decay_rate,
        powerlaw_amplitude,
        powerlaw_index,
    };

    (mags, mags_upper, mags_lower, chi2_best, format!("{:?}", variant), param_summary, params, timescale_params)
}

fn process_file(input_path: &str, output_dir: &Path) -> Result<(f64, Vec<VillarTimescaleParams>), Box<dyn std::error::Error + Send + Sync>> {
    let object_name = input_path
        .split('/')
        .last()
        .unwrap_or("unknown")
        .trim_end_matches(".csv");

    let bands_raw = read_lightcurve(input_path)?;
    if bands_raw.is_empty() {
        eprintln!("No valid data in {}", input_path);
        return Ok((0.0, Vec::new()));
    }

    // Prepare per-band data
    let data_length = bands_raw.len();
    let mut band_plots: Vec<BandPlot> = Vec::with_capacity(data_length);

    // Determine time grid across bands
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for (t, _, _) in bands_raw.values() {
        for &x in t {
            t_min = t_min.min(x);
            t_max = t_max.max(x);
        }
    }
    let duration = t_max - t_min;
    let n_pred = 200;
    let times_pred: Vec<f64> = (0..n_pred)
        .map(|i| t_min + (i as f64) * duration / (n_pred - 1) as f64)
        .collect();

    // Build fit data for all bands first
    let mut band_data: Vec<(String, BandFitData, Vec<f64>)> = Vec::with_capacity(data_length);
    for (band_name, (times, fluxes, flux_errs)) in bands_raw.iter() {
        if fluxes.is_empty() {
            continue;
        }
        let peak_flux = fluxes.iter().copied().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 {
            continue;
        }
        let inv_peak_flux = 1.0 / peak_flux;
        let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f * inv_peak_flux).collect();
        let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e * inv_peak_flux).collect();
        let mut frac_noises = Vec::with_capacity(data_length);
        frac_noises.extend(
            fluxes.iter().zip(flux_errs)
                .filter_map(|(&f, &e)| if f > 0.0 { Some(e / f) } else { None })
        );

        let noise_frac_median = median(&mut frac_noises).unwrap_or(0.0);
        let mut mags_obs = Vec::with_capacity(fluxes.len());
        for f in fluxes {
            let m = flux_to_mag(*f);
            mags_obs.push(m);
        }

        // Precompute weights outside of cost loop
        let inverse_var_calculation: Vec<f64> = normalized_err.iter().map(|e| 1.0 / (e * e + 1e-10)).collect();
        let fit_data = BandFitData {
            times: times,
            flux: normalized_flux,
            flux_err: normalized_err,
            noise_frac_median,
            peak_flux_obs: peak_flux,
            weights: inverse_var_calculation
        };

        band_data.push((band_name.clone(), fit_data, mags_obs));
    }

    // Sort by number of points descending; fit only the band with most points for timescales
    band_data.sort_by(|a, b| b.1.times.len().cmp(&a.1.times.len()));

    let mut ref_fit: Option<RefFit> = None;
    let mut total_fit_time = 0.0;
    let mut timescale_params_all: Vec<VillarTimescaleParams> = Vec::new();
    
    // Fit all bands but only save timescale params for the first (most observations)
    for (i, (band_name, fit_data, mags_obs)) in band_data.iter().enumerate() {
        let fit_start = Instant::now();
        
        // Only the first (reference) band tries all variants; others use the reference variant
        let force_variant = if i == 0 { None } else { ref_fit.as_ref().map(|rf| rf.variant) };
        
        let (mags_model, mags_upper, mags_lower, chi2, variant_str, param_summary, params, mut timescale_params) = fit_band(fit_data, &times_pred, ref_fit.as_ref(), force_variant);
        let fit_elapsed = fit_start.elapsed().as_secs_f64();
        total_fit_time += fit_elapsed;

        // Store reference fit from first (densest) band BEFORE processing other bands
        // Always set reference from first band, regardless of point count
        if i == 0 {
            let variant = if variant_str.contains("Full") {
                ModelVariant::Full
            } else if variant_str.contains("PowerLaw") {
                ModelVariant::PowerLaw
            } else if variant_str.contains("Bazin") {
                ModelVariant::Bazin
            } else {
                ModelVariant::FastDecay
            };
            ref_fit = Some(RefFit {
                params: params.clone(),
                variant,
            });
        }

        // Save timescale params for all bands (they all use same variant after first)
        timescale_params.band = band_name.clone();
        timescale_params_all.push(timescale_params.clone());

        let legend_label = format!("{} ({}; chi2={:.2}; N={})", band_name, param_summary, chi2, fit_data.times.len());

        eprintln!("  {} fit: chi2={:.3}, N={}", band_name, chi2, fit_data.times.len());

        let mag_errors: Vec<f64> = fit_data.flux_err.iter()
            .zip(fit_data.flux.iter())
            .map(|(err, flux)| if *flux > 0.0 { 1.0857 * err / flux } else { 0.1 })
            .collect();

        band_plots.push(BandPlot {
            times_obs: &fit_data.times,
            mags_obs: &mags_obs,
            mag_errors,
            times_pred: &times_pred,
            mags_model,
            mags_upper,
            mags_lower,
            label: &band_name,
            chi2,
            legend_label,
        });
    }

    if band_plots.is_empty() {
        eprintln!("No bands could be fit in {}", input_path);
        return Ok((0.0, timescale_params_all));
    }

    // Determine mag range
    // let mut mag_min = f64::INFINITY;
    // let mut mag_max = f64::NEG_INFINITY;
    // for b in &band_plots {
    //     for &m in b.mags_obs.iter().chain(b.mags_model.iter()) {
    //         mag_min = mag_min.min(m);
    //         mag_max = mag_max.max(m);
    //     }
    // }
    // let mag_pad = (mag_max - mag_min) * 0.15;
    // let y_top = (mag_max + mag_pad).min(25.0);
    // let y_bottom = (mag_min - mag_pad).max(15.0);

    // let output_path = output_dir.join(format!("{}.png", object_name));
    // let root = BitMapBackend::new(&output_path, (1600, 900)).into_drawing_area();
    // root.fill(&WHITE)?;
    // let mut chart = ChartBuilder::on(&root)
    //     .margin(12)
    //     .x_label_area_size(70)
    //     .y_label_area_size(90)
    //     .build_cartesian_2d(t_min..t_max, y_top..y_bottom)?;;

    // chart.configure_mesh()
    //     .x_desc("Time (days)")
    //     .y_desc("Flux")
    //     .x_label_style(("sans-serif", 24))
    //     .y_label_style(("sans-serif", 24))
    //     .draw()?;

    // // Draw timescale markers (if available)
    // if !timescale_params_all.is_empty() {
    //     let params = &timescale_params_all[0];  // Use the first (only) fitted band
    //     let t0 = params.peak_time;
        
    //     // FWHM shaded region - draw first so it's behind the t0 line
    //     let peak_mag = params.peak_mag;
    //     // Try to draw FWHM region from fitted model curve regardless of stored FWHM value
    //     if !band_plots.is_empty() {
    //         let first_band = &band_plots[0];
    //         let half_max_mag = peak_mag + 0.75;  // 0.75 mag fainter = 50% flux
            
    //         // Find time bounds for FWHM by scanning the fitted model curve
    //         let mut t_before = f64::NAN;
    //         let mut t_after = f64::NAN;
            
    //         // Find peak index in fitted magnitudes
    //         let mut peak_idx = 0;
    //         let mut min_mag = f64::INFINITY;
    //         for (i, &mag) in first_band.mags_model.iter().enumerate() {
    //             if mag < min_mag {
    //                 min_mag = mag;
    //                 peak_idx = i;
    //             }
    //         }
            
    //         // Find time before peak where mag crosses half maximum
    //         for i in (0..peak_idx).rev() {
    //             if first_band.mags_model[i] >= half_max_mag {
    //                 t_before = first_band.times_pred[i];
    //                 break;
    //             }
    //         }
            
    //         // Find time after peak where mag crosses half maximum
    //         for i in (peak_idx + 1)..first_band.mags_model.len() {
    //             if first_band.mags_model[i] >= half_max_mag {
    //                 t_after = first_band.times_pred[i];
    //                 break;
    //             }
    //         }
            
    //         // Draw shaded region if both bounds are valid and within plot range
    //         if !t_before.is_nan() && !t_after.is_nan() && 
    //            t_before >= t_min && t_after <= t_max {
    //             chart.draw_series(std::iter::once(plotters::prelude::Polygon::new(
    //                 vec![
    //                     (t_before, y_top),
    //                     (t_after, y_top),
    //                     (t_after, y_bottom),
    //                     (t_before, y_bottom),
    //                 ],
    //                 CYAN.mix(0.4).filled()  // More opaque cyan for visibility
    //             )))?;
    //         }
    //     }
        
    //     // t0 line (peak) - solid black, drawn on top of FWHM region
    //     if t0.is_finite() && t0 >= t_min && t0 <= t_max {
    //         chart.draw_series(std::iter::once(PathElement::new(
    //             vec![(t0, y_top), (t0, y_bottom)],
    //             BLACK.stroke_width(2)
    //         )))?;
    //     }
    // }

    // for b in &band_plots {
    //     let color = get_band_color(&b.label);

    //     // band uncertainty band
    //     if !b.mags_upper.is_empty() && b.mags_upper.len() == b.times_pred.len() {
    //         let mut area: Vec<(f64, f64)> = Vec::with_capacity(b.times_pred.len() * 2);
    //         for i in 0..b.times_pred.len() {
    //             area.push((b.times_pred[i], b.mags_upper[i]));
    //         }
    //         for i in (0..b.times_pred.len()).rev() {
    //             area.push((b.times_pred[i], b.mags_lower[i]));
    //         }
    //         chart.draw_series(std::iter::once(Polygon::new(area, color.mix(0.18).filled())))?;
    //     }

    //     // model line
    //     chart.draw_series(LineSeries::new(
    //         b.times_pred.iter().zip(b.mags_model.iter()).map(|(t, m)| (*t, *m)),
    //         color.stroke_width(2),
    //     ))?
    //     .label(b.legend_label.clone())
    //     .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));

    //     // Error bars for observations
    //     chart.draw_series(
    //         b.times_obs.iter()
    //             .zip(b.mags_obs.iter())
    //             .zip(b.mag_errors.iter())
    //             .map(|((&t, &m), &err)| {
    //                 PathElement::new(vec![(t, m - err), (t, m + err)], color.stroke_width(1))
    //             })
    //     )?;
        
    //     // observations (drawn on top of error bars)
    //     chart.draw_series(b.times_obs.iter().zip(b.mags_obs.iter()).map(|(t, m)| {
    //         Circle::new((*t, *m), 3, color.filled())
    //     }))?;
    // }

    // chart.configure_series_labels()
    //     .background_style(&WHITE.mix(0.8))
    //     .border_style(&BLACK)
    //     .label_font(("sans-serif", 30))
    //     .margin(20)
    //     .draw()?;

    // root.present()?;
    // println!("✓ Villar plot {}", output_path.display());
    
    // // Store object name with band in all timescale params
    // for param in &mut timescale_params_all {
    //     param.band = format!("{}|{}", object_name, param.band);
    // }
    
    Ok((total_fit_time, timescale_params_all))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut targets: Vec<String> = Vec::new();
    if args.len() >= 2 {
        targets.push(args[1].clone());
    } else {
        let dir = Path::new("lightcurves_csv");
        if !dir.exists() {
            eprintln!("lightcurves_csv not found");
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
            eprintln!("No CSVs found in lightcurves_csv");
            std::process::exit(1);
        }
        targets.sort();
    }

    let output_dir = Path::new("parametric_plots");
    fs::create_dir_all(output_dir)?;

    let mut total_fit_time = 0.0;
    let clock_start = Instant::now();
    let mut all_params: Vec<VillarTimescaleParams> = Vec::new();
    
    let parallel_results: Vec<_> = targets.par_iter().map(|t| {(t, process_file(t, output_dir))}).collect();
    let clock_duration = clock_start.elapsed().as_secs_f64();

    for (t, result) in parallel_results {
         match result{
                 Ok((fit_time, params)) => {
                 total_fit_time += fit_time;
                 all_params.extend(params);
             },
         Err(e) => eprintln!("Error processing {}: {}", t, e),
         }
    }

//    for (idx, t) in targets.iter().enumerate() {
//        println!("\n[{}/{}] Villar fitting {}", idx + 1, targets.len(), t);
//        match process_file(t, output_dir) {
//            Ok((fit_time, params)) => {
//                total_fit_time += fit_time;
//                all_params.extend(params);
//            },
//            Err(e) => eprintln!("Error processing {}: {}", t, e),
//        }
//    }

    // Save timescale parameters to CSV
    let csv_path = "parametric_timescale_parameters.csv";
    if !all_params.is_empty() {
        let mut csv_content = String::from("object,band,variant,rise_time_days,decay_time_days,peak_time_days,chi2,n_obs,fwhm_days,rise_rate_mag_per_day,decay_rate_mag_per_day,powerlaw_amplitude,powerlaw_index\n");
        for param in &all_params {
            // Extract object and band from the combined "object|band" format
            let (object, band) = if let Some(pipe_idx) = param.band.find('|') {
                (&param.band[..pipe_idx], &param.band[pipe_idx+1..])
            } else {
                ("unknown", param.band.as_str())
            };
            csv_content.push_str(&format!("{},{},{},{},{},{},{:.3},{},{},{},{},{},{}\n",
                object, band, param.variant,
                if param.rise_time.is_nan() { String::from("NaN") } else { format!("{:.3}", param.rise_time) },
                if param.decay_time.is_nan() { String::from("NaN") } else { format!("{:.3}", param.decay_time) },
                format!("{:.3}", param.peak_time),
                param.chi2, param.n_obs,
                if param.fwhm.is_nan() { String::from("NaN") } else { format!("{:.3}", param.fwhm) },
                if param.rise_rate.is_nan() { String::from("NaN") } else { format!("{:.6}", param.rise_rate) },
                if param.decay_rate.is_nan() { String::from("NaN") } else { format!("{:.6}", param.decay_rate) },
                if param.powerlaw_amplitude.is_nan() { String::from("NaN") } else { format!("{:.6}", param.powerlaw_amplitude) },
                if param.powerlaw_index.is_nan() { String::from("NaN") } else { format!("{:.6}", param.powerlaw_index) }
            ));
        }
        fs::write(csv_path, csv_content)?;
        println!("✓ Timescale parameters saved to: {}", csv_path);
    }

    println!("\n✓ Completed {} light curves", targets.len());
    println!("  Plots in {}", output_dir.display());
    println!("  Total fitting time (all cores): {:.2}s", total_fit_time);
    println!("  Fitting time: {:.2}s", clock_duration);
    Ok(())
}
