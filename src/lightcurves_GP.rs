use crate::models::{Band, PhotometryMag};
use std::collections::HashMap;
use scirs2_core::ndarray::{Array1, Array2, Axis, s};
use sklears_gaussian_process::{
    GaussianProcessRegressor, GprTrained,
    kernels::{ConstantKernel, RBF, WhiteKernel, ProductKernel, SumKernel},
};
use sklears_core::traits::{Fit, Predict, Trained};
use serde::{Serialize, Deserialize};

use argmin::core::{Error, Executor, State};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;

/// Prepare photometry: sort by time and remove duplicates
pub fn prepare_photometry(photometry: &mut Vec<PhotometryMag>) {           
    photometry.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    photometry.dedup_by(|a, b| a.time == b.time && a.band == b.band);        
}    

/// Per-band Villar properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerBandPropertiesVillar {
    pub times: Array1<f64>,
    pub mags: Array1<f64>,
    pub y_pred: Array1<f64>,
    pub scale: f64,
}

/// All-band Villar properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllBandsPropertiesVillar {
    pub ref_band: String,
    pub per_band: HashMap<String, PerBandPropertiesVillar>,
}

/// Fit a single-band Villar and predict at specified points
fn fit_gp_predict(
    times: &Array1<f64>,
    mags: &Array1<f64>,
    times_pred: &Array1<f64>,
) -> (GaussianProcessRegressor<GprTrained>, Array1<f64>) {
    let n = times.len();
    let mut xt = Array2::<f64>::zeros((n, 1));
    for (i, t) in times.iter().enumerate() {
        xt[[i, 0]] = *t;
    }

    // Composite kernel: Constant * RBF + White
    let cst: Box<dyn sklears_gaussian_process::Kernel> = Box::new(ConstantKernel::new(1.0));
    let rbf: Box<dyn sklears_gaussian_process::Kernel> = Box::new(RBF::new(1.0));
    let product = ProductKernel::new(vec![cst, rbf]);
    let prod_box: Box<dyn sklears_gaussian_process::Kernel> = Box::new(product);
    let white_box: Box<dyn sklears_gaussian_process::Kernel> = Box::new(WhiteKernel::new(0.4));
    let sum_kernel = SumKernel::new(vec![prod_box, white_box]);

    let mut gpr = GaussianProcessRegressor::new()
        .kernel(Box::new(sum_kernel))
        .alpha(1e-6)
        .normalize_y(true);

    let fitted = gpr.fit(&xt, mags).expect("Failed to fit Villar");

    let mut xpred_arr = Array2::<f64>::zeros((times_pred.len(), 1));
    for (i, t) in times_pred.iter().enumerate() {
        xpred_arr[[i, 0]] = *t;
    }
    let y_pred = fitted.predict(&xpred_arr).expect("Failed to predict Villar");

    (fitted, y_pred)
}

/// Effective wavelengths for each band [Angstrom]
pub fn effective_wavelengths() -> HashMap<String, f64> {
    let mut map = HashMap::new();
    map.insert("G".to_string(), 4770.0);
    map.insert("R".to_string(), 6231.0);
    map.insert("I".to_string(), 7625.0);
    map.insert("UVW2".to_string(), 2079.0);
    map.insert("U".to_string(), 3465.0);
    map.insert("J".to_string(), 12350.0);
    map
}

/// Convert AB magnitudes to flux [erg/cm^2/s/Hz]
fn mag_to_flux(mag: f64) -> f64 {
    10f64.powf(-(mag + 48.6) / 2.5)
}

/// Blackbody flux at wavelength lambda [Angstrom] and temperature T [K]
fn blackbody_flux(lambda: f64, temp: f64) -> f64 {
    const H: f64 = 6.62607015e-27; // erg*s
    const C: f64 = 2.99792458e10;  // cm/s
    const K: f64 = 1.380649e-16;   // erg/K
    let lambda_cm = lambda * 1e-8; // Å -> cm
    let b = 2.0 * H * C.powi(2) / lambda_cm.powi(5);
    let e = (H * C) / (lambda_cm * K * temp);
    b / (e.exp() - 1.0)
}

/// Fit blackbody temperature T and scaling A to observed fluxes at a single time
fn fit_blackbody(leffs: &[f64], fluxes: &[f64]) -> f64 {
    // Simple brute-force grid search (can replace with proper least-squares)
    let mut best_t = 0.0;
    let mut best_res = f64::MAX;

    for t in (2000..40000).step_by(50) {
        let temp = t as f64;
        let mut res = 0.0;

        // Scale factor A via linear least squares
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (lambda, &f_obs) in leffs.iter().zip(fluxes.iter()) {
            let f_bb = blackbody_flux(*lambda, temp);
            numerator += f_obs * f_bb;
            denominator += f_bb * f_bb;
        }
        let a = numerator / denominator;

        for (lambda, &f_obs) in leffs.iter().zip(fluxes.iter()) {
            let f_model = a * blackbody_flux(*lambda, temp);
            res += (f_obs - f_model).powi(2);
        }

        if res < best_res {
            best_res = res;
            best_t = temp;
        }
    }

    best_t
}

/// Compute temperature curve from per-band GP predictions using blackbody fit
pub fn compute_temperature_curve(
    per_band_map: &HashMap<String, PerBandPropertiesVillar>,
    times_grid: &Array1<f64>,
    y_pred_agg: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let wave_map = effective_wavelengths();
    let n = times_grid.len();
    let mut temp_curve = Array1::<f64>::zeros(n);

    for (i, &t) in times_grid.iter().enumerate() {
        let mut fluxes = Vec::new();
        let mut lambdas = Vec::new();

        for (band, props) in per_band_map.iter() {
            if let Some(&lambda) = wave_map.get(band) {
                lambdas.push(lambda);

                // Apply the per-band scale to the aggregated GP prediction
                let flux = mag_to_flux(y_pred_agg[i] * props.scale);
                fluxes.push(flux);

                println!(
                    "t = {:.3}, band: {}, scaled_mag_pred: {:.3}, flux_pred: {:.6}, lambda: {:.1} Å",
                    t,
                    band,
                    y_pred_agg[i] * props.scale,
                    flux,
                    lambda
                );
            }
        }

        if !fluxes.is_empty() {
            temp_curve[i] = fit_blackbody(&lambdas, &fluxes);
            println!("Fitted T_bb = {:.0} K", temp_curve[i]);
        } else {
            temp_curve[i] = 0.0;
        }
    }

    (times_grid.clone(), temp_curve)
}

/// Main analysis function
pub fn analyze_photometry(
    photometry: &[PhotometryMag],
) -> (PerBandPropertiesVillar, AllBandsPropertiesVillar, bool) {
    if photometry.is_empty() {
        panic!("Empty photometry array");
    }

    // Sort and group by band
    let mut sorted_photometry = photometry.to_vec();
    sorted_photometry.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    let mut band_map: HashMap<String, Vec<PhotometryMag>> = HashMap::new();
    for m in sorted_photometry.iter() {
        band_map.entry(format!("{:?}", m.band)).or_default().push(m.clone());
    }

    // Reference band = most points
    let ref_band = band_map
        .iter()
        .max_by_key(|(_, v)| v.len())
        .unwrap()
        .0
        .clone();

    let ref_data = &band_map[&ref_band];
    let times_ref = Array1::from(ref_data.iter().map(|m| m.time).collect::<Vec<_>>());
    let mags_ref = Array1::from(ref_data.iter().map(|m| m.mag as f64).collect::<Vec<_>>());

    let (fitted_ref, y_pred_ref) = fit_gp_predict(&times_ref, &mags_ref, &times_ref);

    let mut per_band_map: HashMap<String, PerBandPropertiesVillar> = HashMap::new();
    per_band_map.insert(
        ref_band.clone(),
        PerBandPropertiesVillar {
            times: times_ref.clone(),
            mags: mags_ref.clone(),
            y_pred: y_pred_ref.clone(),
            scale: 1.0,
        },
    );

    // Least-squares scale for other bands
    let mut scales: HashMap<String, f64> = HashMap::new();
    for (band_name, band_data) in band_map.iter() {
        if *band_name == ref_band {
            continue;
        }

        let times_band = Array1::from(band_data.iter().map(|m| m.time).collect::<Vec<_>>());
        let mags_band = Array1::from(band_data.iter().map(|m| m.mag as f64).collect::<Vec<_>>());

        // Interpolate reference Villar at band times
        let mut xpred = Array2::<f64>::zeros((times_band.len(), 1));
        for (i, t) in times_band.iter().enumerate() {
            xpred[[i, 0]] = *t;
        }
        let y_pred_interp = fitted_ref.predict(&xpred).expect("Villar predict failed");

        // Least squares scale
        let numer: f64 = y_pred_interp
            .iter()
            .zip(mags_band.iter())
            .map(|(yp, ym)| yp * ym)
            .sum();
        let denom: f64 = y_pred_interp.iter().map(|yp| yp.powi(2)).sum();
        let scale = numer / denom;
        scales.insert(band_name.clone(), scale);

        let mags_scaled = &mags_band / scale;
        let (_fitted_band, y_pred_band) = fit_gp_predict(&times_band, &mags_scaled, &times_band);

        per_band_map.insert(
            band_name.clone(),
            PerBandPropertiesVillar {
                times: times_band.clone(),
                mags: mags_band.clone(),
                y_pred: y_pred_band.clone() * scale,
                scale,
            },
        );
    }

    // --- Second aggregated Villar including per-band offsets ---
    let mut all_times: Vec<f64> = Vec::new();
    let mut all_mags: Vec<f64> = Vec::new();
    let mut band_labels: Vec<String> = Vec::new();
    for (band_name, props) in per_band_map.iter() {
        let offset = props.scale - 1.0;
        for (&t, &y) in props.times.iter().zip(props.mags.iter()) {
            all_times.push(t);
            all_mags.push(y / props.scale); // scaled
            band_labels.push(band_name.clone());
        }
    }

    let times_all = Array1::from(all_times);
    let mags_all = Array1::from(all_mags);


// Generate a linspace covering the full range of times
let t_min = times_all.iter().cloned().fold(f64::INFINITY, f64::min);
let t_max = times_all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let n_pred = 500; // number of points in the uniform grid
let times_grid = Array1::linspace(t_min, t_max, n_pred);

// Fit aggregated GP and predict on uniform grid
let (_fitted_agg, y_pred_agg) = fit_gp_predict(&times_all, &mags_all, &times_grid);

// Apply per-band offsets on interpolated grid
let mut idx = 0;
for (_band_name, props) in per_band_map.iter_mut() {
    let n = props.times.len();
    // Here you might want to interpolate the GP predictions back onto the band's original times,
    // or you can just use the uniform grid for plotting temperature curves.
    let y_slice = y_pred_agg.slice(s![idx..idx + n]).to_owned();
    props.y_pred = &y_slice * props.scale;
    idx += n;
}

    println!("--- Time-dependent Blackbody Temperature Curve ---");
    let (times_bb, temp_curve) =
        compute_temperature_curve(&per_band_map, &times_grid, &y_pred_agg);

    println!("--- Time-dependent Blackbody Temperature Curve ---");
    for (t, temp) in times_bb.iter().zip(temp_curve.iter()) {
        println!("t = {:.3}, T_bb ~ {:.0} K", t, temp);
    }


    (
        per_band_map[&ref_band].clone(),
        AllBandsPropertiesVillar {
            ref_band,
            per_band: per_band_map,
        },
        true,
    )
}

