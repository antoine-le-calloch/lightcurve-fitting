
use crate::models::{Band, PhotometryMag};
use argmin::core::{CostFunction, Gradient, Error, Executor, State};             
use argmin::solver::particleswarm::ParticleSwarm;
use rand::thread_rng;
use itertools::izip;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use plotters::prelude::*;
use rand::Rng;
use ndarray::{Array2, Axis};


/// Prepare photometry: sort by time and remove duplicates
pub fn prepare_photometry(photometry: &mut Vec<PhotometryMag>) {
    photometry.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    photometry.dedup_by(|a, b| a.time == b.time && a.band == b.band);
}

#[derive(Debug, Clone)]
pub struct BandData {
    pub xt: Vec<f64>,
    pub yt: Vec<f64>,
    pub sigma: Vec<f64>,
    pub band_name: String,
    pub is_ref: bool,
}

/// Villar photometry cost struct
#[derive(Debug, Clone)]  
pub struct VillarCost {
    pub times: Vec<f64>,
    pub flux: Vec<f64>,
    pub flux_err: Vec<f64>,
}

impl CostFunction for VillarCost {
    type Param = Vec<f64>;   // parameters in log-space
    type Output = f64;       // chi2 value

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // unpack parameters
        let log_a = p[0];
        let beta = p[1];
        let log_gamma = p[2];
        let t0 = p[3];
        let log_tau_rise = p[4];
        let log_tau_fall = p[5];

        // turn log parameters into physical ones
        let a = log_a.exp();
        let gamma = log_gamma.exp();
        let tau_rise = log_tau_rise.exp();
        let tau_fall = log_tau_fall.exp();

        if !a.is_finite() || !gamma.is_finite() || !tau_rise.is_finite() || !tau_fall.is_finite() {
            return Ok(1e99); // safe fail
        }

        let mut chi2 = 0.0;

        for i in 0..self.times.len() {
            let model = villar_flux(a, beta, gamma, t0, tau_rise, tau_fall, self.times[i]);

            let diff = model - self.flux[i];
            let inv_var = 1.0 / (self.flux_err[i].powi(2) + 1e-8);

            chi2 += diff * diff * inv_var;
        }

        // Optional: enforce constraints (soft penalties)
        let mut penalty = 0.0;

        let expterm = (-gamma / tau_rise).exp();
        let numerator = expterm * (1.0/beta - gamma);
        let denom = 1.0 + expterm;
        let rhs = numerator / denom;

        let c1 = tau_rise - rhs;   // c1 <= 0 is the valid region
        let c2 = gamma - (1.0 - beta * tau_fall) / beta;

        if c1 > 0.0 { penalty += 1e6 * c1; }
        if c2 > 0.0 { penalty += 1e6 * c2; }

        Ok(chi2 + penalty)
    }
}

/// Villar flux model (log-space parameters applied outside)
pub fn villar_flux(
    a: f64,
    beta: f64,
    gamma: f64,
    t0: f64,
    tau_rise: f64,
    tau_fall: f64,
    t: f64,
) -> f64 {
    let phase = t - t0;

    let sigmoid = 1.0 / (1.0 + (-phase / tau_rise).exp());

    let piece = if phase < gamma {
        1.0 - beta * phase
    } else {
        (1.0 - beta * gamma) * ((gamma - phase) / tau_fall).exp()
    };

    a * sigmoid * piece
}

/// Convert magnitude to flux (arbitrary units)
fn mag_to_flux(mag: f64) -> f64 {
    10f64.powf(-0.4 * mag)
}

/// Convert magnitude error to flux error
fn mag_err_to_flux_err(mag: f64, mag_err: f64) -> f64 {
    let f = mag_to_flux(mag);
    f * (mag_err * 0.4 * std::f64::consts::LN_10)
}

/// Priors and bounds based on your paper table
pub fn pso_bounds() -> (Vec<f64>, Vec<f64>) {
    let lower = vec![
        -0.3f64, // log(A min)
        0.0f64,             // beta
        1e-8f64,     // log(gamma)
        -100.0f64,          // t0
        1e-8f64,     // log(tau_rise)
        1e-8f64,     // log(tau_fall)
    ];

    let upper = vec![
        0.5f64, // log(A max)
        0.03f64,           // beta
        3.5f64, // log(gamma)
        30.0,           // t0
        4.0f64, // log(tau_rise)
        4.0f64, // log(tau_fall)
    ];

    (lower, upper)
}

/// Generate random start points inside prior space
pub fn random_start_params() -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (lo, hi) = pso_bounds();
    lo.iter().zip(&hi).map(|(a,b)| rng.gen_range(*a..=*b)).collect()
}

fn covariance_matrix(samples: &Vec<Vec<f64>>) -> Array2<f64> {
    let n = samples.len();
    let d = samples[0].len();
    let mut mat = Array2::<f64>::zeros((n, d));

    for (i, row) in samples.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            mat[[i, j]] = *val;
        }
    }

    let mean = mat.mean_axis(Axis(0)).unwrap();
    let centered = mat - &mean;

    centered.t().dot(&centered) / (n as f64 - 1.0)
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    let idx = ((p/100.0) * (v.len()-1) as f64).round() as usize;
    v[idx]
}

/// Main analysis function
pub fn analyze_photometry(
    sorted_photometry: &[PhotometryMag],
) -> Result<(), anyhow::Error> {
    // Choose t0 as the earliest time
    let t0 = sorted_photometry
        .iter()
        .map(|m| m.time)
        .fold(f64::INFINITY, f64::min);

    // Group photometry by band
    let mut band_map: HashMap<String, Vec<PhotometryMag>> = HashMap::new();
    for m in sorted_photometry.iter() {
        band_map.entry(format!("{:?}", m.band))
                .or_default()
                .push(m.clone());
    }

    // Pick reference band (first one)
    let ref_band_name = band_map
        .iter()
        .max_by_key(|(_, v)| v.len())
        .map(|(k, _)| k.clone())
        .unwrap();

    // Build BandData structs in flux space, normalized by peak
    let mut bands = Vec::new();
    for (band_name, mut mags) in band_map.into_iter() {
        if mags.is_empty() { continue; }

        // Sort by time first
        mags.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        
        // Convert to flux
        let mut fluxes: Vec<f64> = mags.iter().map(|m| mag_to_flux(m.mag as f64)).collect();
        let mut flux_errs: Vec<f64> = mags.iter()
            .map(|m| mag_err_to_flux_err(m.mag as f64, m.mag_err as f64))
            .collect();
        
        // Find peak flux for normalization
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux > 0.0 {
            for f in fluxes.iter_mut() { *f /= peak_flux; }
            for e in flux_errs.iter_mut() { *e /= peak_flux; }
        }
        
        // Compute binned values (mean in bin)
        let mut binned_time = Vec::new();
        let mut binned_flux = Vec::new();
        let mut binned_err = Vec::new();
        let bin_width = 0.01; // adjust as needed
        let mut i = 0;
        while i < fluxes.len() {
            let t0 = mags[i].time;
            let mut t_bin = vec![];
            let mut f_bin = vec![];
            let mut e_bin = vec![];
            while i < fluxes.len() && (mags[i].time - t0).abs() <= bin_width {
                t_bin.push(mags[i].time);
                f_bin.push(fluxes[i]);
                e_bin.push(flux_errs[i]);
                i += 1;
            }
            let t_avg = t_bin.iter().sum::<f64>() / t_bin.len() as f64;
            let f_avg = f_bin.iter().sum::<f64>() / f_bin.len() as f64;
            let e_avg = if e_bin.len() > 1 {
                let var = e_bin.iter().map(|x| (x - f_avg).powi(2)).sum::<f64>() / (e_bin.len() - 1) as f64;
                var.sqrt()
            } else {
                e_bin[0]
            };
            binned_time.push(t_avg);
            binned_flux.push(f_avg);
            binned_err.push(e_avg);
        }

        // Sort binned results by time
        let mut combined: Vec<(f64, f64, f64)> = binned_time
            .into_iter()
            .zip(binned_flux.into_iter())
            .zip(binned_err.into_iter())
            .map(|((t, f), e)| (t, f, e))
            .collect();
        
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Split into separate vectors
        let xt: Vec<f64> = combined.iter().map(|x| x.0 - t0).collect();
        let yt: Vec<f64> = combined.iter().map(|x| x.1).collect();
        let sigma: Vec<f64> = combined.iter().map(|x| x.2).collect();

        bands.push(BandData {
            xt,
            yt,
            sigma,
            band_name: band_name.clone(),
            is_ref: band_name == ref_band_name,
        });
    }

    for band_data in &bands {
        println!("--- Band: {} ---", band_data.band_name);
        for i in 0..band_data.yt.len() {
            println!(
                "t = {:.4}, flux = {:.6}, flux_err = {:.6}",
                band_data.xt[i],
                band_data.yt[i],
                band_data.sigma[i],
            );
        }
    }

    println!("Prepared {} bands (flux-normalized) for fitting.", bands.len());

    // Select the band with the most points
    let best_band = bands
        .iter()
        .max_by_key(|b| b.xt.len())
        .expect("No bands available");
    
    // Extract times, flux, and flux errors
    let times = &best_band.xt;
    let flux = &best_band.yt;
    let flux_err = &best_band.sigma;
    
    println!("Fitting band: {}", best_band.band_name);
    println!("Number of points: {}", times.len());

    let t_max = times[flux.iter().position(|&f| f == *flux.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap()).unwrap()];

    let (lower, upper) = pso_bounds();
    let bounds = (lower.clone(), upper.clone());
    let n_particles = 100;

    let solver = ParticleSwarm::new(bounds, n_particles);

    let problem = VillarCost {
        times: times.to_vec(),
        flux: flux.to_vec(),
        flux_err: flux_err.to_vec(),
    };

    let res = argmin::core::Executor::new(problem.clone(), solver)
        .configure(|state| state.max_iters(100))
        .run()
        .expect("PSO failed");

    let best_params = res.state().get_best_param().unwrap().position.to_vec();

    let swarm_opt = res.state().get_population();

    if swarm_opt.is_none() {
        panic!("PSO returned no swarm particles");
    }
    
    let swarm = swarm_opt.unwrap();
    
    // Extract particle positions
    let swarm_positions: Vec<Vec<f64>> = swarm
        .iter()
        .map(|p| p.position.clone())
        .collect();

    println!("Positions:\n{swarm_positions:?}");

    let cov = covariance_matrix(&swarm_positions);
    println!("Covariance matrix:\n{cov:?}");

    let ndim = swarm_positions[0].len();
    let mut perc_16 = vec![0.0; ndim];
    let mut perc_50 = vec![0.0; ndim];
    let mut perc_84 = vec![0.0; ndim];
    
    for j in 0..ndim {
        let mut column: Vec<f64> = swarm_positions.iter().map(|p| p[j]).collect();
        column.sort_by(|a,b| a.partial_cmp(b).unwrap());
    
        perc_16[j] = percentile(&mut column.clone(), 16.0);
        perc_50[j] = percentile(&mut column.clone(), 50.0);
        perc_84[j] = percentile(&mut column.clone(), 84.0);
    }
    
    println!("Median: {:?}", perc_50);
    println!("−1σ:   {:?}", perc_50.iter().zip(&perc_16).map(|(m,l)| m-l).collect::<Vec<_>>());
    println!("+1σ:   {:?}", perc_84.iter().zip(&perc_50).map(|(u,m)| u-m).collect::<Vec<_>>());

    println!("Best parameters (log space): {:?}", best_params);
    println!("Best chi2: {}", res.state().get_cost());

    // Generate model curve
    let tmin = *times.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    let tmax = *times.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    let npts = 100;
    let dt = (tmax - tmin) / (npts as f64 - 1.0);
    let mut model_t = Vec::new();
    let mut model_f = Vec::new();
    for i in 0..npts {
        let t = tmin + i as f64 * dt;
        model_t.push(t);
        let a = best_params[0].exp();
        let beta = best_params[1];
        let gamma = best_params[2].exp();
        let t0 = best_params[3];
        let tau_rise = best_params[4].exp();
        let tau_fall = best_params[5].exp();
        model_f.push(villar_flux(a, beta, gamma, t0, tau_rise, tau_fall, t));
    }

    // Plot using plotters
    let root = BitMapBackend::new("villar_fit.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE);
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Villar Fit", ("sans-serif", 30))
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(tmin..tmax, 0.0..flux.iter().cloned().fold(0./0., f64::max))?;

    chart.configure_mesh().draw()?;

    // Draw points with error bars
    for (&x, (&y, &yerr)) in times.iter().zip(flux.iter().zip(flux_err.iter())) {
        // point
        chart.draw_series(std::iter::once(Circle::new((x, y), 3, RED.filled())))?;
        // vertical error bar
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y - yerr), (x, y + yerr)],
            &BLACK,
        )))?;
    }

    chart.draw_series(LineSeries::new(
        model_t.iter().cloned().zip(model_f.iter().cloned()),
        &BLUE,
    ))?;

    Ok(())
}

