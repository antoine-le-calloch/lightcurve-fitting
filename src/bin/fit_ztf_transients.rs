use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;
use plotters::prelude::*;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::particleswarm::ParticleSwarm;

// Model variants
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ModelVariant {
    Full,
    FastDecay,
    PowerLaw,
}

// Villar flux models
pub fn villar_flux(a: f64, beta: f64, gamma: f64, t0: f64, tau_rise: f64, tau_fall: f64, t: f64) -> f64 {
    let phase = t - t0;
    let sigmoid = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let piece = if phase < gamma {
        1.0 - beta * phase
    } else {
        (1.0 - beta * gamma) * ((gamma - phase) / tau_fall).exp()
    };
    a * sigmoid * piece
}

pub fn villar_flux_decay(a: f64, beta: f64, gamma: f64, t0: f64, tau_fall: f64, t: f64) -> f64 {
    let phase = t - t0;
    let piece = if phase < gamma {
        1.0 - beta * phase
    } else {
        (1.0 - beta * gamma) * ((gamma - phase) / tau_fall).exp()
    };
    a * piece
}

pub fn powerlaw_flux(a: f64, alpha: f64, t0: f64, t: f64) -> f64 {
    let phase = t - t0;
    if phase <= 0.0 {
        a
    } else {
        a * phase.powf(-alpha)
    }
}

#[derive(Clone)]
struct BandFitData {
    times: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
}

#[derive(Clone)]
struct SingleBandVillarCost {
    band: BandFitData,
    variant: ModelVariant,
}

impl CostFunction for SingleBandVillarCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        match self.variant {
            ModelVariant::PowerLaw => {
                let a = p[0].exp();
                let alpha = p[1];
                let t0 = p[2];
                let sigma_extra = p[3].exp();
                
                if !a.is_finite() || !sigma_extra.is_finite() {
                    return Ok(1e99);
                }
                
                let mut total_chi2 = 0.0;
                let mut n = 0usize;
                for i in 0..self.band.times.len() {
                    let model = powerlaw_flux(a, alpha, t0, self.band.times[i]);
                    let diff = model - self.band.flux[i];
                    let var = self.band.flux_err[i].powi(2) + sigma_extra.powi(2) + 1e-10;
                    total_chi2 += diff * diff / var;
                    n += 1;
                }
                
                let penalty = if t0 < -100.0 || t0 > 30.0 || alpha < 0.0 || alpha > 5.0 {
                    1e6
                } else {
                    0.0
                };
                
                Ok(total_chi2 / n.max(1) as f64 + penalty)
            }
            _ => {
                let a = p[0].exp();
                let beta = p[1];
                let gamma = p[2].exp();
                let t0 = p[3];
                let tau_rise = p[4].exp();
                let tau_fall = p[5].exp();
                let sigma_extra = p[6].exp();

                if !a.is_finite() || !gamma.is_finite() || !tau_rise.is_finite() || !tau_fall.is_finite() || !sigma_extra.is_finite() {
                    return Ok(1e99);
                }

                let mut total_chi2 = 0.0;
                let mut n = 0usize;
                for i in 0..self.band.times.len() {
                    let model = match self.variant {
                        ModelVariant::Full => villar_flux(a, beta, gamma, t0, tau_rise, tau_fall, self.band.times[i]),
                        ModelVariant::FastDecay => villar_flux_decay(a, beta, gamma, t0, tau_fall, self.band.times[i]),
                        ModelVariant::PowerLaw => unreachable!(),
                    };
                    let diff = model - self.band.flux[i];
                    let var = self.band.flux_err[i].powi(2) + sigma_extra.powi(2) + 1e-10;
                    total_chi2 += diff * diff / var;
                    n += 1;
                }

                let penalty = if t0 < -100.0 || t0 > 100.0 || tau_rise < 1e-6 || tau_rise > 1e4 || tau_fall < 1e-6 || tau_fall > 1e4 {
                    1e6
                } else {
                    0.0
                };

                Ok(total_chi2 / n.max(1) as f64 + penalty)
            }
        }
    }
}

fn pso_bounds(variant: ModelVariant, t_min: f64, t_max: f64) -> (Vec<f64>, Vec<f64>) {
    if matches!(variant, ModelVariant::PowerLaw) {
        // PowerLaw: [log_a, alpha, t0, log_sigma_extra]
        let lower = vec![-0.5, 0.3, t_min - 0.5, -3.0];
        let upper = vec![0.8, 3.0, t_max + 0.2, 0.0];
        return (lower, upper);
    }
    
    // Villar variants: adapt bounds to data time range
    let t_span = t_max - t_min;
    let lower = vec![-0.5, 0.0, -8.0, t_min - t_span * 0.3, -5.0, -5.0, -3.0];
    let upper = vec![0.8, 0.1, 4.0, t_max + t_span * 0.1, 5.0, 5.0, 0.0];
    (lower, upper)
}

fn pso_bounds_with_prior(variant: ModelVariant, ref_params: &[f64], t_min: f64, t_max: f64) -> (Vec<f64>, Vec<f64>) {
    if matches!(variant, ModelVariant::PowerLaw) {
        // PowerLaw: [log_a, alpha, t0, log_sigma_extra]
        // Tighten around reference with modest spread
        let spreads = vec![0.3, 0.5, 2.0, 0.5];
        let mut lower = vec![-1.0, 0.3, t_min - 1.0, -1.5];
        let mut upper = vec![1.0, 3.0, t_max + 0.5, 1.0];
        
        for i in 0..ref_params.len().min(4) {
            lower[i] = (ref_params[i] - spreads[i]).max(lower[i]);
            upper[i] = (ref_params[i] + spreads[i]).min(upper[i]);
        }
        return (lower, upper);
    }
    
    // Villar variants with priors from De Soto et al.
    // Parameters: [log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
    // Priors (std dev): [0.11, 0.27, 0.17, 4.4, 0.19, 0.26, 0.25]
    // Use ~2-3 sigma as bounds around reference
    let spreads = vec![0.3, 0.6, 0.5, 10.0, 0.5, 0.7, 0.6];
    let hard_lower = vec![-1.0, -2.0, -1.5, -50.0, -1.5, -1.5, -1.5];
    let hard_upper = vec![1.0, 1.0, 1.5, 30.0, 1.5, 1.5, 1.0];
    
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    
    for i in 0..7 {
        let l = (ref_params[i] - spreads[i]).max(hard_lower[i]);
        let u = (ref_params[i] + spreads[i]).min(hard_upper[i]);
        lower.push(l);
        upper.push(u);
    }
    
    (lower, upper)
}

struct BandResult {
    band_name: String,
    times_obs: Vec<f64>,
    flux_obs: Vec<f64>,
    flux_err: Vec<f64>,
    times_pred: Vec<f64>,
    flux_model: Vec<f64>,
    chi2: f64,
    sigma_extra: f64,
    variant: ModelVariant,
}

struct FitResult {
    name: String,
    bands: Vec<BandResult>,
    best_variant: ModelVariant,
    fit_time_ms: f64,
}

fn fit_transient(name: &str, csv_path: &Path) -> Option<FitResult> {
    let start = Instant::now();
    
    // Read CSV
    let content = fs::read_to_string(csv_path).ok()?;
    let mut lines = content.lines();
    lines.next(); // skip header
    
    // Parse data by filter
    let mut filter_data: HashMap<String, (Vec<f64>, Vec<f64>, Vec<f64>)> = HashMap::new();
    
    for line in lines {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 { continue; }
        
        let delta_t: f64 = parts[0].parse().ok()?;
        let log_flux: f64 = parts[2].parse().ok()?;
        let log_sigma: f64 = parts[3].parse().ok()?;
        let filter = parts[4].trim();
        
        // Convert from log space to linear
        let flux = 10f64.powf(log_flux);
        // sigma_fluxes is log10(sigma), so convert it
        let flux_err = 10f64.powf(log_sigma);
        
        filter_data.entry(filter.to_string())
            .or_insert((Vec::new(), Vec::new(), Vec::new()))
            .0.push(delta_t);
        filter_data.get_mut(filter).unwrap().1.push(flux);
        filter_data.get_mut(filter).unwrap().2.push(flux_err);
    }
    
    if filter_data.is_empty() {
        return None;
    }
    
    // Prepare band data
    let mut band_fit_data: Vec<(String, BandFitData)> = Vec::new();
    for (filter, (times, fluxes, errors)) in filter_data {
        if times.len() < 3 { continue; }
        
        // Sort by time
        let mut indices: Vec<usize> = (0..times.len()).collect();
        indices.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());
        
        let times: Vec<f64> = indices.iter().map(|&i| times[i]).collect();
        let fluxes: Vec<f64> = indices.iter().map(|&i| fluxes[i]).collect();
        let errors: Vec<f64> = indices.iter().map(|&i| errors[i]).collect();
        
        // Normalize to peak flux
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 { continue; }
        
        let norm_flux: Vec<f64> = fluxes.iter().map(|&f| f / peak_flux).collect();
        let norm_err: Vec<f64> = errors.iter().map(|&e| e / peak_flux).collect();
        
        band_fit_data.push((filter, BandFitData {
            times,
            flux: norm_flux,
            flux_err: norm_err,
        }));
    }
    
    if band_fit_data.is_empty() {
        return None;
    }
    
    // Find best-sampled band
    let best_idx = band_fit_data.iter()
        .enumerate()
        .max_by_key(|(_, (_, data))| data.times.len())
        .map(|(i, _)| i)
        .unwrap_or(0);
    
    // Get time range for bounds
    let all_times: Vec<f64> = band_fit_data.iter()
        .flat_map(|(_, data)| &data.times)
        .cloned()
        .collect();
    let t_min = all_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = all_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Fit best band with all variants
    let run_fit = |band_data: &BandFitData, variant: ModelVariant| {
        let (lower, upper) = pso_bounds(variant, t_min, t_max);
        let solver = ParticleSwarm::new((lower, upper), 60);
        let problem = SingleBandVillarCost { band: band_data.clone(), variant };
        let res = Executor::new(problem, solver)
            .configure(|state| state.max_iters(120))
            .run()
            .ok()?;
        let best = res.state().get_best_param()?;
        let chi2 = res.state().get_cost();
        Some((best.position.clone(), chi2))
    };
    
    let (_, best_band_data) = &band_fit_data[best_idx];
    let (params_full, chi2_full) = run_fit(best_band_data, ModelVariant::Full)?;
    let (params_fast, chi2_fast) = run_fit(best_band_data, ModelVariant::FastDecay)?;
    let (params_power, chi2_power) = run_fit(best_band_data, ModelVariant::PowerLaw)?;
    
    let (best_ref_params, best_variant, _) = {
        let mut best = (params_full, ModelVariant::Full, chi2_full);
        if chi2_fast < best.2 { best = (params_fast, ModelVariant::FastDecay, chi2_fast); }
        if chi2_power < best.2 { best = (params_power, ModelVariant::PowerLaw, chi2_power); }
        best
    };
    
    // Now fit each band: reference band uses its own params, others use priors
    let mut band_results = Vec::new();
    
    // Determine global time range for consistent plotting
    let global_t_min = all_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let global_t_max = all_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let global_t_range = global_t_max - global_t_min;
    let t_pred: Vec<f64> = (0..200).map(|i| 
        global_t_min - global_t_range * 0.1 + i as f64 * global_t_range * 1.2 / 200.0
    ).collect();
    
    for (band_idx, (filter, band_data)) in band_fit_data.iter().enumerate() {
        let (band_params, band_chi2) = if band_idx == best_idx {
            // Reference band - already fitted
            (best_ref_params.clone(), if best_variant == ModelVariant::Full { chi2_full } 
             else if best_variant == ModelVariant::FastDecay { chi2_fast } else { chi2_power })
        } else {
            // Other bands - use priors based on reference band
            let (lower, upper) = pso_bounds_with_prior(best_variant, &best_ref_params, t_min, t_max);
            let solver = ParticleSwarm::new((lower, upper), 60);
            let problem = SingleBandVillarCost { band: band_data.clone(), variant: best_variant };
            let res = Executor::new(problem, solver)
                .configure(|state| state.max_iters(120))
                .run()
                .ok()?;
            let best = res.state().get_best_param()?;
            let chi2 = res.state().get_cost();
            (best.position.clone(), chi2)
        };
        
        // Use global time range for all bands so lines span full x-axis
        let sigma_extra = match best_variant {
            ModelVariant::PowerLaw => band_params[3].exp(),
            _ => band_params[6].exp(),
        };
        
        let flux_model: Vec<f64> = t_pred.iter().map(|&t| {
            match best_variant {
                ModelVariant::Full => {
                    let a = band_params[0].exp();
                    let beta = band_params[1];
                    let gamma = band_params[2].exp();
                    let t0 = band_params[3];
                    let tau_rise = band_params[4].exp();
                    let tau_fall = band_params[5].exp();
                    villar_flux(a, beta, gamma, t0, tau_rise, tau_fall, t)
                }
                ModelVariant::FastDecay => {
                    let a = band_params[0].exp();
                    let beta = band_params[1];
                    let gamma = band_params[2].exp();
                    let t0 = band_params[3];
                    let tau_fall = band_params[5].exp();
                    villar_flux_decay(a, beta, gamma, t0, tau_fall, t)
                }
                ModelVariant::PowerLaw => {
                    let a = band_params[0].exp();
                    let alpha = band_params[1];
                    let t0 = band_params[2];
                    powerlaw_flux(a, alpha, t0, t)
                }
            }
        }).collect();
        
        band_results.push(BandResult {
            band_name: filter.clone(),
            times_obs: band_data.times.clone(),
            flux_obs: band_data.flux.clone(),
            flux_err: band_data.flux_err.clone(),
            times_pred: t_pred.clone(),
            flux_model,
            chi2: band_chi2,
            sigma_extra,
            variant: best_variant,
        });
    }
    
    let fit_time = start.elapsed().as_secs_f64() * 1000.0;
    
    Some(FitResult {
        name: name.to_string(),
        bands: band_results,
        best_variant,
        fit_time_ms: fit_time,
    })
}

fn plot_transient(result: &FitResult, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find flux range
    let mut flux_min = f64::INFINITY;
    let mut flux_max = f64::NEG_INFINITY;
    for band in &result.bands {
        for &f in &band.flux_obs {
            flux_min = flux_min.min(f);
            flux_max = flux_max.max(f);
        }
        for &f in &band.flux_model {
            if f > 0.0 {
                flux_min = flux_min.min(f);
                flux_max = flux_max.max(f);
            }
        }
    }
    
    let t_min = result.bands.iter().flat_map(|b| &b.times_obs).cloned().fold(f64::INFINITY, f64::min);
    let t_max = result.bands.iter().flat_map(|b| &b.times_obs).cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Add margin
    let flux_margin = (flux_max - flux_min) * 0.15;
    flux_min = (flux_min - flux_margin).max(0.0);
    flux_max += flux_margin;
    
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - Model: {:?}", result.name, result.best_variant),
            ("sans-serif", 30).into_font()
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, flux_min..flux_max)?;
    
    chart.configure_mesh()
        .x_desc("Time (days)")
        .y_desc("Normalized Flux")
        .draw()?;
    
    let colors = vec![&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];
    
    for (idx, band) in result.bands.iter().enumerate() {
        let color = colors[idx % colors.len()];
        
        // Plot observations with error bars
        chart.draw_series(
            band.times_obs.iter().zip(&band.flux_obs).zip(&band.flux_err)
                .map(|((&t, &f), &e)| {
                    ErrorBar::new_vertical(t, f - e, f, f + e, color.filled(), 5)
                })
        )?;
        
        chart.draw_series(
            band.times_obs.iter().zip(&band.flux_obs)
                .map(|(&t, &f)| Circle::new((t, f), 4, color.filled()))
        )?
        .label(format!("{}-band (χ²={:.3})", band.band_name, band.chi2))
        .legend(move |(x, y)| Circle::new((x + 10, y), 4, color.filled()));
        
        // Plot model
        chart.draw_series(LineSeries::new(
            band.times_pred.iter().zip(&band.flux_model)
                .filter(|(_, f)| **f > 0.0)
                .map(|(&t, &f)| (t, f)),
            color.stroke_width(2),
        ))?;
    }
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .draw()?;
    
    root.present()?;
    Ok(())
}

fn main() {
    println!("Fitting ZTF Transients with Villar Models\n");
    
    let csv_dir = Path::new("recovered_csv");
    if !csv_dir.exists() {
        eprintln!("Error: recovered_csv directory not found");
        return;
    }
    
    // Create output directory
    let output_dir = Path::new("ztf_fits");
    fs::create_dir_all(output_dir).ok();
    
    // Get list of CSV files
    let entries = fs::read_dir(csv_dir).expect("Failed to read directory");
    let mut csv_files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "csv").unwrap_or(false))
        .collect();
    
    csv_files.sort_by_key(|e| e.file_name());
    
    println!("Found {} CSV files\n", csv_files.len());
    
    let mut success_count = 0;
    let mut fail_count = 0;
    let overall_start = Instant::now();
    
    // Process first 10 for now
    for (i, entry) in csv_files.iter().take(10).enumerate() {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_str().unwrap();
        
        print!("Fitting {}/{}: {}... ", i + 1, csv_files.len().min(10), name);
        
        match fit_transient(name, &path) {
            Some(result) => {
                let avg_chi2: f64 = result.bands.iter().map(|b| b.chi2).sum::<f64>() / result.bands.len() as f64;
                let output_path = output_dir.join(format!("{}.png", name));
                match plot_transient(&result, output_path.to_str().unwrap()) {
                    Ok(_) => {
                        println!("✓ ({:.1} ms, {:?}, avg χ²={:.3})", 
                                result.fit_time_ms, result.best_variant, avg_chi2);
                        if i < 3 && result.best_variant != ModelVariant::PowerLaw {
                            // Show params for first 3
                            if result.best_variant == ModelVariant::Full {
                                let p = &result.bands[0];
                                println!("    → bands: {}", result.bands.iter().map(|b| b.band_name.as_str()).collect::<Vec<_>>().join(", "));
                            }
                        }
                        success_count += 1;
                    }
                    Err(e) => {
                        println!("✗ plot failed: {}", e);
                        fail_count += 1;
                    }
                }
            }
            None => {
                println!("✗ fit failed");
                fail_count += 1;
            }
        }
    }
    
    let total_time = overall_start.elapsed().as_secs_f64();
    
    println!("\n=== Summary ===");
    println!("Success: {}", success_count);
    println!("Failed: {}", fail_count);
    println!("Total time: {:.2}s", total_time);
    println!("Average time per object: {:.1}ms", total_time * 1000.0 / success_count.max(1) as f64);
    println!("\nPlots saved to: {}/", output_dir.display());
}
