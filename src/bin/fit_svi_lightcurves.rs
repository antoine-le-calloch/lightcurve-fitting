use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use argmin::core::{CostFunction, Error as ArgminError, Executor, State};
use argmin::solver::particleswarm::ParticleSwarm;
use plotters::prelude::*;

use lightcurve_fiting::lightcurve_common::read_ztf_lightcurve;

const ZP: f64 = 23.9;

// Physical constants (CGS)
const MSUN_CGS: f64 = 1.989e33; // grams
const C_CGS: f64 = 2.998e10;    // cm/s
const SECS_PER_DAY: f64 = 86400.0;

// ---------------------------------------------------------------------------
// Manual Adam optimizer (avoids built-in Adam issues with variable updates)
// ---------------------------------------------------------------------------

struct ManualAdam {
    m: Vec<f64>,
    v: Vec<f64>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
}

impl ManualAdam {
    fn new(n_params: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
        }
    }

    fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grads[i];
            if !g.is_finite() {
                continue;
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Model definitions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum SviModel {
    Bazin,
    Villar,
    MetzgerKN,
    Tde,
    Arnett,
    Magnetar,
    ShockCooling,
    Afterglow,
}

impl SviModel {
    fn n_params(self) -> usize {
        match self {
            SviModel::Bazin => 6,         // log_a, b, t0, log_tau_rise, log_tau_fall, log_sigma_extra
            SviModel::Villar => 7,        // log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra
            SviModel::MetzgerKN => 5,     // log10_mej, log10_vej, log10_kappa_r, t0, log_sigma_extra
            SviModel::Tde => 7,           // log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra
            SviModel::Arnett => 5,        // log_a, t0, log_tau_m, logit_f, log_sigma_extra
            SviModel::Magnetar => 5,      // log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra
            SviModel::ShockCooling => 5,  // log_a, t0, n, log_tau_tr, log_sigma_extra
            SviModel::Afterglow => 6,    // log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra
        }
    }

    /// Index of log_sigma_extra in the parameter vector for this model.
    fn sigma_extra_idx(self) -> usize {
        match self {
            SviModel::Bazin => 5,
            SviModel::Villar => 6,
            SviModel::MetzgerKN => 4,
            SviModel::Tde => 6,
            SviModel::Arnett => 4,
            SviModel::Magnetar => 4,
            SviModel::ShockCooling => 4,
            SviModel::Afterglow => 5,
        }
    }

    fn param_names(self) -> Vec<&'static str> {
        match self {
            SviModel::Bazin => vec!["log_a", "b", "t0", "log_tau_rise", "log_tau_fall", "log_sigma_extra"],
            SviModel::Villar => vec!["log_a", "beta", "log_gamma", "t0", "log_tau_rise", "log_tau_fall", "log_sigma_extra"],
            SviModel::MetzgerKN => vec!["log10_mej", "log10_vej", "log10_kappa_r", "t0", "log_sigma_extra"],
            SviModel::Tde => vec!["log_a", "b", "t0", "log_tau_rise", "log_tau_fall", "alpha", "log_sigma_extra"],
            SviModel::Arnett => vec!["log_a", "t0", "log_tau_m", "logit_f", "log_sigma_extra"],
            SviModel::Magnetar => vec!["log_a", "t0", "log_tau_sd", "log_tau_diff", "log_sigma_extra"],
            SviModel::ShockCooling => vec!["log_a", "t0", "n", "log_tau_tr", "log_sigma_extra"],
            SviModel::Afterglow => vec!["log_a", "t0", "log_t_b", "alpha1", "alpha2", "log_sigma_extra"],
        }
    }

    fn name(self) -> &'static str {
        match self {
            SviModel::Bazin => "Bazin",
            SviModel::Villar => "Villar",
            SviModel::MetzgerKN => "MetzgerKN",
            SviModel::Tde => "Tde",
            SviModel::Arnett => "Arnett",
            SviModel::Magnetar => "Magnetar",
            SviModel::ShockCooling => "ShockCooling",
            SviModel::Afterglow => "Afterglow",
        }
    }

    /// Whether this model requires batch (whole-lightcurve) evaluation.
    fn is_sequential(self) -> bool {
        matches!(self, SviModel::MetzgerKN)
    }
}

// Plain f64 model evaluation
fn bazin_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let b = params[1];
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let dt = t - t0;
    let e_fall = (-dt / tau_fall).exp();
    let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
    a * e_fall * sig + b
}

/// Analytical d(flux)/d(theta_j) for Bazin.
/// params: [log_a, b, t0, log_tau_rise, log_tau_fall]
fn bazin_flux_grad(params: &[f64], t: f64) -> Vec<f64> {
    let a = params[0].exp();
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let dt = t - t0;
    let e_fall = (-dt / tau_fall).exp();
    let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
    let base = a * e_fall * sig; // = flux - b

    // d(flux)/d(log_a) = a * e_fall * sig = base
    let d_log_a = base;
    // d(flux)/d(b) = 1
    let d_b = 1.0;
    // d(flux)/d(t0): d(dt)/d(t0) = -1
    //   d(e_fall)/d(t0) = e_fall / tau_fall
    //   d(sig)/d(t0) = -sig*(1-sig) / tau_rise
    let d_t0 = base * (1.0 / tau_fall - (1.0 - sig) / tau_rise);
    // d(flux)/d(log_tau_rise) = a * e_fall * sig*(1-sig) * (-dt/tau_rise)
    let d_log_tau_rise = -base * (1.0 - sig) * dt / tau_rise;
    // d(flux)/d(log_tau_fall) = a * sig * e_fall * dt / tau_fall = base * dt / tau_fall
    let d_log_tau_fall = base * dt / tau_fall;

    // d(flux)/d(log_sigma_extra) = 0 (sigma_extra affects likelihood, not flux prediction)
    let d_log_sigma_extra = 0.0;
    vec![d_log_a, d_b, d_t0, d_log_tau_rise, d_log_tau_fall, d_log_sigma_extra]
}

fn villar_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let beta = params[1];
    let gamma = params[2].exp();
    let t0 = params[3];
    let tau_rise = params[4].exp();
    let tau_fall = params[5].exp();
    let phase = t - t0;
    let sig_rise = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let k = 10.0;
    let w = 1.0 / (1.0 + (-k * (phase - gamma)).exp());
    let piece_left = 1.0 - beta * phase;
    let piece_right = (1.0 - beta * gamma) * ((gamma - phase) / tau_fall).exp();
    let piece = (1.0 - w) * piece_left + w * piece_right;
    a * sig_rise * piece
}

/// Analytical d(flux)/d(theta_j) for Villar.
/// params: [log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
fn villar_flux_grad(params: &[f64], t: f64) -> Vec<f64> {
    let a = params[0].exp();
    let beta = params[1];
    let gamma = params[2].exp();
    let t0 = params[3];
    let tau_rise = params[4].exp();
    let tau_fall = params[5].exp();
    let phase = t - t0;
    let k = 10.0;

    let sig_rise = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let w = 1.0 / (1.0 + (-k * (phase - gamma)).exp());
    let piece_left = 1.0 - beta * phase;
    let e_decay = ((gamma - phase) / tau_fall).exp();
    let piece_right = (1.0 - beta * gamma) * e_decay;
    let piece = (1.0 - w) * piece_left + w * piece_right;
    let flux = a * sig_rise * piece;

    // d(flux)/d(log_a) = flux
    let d_log_a = flux;

    // d(flux)/d(beta)
    let d_pl_dbeta = -phase;
    let d_pr_dbeta = -gamma * e_decay;
    let d_piece_dbeta = (1.0 - w) * d_pl_dbeta + w * d_pr_dbeta;
    let d_beta = a * sig_rise * d_piece_dbeta;

    // d(flux)/d(log_gamma): gamma = exp(log_gamma), chain rule with d(gamma)/d(log_gamma) = gamma
    let dw_dgamma = -k * w * (1.0 - w); // d(w)/d(gamma)
    let dw_dloggamma = dw_dgamma * gamma;
    let dpr_dgamma = e_decay * (-beta + (1.0 - beta * gamma) / tau_fall);
    let dpr_dloggamma = dpr_dgamma * gamma;
    let d_piece_dloggamma = dw_dloggamma * (piece_right - piece_left) + w * dpr_dloggamma;
    let d_log_gamma = a * sig_rise * d_piece_dloggamma;

    // d(flux)/d(t0): d(phase)/d(t0) = -1
    let dsig_dphase = sig_rise * (1.0 - sig_rise) / tau_rise;
    let dsig_dt0 = -dsig_dphase;
    let dw_dphase = k * w * (1.0 - w);
    let dw_dt0 = -dw_dphase;
    let dpl_dt0 = beta; // d(1-beta*phase)/d(t0) = beta
    let dpr_dt0 = (1.0 - beta * gamma) * e_decay / tau_fall; // d((gamma-phase)/tau_fall)/d(t0)=1/tau_fall
    let d_piece_dt0 = dw_dt0 * (piece_right - piece_left) + (1.0 - w) * dpl_dt0 + w * dpr_dt0;
    let d_t0 = a * (dsig_dt0 * piece + sig_rise * d_piece_dt0);

    // d(flux)/d(log_tau_rise) = a * piece * sig_rise*(1-sig_rise)*(-phase/tau_rise)
    let d_log_tau_rise = a * piece * sig_rise * (1.0 - sig_rise) * (-phase / tau_rise);

    // d(flux)/d(log_tau_fall): only piece_right depends on tau_fall
    // d(piece_right)/d(log_tau_fall) = piece_right * (phase - gamma) / tau_fall
    let d_pr_dlogtf = piece_right * (phase - gamma) / tau_fall;
    let d_log_tau_fall = a * sig_rise * w * d_pr_dlogtf;

    // d(flux)/d(log_sigma_extra) = 0 (not used in prediction)
    let d_log_sigma_extra = 0.0;

    vec![d_log_a, d_beta, d_log_gamma, d_t0, d_log_tau_rise, d_log_tau_fall, d_log_sigma_extra]
}

/// TDE model: sigmoid rise + power-law decay + baseline.
/// params: [log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra]
/// flux = a * sig(phase/tau_rise) * (1 + softplus(phase)/tau_fall)^(-alpha) + b
fn tde_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let b = params[1];
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let alpha = params[5];
    let phase = t - t0;

    // Sigmoid rise
    let sig = 1.0 / (1.0 + (-phase / tau_rise).exp());
    // Softplus for smooth max(0, phase)
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    // Power-law decay
    let w = 1.0 + phase_soft / tau_fall;
    let decay = w.powf(-alpha);

    a * sig * decay + b
}

/// Analytical d(flux)/d(theta_j) for TDE model.
/// params: [log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra]
fn tde_flux_grad(params: &[f64], t: f64) -> Vec<f64> {
    let a = params[0].exp();
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let alpha = params[5];
    let phase = t - t0;

    let sig = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_phase = phase.exp() / (1.0 + phase.exp()); // d(softplus)/d(phase)
    let w = 1.0 + phase_soft / tau_fall;
    let decay = w.powf(-alpha);
    let base = a * sig * decay; // flux - b

    // d(flux)/d(log_a) = a * sig * decay  (chain: d(a)/d(log_a) = a)
    let d_log_a = base;

    // d(flux)/d(b) = 1
    let d_b = 1.0;

    // d(flux)/d(t0): d(phase)/d(t0) = -1
    //   d(sig)/d(t0) = -sig*(1-sig)/tau_rise
    //   d(decay)/d(t0) = alpha * w^(-alpha-1) * sig_phase / tau_fall
    let dsig_dt0 = -sig * (1.0 - sig) / tau_rise;
    let ddecay_dt0 = alpha * w.powf(-alpha - 1.0) * sig_phase / tau_fall;
    let d_t0 = a * (dsig_dt0 * decay + sig * ddecay_dt0);

    // d(flux)/d(log_tau_rise) = a * decay * d(sig)/d(log_tau_rise)
    //   d(sig)/d(log_tau_rise) = sig*(1-sig) * (-phase/tau_rise) * tau_rise = -sig*(1-sig)*phase/tau_rise
    //   but d(tau_rise)/d(log_tau_rise) = tau_rise, so total:
    //   = -sig*(1-sig) * phase/tau_rise  (the tau_rise in numerator from chain cancels with denominator)
    let d_log_tau_rise = a * decay * (-sig * (1.0 - sig) * phase / tau_rise);

    // d(flux)/d(log_tau_fall) = a * sig * d(decay)/d(log_tau_fall)
    //   d(w)/d(tau_fall) = -phase_soft/tau_fall^2, d(tau_fall)/d(log_tau_fall) = tau_fall
    //   d(w)/d(log_tau_fall) = -phase_soft/tau_fall
    //   d(decay)/d(log_tau_fall) = -alpha * w^(-alpha-1) * (-phase_soft/tau_fall)
    //                            = alpha * w^(-alpha-1) * phase_soft / tau_fall
    let d_log_tau_fall = a * sig * alpha * w.powf(-alpha - 1.0) * phase_soft / tau_fall;

    // d(flux)/d(alpha) = a * sig * d(decay)/d(alpha)
    //   decay = w^(-alpha) = exp(-alpha * ln(w))
    //   d(decay)/d(alpha) = -ln(w) * decay
    let d_alpha = a * sig * (-w.ln()) * decay;

    // d(flux)/d(log_sigma_extra) = 0
    let d_log_sigma_extra = 0.0;

    vec![d_log_a, d_b, d_t0, d_log_tau_rise, d_log_tau_fall, d_alpha, d_log_sigma_extra]
}

/// Arnett model: Ni-56/Co-56 radioactive decay with diffusion trapping.
/// params: [log_a, t0, log_tau_m, logit_f, log_sigma_extra]
/// flux = a * heat(t) * trap(t)
///   heat = f*exp(-t/τ_Ni) + (1-f)*exp(-t/τ_Co)  (radioactive heating)
///   trap = 1 - exp(-(t/τ_m)²)                    (diffusion trapping)
fn arnett_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_m = params[2].exp();
    let logit_f = params[3];

    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;

    let f = 1.0 / (1.0 + (-logit_f).exp());

    const TAU_NI: f64 = 8.8;
    const TAU_CO: f64 = 111.3;

    let e_ni = (-phase_soft / TAU_NI).exp();
    let e_co = (-phase_soft / TAU_CO).exp();
    let heat = f * e_ni + (1.0 - f) * e_co;

    let x = phase_soft / tau_m;
    let trap = 1.0 - (-x * x).exp();

    a * heat * trap
}

fn arnett_flux_grad(params: &[f64], t: f64) -> Vec<f64> {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_m = params[2].exp();
    let logit_f = params[3];

    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());

    let f = 1.0 / (1.0 + (-logit_f).exp());

    const TAU_NI: f64 = 8.8;
    const TAU_CO: f64 = 111.3;

    let e_ni = (-phase_soft / TAU_NI).exp();
    let e_co = (-phase_soft / TAU_CO).exp();
    let heat = f * e_ni + (1.0 - f) * e_co;

    let x = phase_soft / tau_m;
    let exp_x2 = (-x * x).exp();
    let trap = 1.0 - exp_x2;

    let flux = a * heat * trap;

    let d_log_a = flux;

    // d/d(t0): chain through softplus
    let dheat_dps = -f * e_ni / TAU_NI - (1.0 - f) * e_co / TAU_CO;
    let dtrap_dps = 2.0 * phase_soft * exp_x2 / (tau_m * tau_m);
    let d_t0 = a * (-sig_p) * (dheat_dps * trap + heat * dtrap_dps);

    // d/d(log_tau_m): increasing tau_m decreases trap
    let d_log_tau_m = -2.0 * a * heat * exp_x2 * x * x;

    // d/d(logit_f): sigmoid derivative
    let d_logit_f = a * trap * (e_ni - e_co) * f * (1.0 - f);

    vec![d_log_a, d_t0, d_log_tau_m, d_logit_f, 0.0]
}

/// Magnetar model: spindown luminosity with diffusion trapping.
/// params: [log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra]
/// flux = a * (1 + t/τ_sd)^(-2) * (1 - exp(-(t/τ_diff)²))
fn magnetar_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_sd = params[2].exp();
    let tau_diff = params[3].exp();

    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;

    let w = 1.0 + phase_soft / tau_sd;
    let spindown = w.powi(-2);

    let x = phase_soft / tau_diff;
    let trap = 1.0 - (-x * x).exp();

    a * spindown * trap
}

fn magnetar_flux_grad(params: &[f64], t: f64) -> Vec<f64> {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_sd = params[2].exp();
    let tau_diff = params[3].exp();

    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());

    let w = 1.0 + phase_soft / tau_sd;
    let spindown = w.powi(-2);

    let x = phase_soft / tau_diff;
    let exp_x2 = (-x * x).exp();
    let trap = 1.0 - exp_x2;

    let flux = a * spindown * trap;

    let d_log_a = flux;

    // d/d(t0)
    let dspindown_dps = -2.0 * w.powi(-3) / tau_sd;
    let dtrap_dps = 2.0 * phase_soft * exp_x2 / (tau_diff * tau_diff);
    let d_t0 = a * (-sig_p) * (dspindown_dps * trap + spindown * dtrap_dps);

    // d/d(log_tau_sd): increasing tau_sd → spindown decays slower → more flux
    let d_log_tau_sd = a * trap * 2.0 * phase_soft * w.powi(-3) / tau_sd;

    // d/d(log_tau_diff): increasing tau_diff → less escapes → less flux
    let d_log_tau_diff = -2.0 * a * spindown * exp_x2 * x * x;

    vec![d_log_a, d_t0, d_log_tau_sd, d_log_tau_diff, 0.0]
}

/// Shock cooling model: power-law cooling with Gaussian transparency cutoff.
/// params: [log_a, t0, n, log_tau_tr, log_sigma_extra]
/// flux = a * sigmoid(5*phase) * phase_soft^(-n) * exp(-(phase_soft/τ_tr)²)
fn shockcooling_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let n = params[2];
    let tau_tr = params[3].exp();

    let phase = t - t0;
    let sig5 = 1.0 / (1.0 + (-phase * 5.0).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;

    let cooling = phase_soft.powf(-n);
    let ratio = phase_soft / tau_tr;
    let cutoff = (-ratio * ratio).exp();

    a * sig5 * cooling * cutoff
}

fn shockcooling_flux_grad(params: &[f64], t: f64) -> Vec<f64> {
    let a = params[0].exp();
    let t0 = params[1];
    let n = params[2];
    let tau_tr = params[3].exp();

    let phase = t - t0;
    let sig5 = 1.0 / (1.0 + (-phase * 5.0).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());

    let cooling = phase_soft.powf(-n);
    let ratio = phase_soft / tau_tr;
    let cutoff = (-ratio * ratio).exp();
    let base = cooling * cutoff; // without sig5
    let flux = a * sig5 * base;

    let d_log_a = flux;

    // d/d(t0): d(phase)/d(t0) = -1
    let d_t0 = a * base * (
        -5.0 * sig5 * (1.0 - sig5)
        + sig5 * sig_p * (n / phase_soft + 2.0 * phase_soft / (tau_tr * tau_tr))
    );

    // d/d(n)
    let d_n = -flux * phase_soft.ln();

    // d/d(log_tau_tr)
    let d_log_tau_tr = flux * 2.0 * ratio * ratio;

    vec![d_log_a, d_t0, d_n, d_log_tau_tr, 0.0]
}

/// Afterglow model: smoothly broken power law (Beuermann+1999).
/// params: [log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra]
/// flux = a * [(phase/t_b)^(2*alpha1) + (phase/t_b)^(2*alpha2)]^(-1/2)
/// alpha1 < alpha2: pre-break slope vs post-break (steeper) slope.
/// For t << t_b: flux ~ a * (t/t_b)^(-alpha1)
/// For t >> t_b: flux ~ a * (t/t_b)^(-alpha2)
fn afterglow_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let t_b = params[2].exp();
    let alpha1 = params[3];
    let alpha2 = params[4];

    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;

    let r = phase_soft / t_b;
    let ln_r = r.ln();
    let u1 = (2.0 * alpha1 * ln_r).exp();
    let u2 = (2.0 * alpha2 * ln_r).exp();
    let u = u1 + u2;

    a * u.powf(-0.5)
}

fn afterglow_flux_grad(params: &[f64], t: f64) -> Vec<f64> {
    let a = params[0].exp();
    let t0 = params[1];
    let t_b = params[2].exp();
    let alpha1 = params[3];
    let alpha2 = params[4];

    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp()); // d(softplus)/d(phase)

    let r = phase_soft / t_b;
    let ln_r = r.ln();
    let u1 = (2.0 * alpha1 * ln_r).exp();
    let u2 = (2.0 * alpha2 * ln_r).exp();
    let u = u1 + u2;
    let flux = a * u.powf(-0.5);

    // d(flux)/d(log_a) = flux
    let d_log_a = flux;

    // d(flux)/d(t0): chain through softplus, d(phase)/d(t0) = -1
    // du/d(phase) = (2*alpha1/phase_soft)*u1 + (2*alpha2/phase_soft)*u2
    let du_dps = (2.0 * alpha1 * u1 + 2.0 * alpha2 * u2) / phase_soft;
    // d(flux)/d(phase_soft) = a * (-0.5) * u^(-1.5) * du/d(phase_soft)
    let dflux_dps = a * (-0.5) * u.powf(-1.5) * du_dps;
    let d_t0 = dflux_dps * (-sig_p); // d(phase)/d(t0) = -1, d(phase_soft)/d(phase) = sig_p

    // d(flux)/d(log_t_b): d(r)/d(log_t_b) = -r, d(ln_r)/d(log_t_b) = -1
    // du/d(log_t_b) = -2*alpha1*u1 + -2*alpha2*u2
    let du_dlog_tb = -(2.0 * alpha1 * u1 + 2.0 * alpha2 * u2);
    let d_log_t_b = a * (-0.5) * u.powf(-1.5) * du_dlog_tb;

    // d(flux)/d(alpha1) = a * (-0.5) * u^(-1.5) * du/d(alpha1)
    // du/d(alpha1) = 2 * ln(r) * u1
    let d_alpha1 = a * (-0.5) * u.powf(-1.5) * 2.0 * ln_r * u1;

    // d(flux)/d(alpha2) = a * (-0.5) * u^(-1.5) * du/d(alpha2)
    // du/d(alpha2) = 2 * ln(r) * u2
    let d_alpha2 = a * (-0.5) * u.powf(-1.5) * 2.0 * ln_r * u2;

    // d(flux)/d(log_sigma_extra) = 0
    vec![d_log_a, d_t0, d_log_t_b, d_alpha1, d_alpha2, 0.0]
}

fn eval_model(model: SviModel, params: &[f64], t: f64) -> f64 {
    match model {
        SviModel::Bazin => bazin_flux_eval(params, t),
        SviModel::Villar => villar_flux_eval(params, t),
        SviModel::Tde => tde_flux_eval(params, t),
        SviModel::Arnett => arnett_flux_eval(params, t),
        SviModel::Magnetar => magnetar_flux_eval(params, t),
        SviModel::ShockCooling => shockcooling_flux_eval(params, t),
        SviModel::Afterglow => afterglow_flux_eval(params, t),
        SviModel::MetzgerKN => panic!("MetzgerKN requires batch evaluation"),
    }
}

/// Analytical d(flux)/d(theta_j) for pointwise models.
fn eval_model_grad(model: SviModel, params: &[f64], t: f64) -> Vec<f64> {
    match model {
        SviModel::Bazin => bazin_flux_grad(params, t),
        SviModel::Villar => villar_flux_grad(params, t),
        SviModel::Tde => tde_flux_grad(params, t),
        SviModel::Arnett => arnett_flux_grad(params, t),
        SviModel::Magnetar => magnetar_flux_grad(params, t),
        SviModel::ShockCooling => shockcooling_flux_grad(params, t),
        SviModel::Afterglow => afterglow_flux_grad(params, t),
        SviModel::MetzgerKN => panic!("MetzgerKN requires batch evaluation"),
    }
}

// ---------------------------------------------------------------------------
// Metzger 1-zone kilonova model (simplified from 300 mass layers)
// ---------------------------------------------------------------------------

/// 1-zone Metzger kilonova: Euler-step thermal energy evolution on a
/// log-spaced grid, return normalized bolometric luminosity.
/// params: [log10_mej, log10_vej, log10_kappa_r, t0, log_sigma_extra]
fn metzger_kn_eval_batch(params: &[f64], obs_times: &[f64]) -> Vec<f64> {
    let m_ej = 10f64.powf(params[0]) * MSUN_CGS;
    let v_ej = 10f64.powf(params[1]) * C_CGS;
    let kappa_r = 10f64.powf(params[2]);
    let t0 = params[3];

    let phases: Vec<f64> = obs_times.iter().map(|&t| t - t0).collect();
    let phase_max = phases.iter().cloned().fold(0.01f64, f64::max);
    if phase_max <= 0.01 {
        return vec![0.0; obs_times.len()];
    }

    // Fine log-spaced integration grid (days)
    let n_grid: usize = 200;
    let log_t_min = 0.01f64.ln();
    let log_t_max = (phase_max * 1.05).ln();
    let grid_t_day: Vec<f64> = (0..n_grid)
        .map(|i| (log_t_min + (log_t_max - log_t_min) * i as f64 / (n_grid - 1) as f64).exp())
        .collect();

    // Neutron / composition parameters (effective 1-zone averages)
    let ye: f64 = 0.1;
    let xn0: f64 = 1.0 - 2.0 * ye; // 0.8

    // Initial conditions (scale by 1e40 to prevent overflow, same as original)
    let scale: f64 = 1e40;
    let e0 = 0.5 * m_ej * v_ej * v_ej;
    let mut e_th = e0 / scale;
    let mut e_kin = e0 / scale;
    let mut v = v_ej;
    let mut r = grid_t_day[0] * SECS_PER_DAY * v;

    let mut grid_lrad: Vec<f64> = Vec::with_capacity(n_grid);

    for i in 0..n_grid {
        let t_day = grid_t_day[i];
        let t_sec = t_day * SECS_PER_DAY;

        // Thermalization efficiency (Barnes+16 eq. 34)
        let eth_factor = 0.34 * t_day.powf(0.74);
        let eth = 0.36
            * ((-0.56 * t_day).exp()
                + if eth_factor > 1e-10 {
                    (1.0 + eth_factor).ln() / eth_factor
                } else {
                    1.0
                });

        // Heating rates (erg/g/s)
        let xn = xn0 * (-t_sec / 900.0).exp();
        let eps_neutron = 3.2e14 * xn;
        // Korobkin+Rosswog r-process
        let time_term =
            (0.5 - ((t_sec - 1.3) / 0.11).atan() / std::f64::consts::PI).max(1e-30);
        let eps_rp = 2e18 * eth * time_term.powf(1.3);
        let l_heat = m_ej * (eps_neutron + eps_rp) / scale;

        // Effective opacity (neutron-decay iron-group + r-process)
        let xr = 1.0 - xn0;
        let xn_decayed = xn0 - xn;
        let kappa_eff = 0.4 * xn_decayed + kappa_r * xr;

        // Diffusion timescale + light-crossing
        let t_diff =
            3.0 * kappa_eff * m_ej / (4.0 * std::f64::consts::PI * C_CGS * v * t_sec) + r / C_CGS;

        // Radiative luminosity
        let l_rad = if e_th > 0.0 && t_diff > 0.0 {
            e_th / t_diff
        } else {
            0.0
        };
        grid_lrad.push(l_rad);

        // PdV work
        let l_pdv = if r > 0.0 { e_th * v / r } else { 0.0 };

        // Euler step
        if i < n_grid - 1 {
            let dt_sec = (grid_t_day[i + 1] - grid_t_day[i]) * SECS_PER_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if e_th < 0.0 {
                e_th = e_th.abs();
            }
            e_kin += l_pdv * dt_sec;
            v = (2.0 * e_kin * scale / m_ej).sqrt().min(C_CGS);
            r += v * dt_sec;
        }
    }

    // Normalize by peak
    let l_peak = grid_lrad.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if l_peak <= 0.0 || !l_peak.is_finite() {
        return vec![0.0; obs_times.len()];
    }
    let grid_norm: Vec<f64> = grid_lrad.iter().map(|l| l / l_peak).collect();

    // Interpolate to observation times
    phases
        .iter()
        .map(|&phase| {
            if phase <= 0.0 {
                return 0.0;
            }
            if phase <= grid_t_day[0] {
                return grid_norm[0];
            }
            if phase >= grid_t_day[n_grid - 1] {
                return *grid_norm.last().unwrap();
            }
            let idx = grid_t_day.partition_point(|&gt| gt < phase).min(n_grid - 1).max(1);
            let frac = (phase - grid_t_day[idx - 1]) / (grid_t_day[idx] - grid_t_day[idx - 1]);
            grid_norm[idx - 1] + frac * (grid_norm[idx] - grid_norm[idx - 1])
        })
        .collect()
}

/// Finite-difference gradient for MetzgerKN (batch).
/// Returns grads[i][j] = d(pred_i)/d(theta_j).
fn metzger_kn_grad_batch(params: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
    let n_times = times.len();
    let n_params = 5; // log10_mej, log10_vej, log10_kappa_r, t0, log_sigma_extra
    let n_phys = 4; // first 4 are physical; sigma_extra has 0 flux gradient

    let base = metzger_kn_eval_batch(params, times);
    let eps = 1e-5;
    let mut grads: Vec<Vec<f64>> = vec![vec![0.0; n_params]; n_times];

    for j in 0..n_phys {
        let mut p_plus = params.to_vec();
        p_plus[j] += eps;
        let f_plus = metzger_kn_eval_batch(&p_plus, times);
        for i in 0..n_times {
            grads[i][j] = (f_plus[i] - base[i]) / eps;
        }
    }
    // grads[i][4] (log_sigma_extra) stays 0.0
    grads
}

// ---------------------------------------------------------------------------
// Batch evaluation dispatch (works for all models)
// ---------------------------------------------------------------------------

/// Evaluate model at all observation times. For pointwise models this just
/// loops; for MetzgerKN it runs the full integration.
fn eval_model_batch(model: SviModel, params: &[f64], times: &[f64]) -> Vec<f64> {
    if model.is_sequential() {
        metzger_kn_eval_batch(params, times)
    } else {
        times.iter().map(|&t| eval_model(model, params, t)).collect()
    }
}

/// Gradient of model predictions w.r.t. params, at all times.
/// Returns grads[i][j] = d(pred_i)/d(theta_j).
fn eval_model_grad_batch(model: SviModel, params: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
    if model.is_sequential() {
        metzger_kn_grad_batch(params, times)
    } else {
        times.iter().map(|&t| eval_model_grad(model, params, t)).collect()
    }
}

// ---------------------------------------------------------------------------
// PSO cost function for quick model selection
// ---------------------------------------------------------------------------

/// Approximate erf(x) using Abramowitz & Stegun 7.1.26.
fn erf_approx(x: f64) -> f64 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let val = 1.0 - poly * (-a * a).exp();
    if x >= 0.0 { val } else { -val }
}

/// Approximate log(Phi(x)) where Phi is the standard normal CDF.
/// Uses the identity Phi(x) = 0.5 * erfc(-x/sqrt(2)) and a rational
/// approximation for erfc, accurate to ~1e-7.
fn log_normal_cdf(x: f64) -> f64 {
    if x > 8.0 { return 0.0; } // Phi(x) ≈ 1
    if x < -30.0 { return -0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln() - (-x).ln(); }
    // Use erfc approximation (Abramowitz & Stegun 7.1.26)
    let z = -x * std::f64::consts::FRAC_1_SQRT_2; // erfc(z) = 2*Phi(x) when z = -x/sqrt(2)
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erfc_z = poly * (-z * z).exp();
    let phi = if z >= 0.0 { 0.5 * erfc_z } else { 1.0 - 0.5 * erfc_z };
    (phi.max(1e-300)).ln()
}

#[derive(Clone)]
struct PsoCost {
    times: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
    is_upper: Vec<bool>,
    upper_flux: Vec<f64>,
    model: SviModel,
}

impl CostFunction for PsoCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let se_idx = self.model.sigma_extra_idx();
        let sigma_extra = p[se_idx].exp();
        let sigma_extra_sq = sigma_extra * sigma_extra;
        let mut preds = eval_model_batch(self.model, p, &self.times);

        // Renormalize MetzgerKN: model peaks at phase~0.01d but observations
        // are normalized by max(observed flux) which occurs later.
        // Clamp scale factor to [0.1, 10.0] to prevent numerical instability.
        if self.model == SviModel::MetzgerKN {
            let max_pred = preds.iter().zip(self.is_upper.iter())
                .filter(|(_, is_up)| !**is_up)
                .map(|(p, _)| *p)
                .fold(f64::NEG_INFINITY, f64::max);
            if max_pred > 1e-10 && max_pred.is_finite() {
                let scale = (1.0 / max_pred).clamp(0.1, 10.0);
                for pred in preds.iter_mut() { *pred *= scale; }
            }
        }

        let n = self.times.len().max(1) as f64;
        let mut neg_ll = 0.0;
        for i in 0..self.times.len() {
            let pred = preds[i];
            if !pred.is_finite() {
                return Ok(1e99);
            }
            let total_var = self.flux_err[i] * self.flux_err[i] + sigma_extra_sq + 1e-10;
            if self.is_upper[i] {
                // Upper limit: log Phi((f_upper - f_pred) / sigma_total)
                let z = (self.upper_flux[i] - pred) / total_var.sqrt();
                neg_ll -= log_normal_cdf(z);
            } else {
                // Detection: standard Gaussian
                let diff = pred - self.flux[i];
                neg_ll += diff * diff / total_var + total_var.ln();
            }
        }
        Ok(neg_ll / n)
    }
}

/// PSO model selection: try models one at a time in priority order,
/// stop as soon as one fits well enough (cost < EARLY_STOP_THRESHOLD).
///
/// Order (cheapest/broadest first):
///   Bazin → Arnett → Tde → Afterglow → Villar → Magnetar → ShockCooling → MetzgerKN
///
/// Returns (best_model, best_params, best_cost).
fn pso_model_select(data: &BandFitData) -> (SviModel, Vec<f64>, f64) {
    const EARLY_STOP: f64 = -3.0;

    let models: &[SviModel] = &[
        SviModel::Bazin,
        SviModel::Arnett,
        SviModel::Tde,
        SviModel::Afterglow,
        SviModel::Villar,
        SviModel::Magnetar,
        SviModel::ShockCooling,
        SviModel::MetzgerKN,
    ];

    let mut best_model = SviModel::Bazin;
    let mut best_params = vec![];
    let mut best_chi2 = f64::INFINITY;

    for &model in models {
        let (lower, upper) = pso_bounds(model);
        let problem = PsoCost {
            times: data.times.clone(),
            flux: data.flux.clone(),
            flux_err: data.flux_err.clone(),
            is_upper: data.is_upper.clone(),
            upper_flux: data.upper_flux.clone(),
            model,
        };
        let solver = ParticleSwarm::new((lower, upper), 40);
        let res = Executor::new(problem, solver)
            .configure(|state| state.max_iters(50))
            .run();
        match res {
            Ok(res) => {
                let chi2 = res.state().get_cost();
                if chi2 < best_chi2 {
                    best_chi2 = chi2;
                    best_model = model;
                    best_params = res.state().get_best_param().unwrap().position.clone();
                }
            }
            Err(e) => {
                eprintln!("  PSO error for {}: {}", model.name(), e);
            }
        }
        // Early stop: if current best is good enough, don't try more models
        if best_chi2 < EARLY_STOP {
            break;
        }
    }

    (best_model, best_params, best_chi2)
}

/// Per-model Gaussian prior: returns (center, half_width) for each parameter.
/// The SVI log-prior is N(center, half_width^2) for each param.
/// For phenomenological models, wide priors (center=0, width=2).
/// For MetzgerKN, tight priors from NMMA Me2017.prior.
fn prior_params(model: SviModel) -> Vec<(f64, f64)> {
    match model {
        SviModel::Bazin => {
            // log_a, b, t0, log_tau_rise, log_tau_fall, log_sigma_extra
            vec![(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
        }
        SviModel::Villar => {
            vec![(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
        }
        SviModel::MetzgerKN => {
            // Tight priors from NMMA: center of prior range, width = half-range
            // log10_mej: U[-3, -0.5] → center=-1.75, width=1.25
            // log10_vej: U[-2, -0.5] → center=-1.25, width=0.75
            // log10_kappa_r: U[-1, 2] → center=0.5, width=1.5
            // t0: U[-2, 1] → center=-0.5, width=1.5
            // log_sigma_extra: wide
            vec![(-1.75, 1.25), (-1.25, 0.75), (0.5, 1.5), (-0.5, 1.5), (0.0, 2.0)]
        }
        SviModel::Tde => {
            // log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra
            // alpha centered at 5/3 (TDE theoretical), width 1.0
            vec![(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (1.67, 1.0), (0.0, 2.0)]
        }
        SviModel::Arnett => {
            // log_a, t0, log_tau_m, logit_f, log_sigma_extra
            // tau_m ~10 days typical for SNe
            vec![(0.0, 2.0), (0.0, 2.0), (2.3, 1.0), (0.0, 2.0), (0.0, 2.0)]
        }
        SviModel::Magnetar => {
            // log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra
            vec![(0.0, 2.0), (0.0, 2.0), (3.0, 1.5), (2.3, 1.0), (0.0, 2.0)]
        }
        SviModel::ShockCooling => {
            // log_a, t0, n, log_tau_tr, log_sigma_extra
            // n ~ 0.5-1.0 typical for shock cooling
            vec![(0.0, 2.0), (0.0, 2.0), (0.5, 1.0), (1.0, 2.0), (0.0, 2.0)]
        }
        SviModel::Afterglow => {
            // log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra
            // alpha1 ~ 0.8 (pre-break, shallow decay), alpha2 ~ 2.0 (post-break, steep)
            vec![(0.0, 2.0), (0.0, 2.0), (1.0, 2.0), (0.8, 1.0), (2.0, 1.0), (0.0, 2.0)]
        }
    }
}

fn pso_bounds(model: SviModel) -> (Vec<f64>, Vec<f64>) {
    match model {
        SviModel::Bazin => {
            // log_a, b, t0, log_tau_rise, log_tau_fall, log_sigma_extra
            let lower = vec![-3.0, -1.0, -100.0, -2.0, -2.0, -5.0];
            let upper = vec![3.0, 1.0, 100.0, 5.0, 6.0, 0.0];
            (lower, upper)
        }
        SviModel::Villar => {
            // log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra
            let lower = vec![-3.0, -0.05, -3.0, -100.0, -2.0, -2.0, -5.0];
            let upper = vec![3.0, 0.1, 5.0, 100.0, 5.0, 7.0, 0.0];
            (lower, upper)
        }
        SviModel::MetzgerKN => {
            // log10_mej, log10_vej, log10_kappa_r, t0, log_sigma_extra
            // Bounds from NMMA Me2017.prior (physically motivated)
            let lower = vec![-3.0, -2.0, -1.0, -2.0, -5.0];
            let upper = vec![-0.5, -0.5,  2.0,  1.0,  0.0];
            (lower, upper)
        }
        SviModel::Tde => {
            // log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra
            let lower = vec![-3.0, -1.0, -100.0, -2.0, -1.0, 0.5, -5.0];
            let upper = vec![ 3.0,  1.0,  100.0,  5.0,  6.0, 4.0,  0.0];
            (lower, upper)
        }
        SviModel::Arnett => {
            // log_a, t0, log_tau_m, logit_f, log_sigma_extra
            let lower = vec![-3.0, -100.0, 0.5, -3.0, -5.0];
            let upper = vec![ 3.0,  100.0, 4.5,  3.0,  0.0];
            (lower, upper)
        }
        SviModel::Magnetar => {
            // log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra
            let lower = vec![-3.0, -100.0, 0.0, 0.5, -5.0];
            let upper = vec![ 3.0,  100.0, 6.0, 4.5,  0.0];
            (lower, upper)
        }
        SviModel::ShockCooling => {
            // log_a, t0, n, log_tau_tr, log_sigma_extra
            let lower = vec![-3.0, -100.0, 0.1, -1.0, -5.0];
            let upper = vec![ 3.0,  100.0, 3.0,  4.0,  0.0];
            (lower, upper)
        }
        SviModel::Afterglow => {
            // log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra
            let lower = vec![-3.0, -100.0, -2.0, -2.0, 0.5, -5.0];
            let upper = vec![ 3.0,  100.0,  6.0,  3.0, 5.0,  0.0];
            (lower, upper)
        }
    }
}

// ---------------------------------------------------------------------------
// Heuristic initialization for variational means
// ---------------------------------------------------------------------------

fn init_variational_means(model: SviModel, data: &BandFitData) -> Vec<f64> {
    // Find approximate peak time and amplitude
    let mut peak_idx = 0;
    let mut peak_val = f64::NEG_INFINITY;
    for (i, &f) in data.flux.iter().enumerate() {
        if f > peak_val {
            peak_val = f;
            peak_idx = i;
        }
    }
    let t_peak = data.times[peak_idx];

    match model {
        SviModel::Bazin => {
            // log_a, b, t0, log_tau_rise, log_tau_fall, log_sigma_extra
            let log_a = peak_val.max(0.01).ln();
            let b = 0.0;
            let t0 = t_peak;
            let log_tau_rise = 2.0_f64.ln(); // ~2 days
            let log_tau_fall = 20.0_f64.ln(); // ~20 days
            let log_sigma_extra = -3.0; // small extra noise
            vec![log_a, b, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
        }
        SviModel::Villar => {
            // log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra
            let log_a = peak_val.max(0.01).ln();
            let beta = 0.01;
            let log_gamma = 10.0_f64.ln(); // ~10 days
            let t0 = t_peak;
            let log_tau_rise = 2.0_f64.ln();
            let log_tau_fall = 20.0_f64.ln();
            let log_sigma_extra = (-3.0_f64).ln().max(-5.0); // small extra noise
            vec![log_a, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
        }
        SviModel::MetzgerKN => {
            // log10_mej, log10_vej, log10_kappa_r, t0, log_sigma_extra
            let log10_mej = -2.0;  // 0.01 Msun
            let log10_vej = -0.7;  // ~0.2c
            let log10_kappa_r = 0.5; // ~3 cm^2/g
            let t0 = t_peak - 2.0; // merger ~2 days before peak
            let log_sigma_extra = -3.0;
            vec![log10_mej, log10_vej, log10_kappa_r, t0, log_sigma_extra]
        }
        SviModel::Tde => {
            // log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra
            let log_a = peak_val.max(0.01).ln();
            let b = 0.0;
            let t0 = t_peak - 10.0;        // onset ~10 days before peak for TDEs
            let log_tau_rise = 2.0_f64.ln(); // ~2 day rise
            let log_tau_fall = 20.0_f64.ln(); // ~20 day fallback timescale
            let alpha = 1.67;               // TDE canonical t^(-5/3)
            let log_sigma_extra = -3.0;
            vec![log_a, b, t0, log_tau_rise, log_tau_fall, alpha, log_sigma_extra]
        }
        SviModel::Arnett => {
            // log_a, t0, log_tau_m, logit_f, log_sigma_extra
            let log_a = peak_val.max(0.01).ln();
            let t0 = t_peak - 15.0;         // explosion ~15 days before peak
            let log_tau_m = 10.0_f64.ln();   // ~10 day diffusion timescale
            let logit_f = 0.0;              // f = 0.5 (equal Ni/Co)
            let log_sigma_extra = -3.0;
            vec![log_a, t0, log_tau_m, logit_f, log_sigma_extra]
        }
        SviModel::Magnetar => {
            // log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra
            let log_a = peak_val.max(0.01).ln();
            let t0 = t_peak - 10.0;         // explosion ~10 days before peak
            let log_tau_sd = 20.0_f64.ln();  // ~20 day spindown timescale
            let log_tau_diff = 10.0_f64.ln(); // ~10 day diffusion timescale
            let log_sigma_extra = -3.0;
            vec![log_a, t0, log_tau_sd, log_tau_diff, log_sigma_extra]
        }
        SviModel::ShockCooling => {
            // log_a, t0, n, log_tau_tr, log_sigma_extra
            let log_a = peak_val.max(0.01).ln();
            let t0 = t_peak - 2.0;          // shock breakout ~2 days before peak
            let n = 0.5;                     // canonical power-law index
            let log_tau_tr = 5.0_f64.ln();   // ~5 day transparency timescale
            let log_sigma_extra = -3.0;
            vec![log_a, t0, n, log_tau_tr, log_sigma_extra]
        }
        SviModel::Afterglow => {
            // log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra
            let log_a = peak_val.max(0.01).ln();
            let t0 = t_peak - 5.0;          // GRB trigger ~5 days before optical peak
            let log_t_b = 10.0_f64.ln();     // jet break ~10 days
            let alpha1 = 0.8;               // pre-break: shallow decay
            let alpha2 = 2.2;               // post-break: steep decay (~ -p)
            let log_sigma_extra = -3.0;
            vec![log_a, t0, log_t_b, alpha1, alpha2, log_sigma_extra]
        }
    }
}

// ---------------------------------------------------------------------------
// SVI fitting
// ---------------------------------------------------------------------------

struct SviFitResult {
    model: SviModel,
    mu: Vec<f64>,        // variational means (unconstrained space)
    log_sigma: Vec<f64>, // log of variational stds (calibrated)
    elbo: f64,           // final ELBO estimate
}

/// Sigma inflation factor to calibrate mean-field VI posteriors.
///
/// Mean-field Gaussian VI systematically underestimates posterior variance
/// because it cannot capture parameter correlations. This constant inflates
/// the variational sigma by a fixed factor, calibrated via P-P plots on
/// synthetic lightcurves from Arnett and Tde models.
///
/// Calibration method: generate 100+ synthetic lightcurves with known true
/// parameters, fit with SVI, compute z = (true - mu) / sigma for each param,
/// then set factor = median(MAD-based std(z)) across parameters and models.
/// A perfectly calibrated posterior yields factor = 1.0.
///
/// Current value: 4.0 (calibrated from all 7 models: Bazin, Villar, Tde,
/// Arnett, Magnetar, ShockCooling, Afterglow P-P plots, excluding t0 which
/// has fundamental multi-modality issues beyond width calibration).
/// Verified: at 4.0x, median MAD*1.48 = 0.904 (10% overcalibrated);
/// at 3.7x, median MAD*1.48 = 1.141 (14% undercalibrated).
/// 4.0x preferred: closer to ideal and slightly conservative (wider posteriors).
const SIGMA_INFLATION_FACTOR: f64 = 4.0;

fn svi_fit(
    model: SviModel,
    data: &BandFitData,
    n_steps: usize,
    n_samples: usize,
    lr: f64,
    pso_init: Option<&[f64]>,
) -> SviFitResult {
    let n_params = model.n_params();
    let n_variational = 2 * n_params; // mu + log_sigma for each param

    // Initialize variational parameters
    let mut var_params = vec![0.0; n_variational];
    let init_mu = if let Some(pso_params) = pso_init {
        pso_params.to_vec()
    } else {
        init_variational_means(model, data)
    };
    for i in 0..n_params {
        var_params[i] = init_mu[i];           // mu
        var_params[n_params + i] = -1.0;      // log_sigma (sigma ~ 0.37)
    }

    let mut adam = ManualAdam::new(n_variational, lr);

    // Precompute observational variance (sigma_obs^2) for each data point
    let obs_var: Vec<f64> = data.flux_err.iter().map(|e| e * e + 1e-10).collect();

    // Index of log_sigma_extra in the parameter vector
    let se_idx = model.sigma_extra_idx();

    let mut final_elbo = f64::NEG_INFINITY;

    for step in 0..n_steps {
        let mu = &var_params[..n_params];
        let log_sigma = &var_params[n_params..];
        let sigma: Vec<f64> = log_sigma.iter().map(|ls| ls.exp()).collect();

        // Accumulators for gradients (we minimize -ELBO, so negate at the end)
        let mut grad_mu = vec![0.0; n_params];
        let mut grad_log_sigma = vec![0.0; n_params];
        let mut elbo_sum = 0.0;

        for _ in 0..n_samples {
            // Draw epsilon ~ N(0, 1) and compute theta via reparameterization
            let mut eps = vec![0.0; n_params];
            let mut theta = vec![0.0; n_params];
            for j in 0..n_params {
                let u1: f64 = rand::random::<f64>().max(1e-10);
                let u2: f64 = rand::random::<f64>();
                eps[j] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                theta[j] = mu[j] + sigma[j] * eps[j];
            }

            // sigma_extra = exp(log_sigma_extra); total_var_i = obs_var_i + sigma_extra^2
            let sigma_extra = theta[se_idx].exp();
            let sigma_extra_sq = sigma_extra * sigma_extra;

            // Compute log-likelihood and its gradient w.r.t. theta
            // using batch evaluation (required for sequential models like MetzgerKN)
            let mut preds = eval_model_batch(model, &theta, &data.times);
            let mut grads = eval_model_grad_batch(model, &theta, &data.times);

            // Renormalize MetzgerKN: model peaks at phase~0.01d but observations
            // are normalized by max(observed flux) at detection times.
            // Clamp scale to [0.1, 10.0]; skip sample if clamping is too severe.
            if model == SviModel::MetzgerKN {
                let max_pred = preds.iter().zip(data.is_upper.iter())
                    .filter(|(_, is_up)| !**is_up)
                    .map(|(p, _)| *p)
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_pred > 1e-10 && max_pred.is_finite() {
                    let raw_scale = 1.0 / max_pred;
                    let scale = raw_scale.clamp(0.1, 10.0);
                    // Skip this MC sample if clamping changed scale by >50%
                    if (raw_scale - scale).abs() / raw_scale > 0.5 {
                        continue;
                    }
                    for pred in preds.iter_mut() { *pred *= scale; }
                    for grad_vec in grads.iter_mut() {
                        for g in grad_vec.iter_mut() { *g *= scale; }
                    }
                }
            }

            let mut log_lik = 0.0;
            let mut dll_dtheta = vec![0.0; n_params];

            for i in 0..data.times.len() {
                let pred = preds[i];
                if !pred.is_finite() { continue; }
                let total_var = obs_var[i] + sigma_extra_sq;
                let sigma_total = total_var.sqrt();

                if data.is_upper[i] {
                    // Upper limit: log Phi(z) where z = (f_upper - pred) / sigma_total
                    let z = (data.upper_flux[i] - pred) / sigma_total;
                    log_lik += log_normal_cdf(z);

                    // d(log Phi(z))/d(pred) = -phi(z) / (Phi(z) * sigma_total)
                    // = -(1/sigma_total) * phi(z)/Phi(z) (inverse Mills ratio)
                    let phi_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                    let cdf_z = (0.5 * (1.0 + erf_approx(z * std::f64::consts::FRAC_1_SQRT_2))).max(1e-300);
                    let dll_dpred = -phi_z / (cdf_z * sigma_total);

                    for j in 0..n_params {
                        if j != se_idx && grads[i][j].is_finite() {
                            dll_dtheta[j] += dll_dpred * grads[i][j];
                        }
                    }

                    // d(log Phi(z))/d(log_sigma_extra) via chain rule on z:
                    // dz/d(sigma_total) = -(f_upper - pred) / sigma_total^2
                    // d(sigma_total)/d(sigma_extra_sq) = 0.5 / sigma_total
                    // d(sigma_extra_sq)/d(log_sigma_extra) = 2 * sigma_extra_sq
                    // => dz/d(log_se) = -(f_upper - pred) * sigma_extra_sq / sigma_total^3
                    let dz_dlse = -(data.upper_flux[i] - pred) * sigma_extra_sq / (sigma_total * total_var);
                    dll_dtheta[se_idx] += (phi_z / cdf_z) * dz_dlse;
                } else {
                    // Detection: standard Gaussian likelihood
                    let residual = data.flux[i] - pred;
                    let inv_total = 1.0 / total_var;
                    let r2 = residual * residual;
                    log_lik += -0.5 * (r2 * inv_total + (2.0 * std::f64::consts::PI * total_var).ln());

                    for j in 0..n_params {
                        if j != se_idx && grads[i][j].is_finite() {
                            dll_dtheta[j] += residual * inv_total * grads[i][j];
                        }
                    }

                    dll_dtheta[se_idx] += (r2 * inv_total * inv_total - inv_total) * sigma_extra_sq;
                }
            }

            // Log-prior: per-model Gaussian priors
            let priors = prior_params(model);
            let mut log_prior = 0.0;
            let mut dlp_dtheta = vec![0.0; n_params];
            for j in 0..n_params {
                let (center, width) = priors[j];
                let var = width * width;
                log_prior += -0.5 * (theta[j] - center).powi(2) / var;
                dlp_dtheta[j] = -(theta[j] - center) / var;
            }

            elbo_sum += log_lik + log_prior;

            // Reparameterization trick gradients:
            // d(ELBO)/d(mu_j) = d(log_lik+log_prior)/d(theta_j) * d(theta_j)/d(mu_j)
            //                  = d(log_lik+log_prior)/d(theta_j) * 1
            // d(ELBO)/d(log_sigma_j) = d(log_lik+log_prior)/d(theta_j) * d(theta_j)/d(log_sigma_j)
            //                         = d(log_lik+log_prior)/d(theta_j) * sigma_j * eps_j
            for j in 0..n_params {
                let df_dtheta = dll_dtheta[j] + dlp_dtheta[j];
                grad_mu[j] += df_dtheta;
                grad_log_sigma[j] += df_dtheta * sigma[j] * eps[j];
            }
        }

        // Average over samples
        let ns = n_samples as f64;
        for j in 0..n_params {
            grad_mu[j] /= ns;
            grad_log_sigma[j] /= ns;
        }
        elbo_sum /= ns;

        // Add entropy: H[q] = sum(log_sigma_j) + 0.5 * P * ln(2*pi*e)
        let entropy: f64 = log_sigma.iter().sum::<f64>()
            + 0.5 * n_params as f64 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln();
        final_elbo = elbo_sum + entropy;

        // d(entropy)/d(log_sigma_j) = 1
        for j in 0..n_params {
            grad_log_sigma[j] += 1.0;
        }

        // Build the full gradient of -ELBO (we minimize -ELBO)
        let mut neg_elbo_grad = Vec::with_capacity(n_variational);
        for j in 0..n_params {
            neg_elbo_grad.push(-grad_mu[j]);
        }
        for j in 0..n_params {
            neg_elbo_grad.push(-grad_log_sigma[j]);
        }

        // Adam step
        adam.step(&mut var_params, &neg_elbo_grad);

        // Clamp log_sigma to prevent collapse or explosion
        for i in 0..n_params {
            var_params[n_params + i] = var_params[n_params + i].clamp(-6.0, 2.0);
        }

        if step % 100 == 0 || step == n_steps - 1 {
            let sigmas: Vec<f64> = (0..n_params).map(|i| var_params[n_params + i].exp()).collect();
            eprintln!(
                "  [{}] step {:>4}/{}: ELBO = {:.4}, sigma = {:?}",
                model.name(),
                step,
                n_steps,
                final_elbo,
                sigmas,
            );
        }
    }

    let mu = var_params[..n_params].to_vec();
    // Apply sigma inflation to calibrate mean-field VI posteriors.
    // Adding ln(factor) to log_sigma is equivalent to multiplying sigma by factor.
    let log_inflation = SIGMA_INFLATION_FACTOR.ln();
    let log_sigma: Vec<f64> = var_params[n_params..]
        .iter()
        .map(|ls| ls + log_inflation)
        .collect();

    SviFitResult {
        model,
        mu,
        log_sigma,
        elbo: final_elbo,
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct BandFitData {
    times: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
    /// true = upper limit (SNR < 3); false = detection
    is_upper: Vec<bool>,
    /// For upper limits: 3-sigma flux ceiling (normalized); for detections: unused (0.0)
    upper_flux: Vec<f64>,
    #[allow(dead_code)]
    noise_frac_median: f64,
    peak_flux_obs: f64,
}

fn flux_to_mag(flux: f64) -> f64 {
    -2.5 * flux.log10() + ZP
}

fn median_f64(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        Some((xs[mid - 1] + xs[mid]) / 2.0)
    } else {
        Some(xs[mid])
    }
}

// ---------------------------------------------------------------------------
// Posterior sampling and prediction
// ---------------------------------------------------------------------------

struct PosteriorPrediction {
    times: Vec<f64>,
    mags_median: Vec<f64>,
    mags_lower: Vec<f64>,   // 16th percentile
    mags_upper: Vec<f64>,   // 84th percentile
}

fn posterior_predict(
    result: &SviFitResult,
    times: &[f64],
    peak_flux: f64,
    n_posterior_samples: usize,
) -> PosteriorPrediction {
    let n_params = result.model.n_params();
    let n_times = times.len();

    // Draw posterior samples
    let mut all_mags: Vec<Vec<f64>> = vec![Vec::with_capacity(n_posterior_samples); n_times];

    for _ in 0..n_posterior_samples {
        // Sample parameters from q(theta) = N(mu, diag(sigma^2))
        let mut params = vec![0.0; n_params];
        for j in 0..n_params {
            let sigma = result.log_sigma[j].exp();
            let u1: f64 = rand::random::<f64>().max(1e-10);
            let u2: f64 = rand::random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            params[j] = result.mu[j] + sigma * z;
        }

        // Evaluate model at all times (batch for sequential models)
        let flux_norms = eval_model_batch(result.model, &params, times);
        for (ti, flux_norm) in flux_norms.iter().enumerate() {
            let flux_abs = flux_norm * peak_flux;
            let mag = flux_to_mag(flux_abs.max(1e-12));
            all_mags[ti].push(mag);
        }
    }

    // Compute percentiles
    let mut mags_median = Vec::with_capacity(n_times);
    let mut mags_lower = Vec::with_capacity(n_times);
    let mut mags_upper = Vec::with_capacity(n_times);

    for ti in 0..n_times {
        all_mags[ti].sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all_mags[ti].len();
        let p16 = (n as f64 * 0.16) as usize;
        let p50 = n / 2;
        let p84 = (n as f64 * 0.84) as usize;
        mags_median.push(all_mags[ti][p50.min(n - 1)]);
        mags_lower.push(all_mags[ti][p16.min(n - 1)]);
        mags_upper.push(all_mags[ti][p84.min(n - 1)]);
    }

    PosteriorPrediction {
        times: times.to_vec(),
        mags_median,
        mags_lower,
        mags_upper,
    }
}

// ---------------------------------------------------------------------------
// Plotting
// ---------------------------------------------------------------------------

struct BandPlot {
    times_obs: Vec<f64>,
    mags_obs: Vec<f64>,
    mag_errors: Vec<f64>,
    pred: PosteriorPrediction,
    label: String,
    legend_label: String,
}

fn plot_results(
    band_plots: &[BandPlot],
    output_path: &Path,
    _object_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if band_plots.is_empty() {
        return Ok(());
    }

    // Determine ranges
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    let mut mag_min = f64::INFINITY;
    let mut mag_max = f64::NEG_INFINITY;

    for b in band_plots {
        for &t in &b.times_obs {
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }
        for &m in b.mags_obs.iter().chain(b.pred.mags_median.iter()) {
            if m.is_finite() {
                mag_min = mag_min.min(m);
                mag_max = mag_max.max(m);
            }
        }
    }

    let mag_pad = (mag_max - mag_min) * 0.15;
    let y_top = (mag_max + mag_pad).min(25.0);
    let y_bottom = (mag_min - mag_pad).max(15.0);

    let colors: HashMap<&str, RGBColor> = [
        ("g", BLUE),
        ("r", RED),
        ("i", GREEN),
        ("ZTF_g", BLUE),
        ("ZTF_r", RED),
        ("ZTF_i", GREEN),
    ]
    .iter()
    .cloned()
    .collect();

    let root = BitMapBackend::new(output_path, (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .x_label_area_size(70)
        .y_label_area_size(90)
        .build_cartesian_2d(t_min..t_max, y_top..y_bottom)?;

    chart
        .configure_mesh()
        .x_desc("Time (days)")
        .y_desc("Magnitude")
        .x_label_style(("sans-serif", 24))
        .y_label_style(("sans-serif", 24))
        .draw()?;

    for b in band_plots {
        let color = colors.get(b.label.as_str()).unwrap_or(&BLACK);

        // Uncertainty band
        if !b.pred.mags_upper.is_empty() {
            let mut area: Vec<(f64, f64)> = Vec::new();
            for i in 0..b.pred.times.len() {
                if b.pred.mags_upper[i].is_finite() {
                    area.push((b.pred.times[i], b.pred.mags_upper[i]));
                }
            }
            for i in (0..b.pred.times.len()).rev() {
                if b.pred.mags_lower[i].is_finite() {
                    area.push((b.pred.times[i], b.pred.mags_lower[i]));
                }
            }
            chart.draw_series(std::iter::once(Polygon::new(area, color.mix(0.18).filled())))?;
        }

        // Model curve (posterior median)
        let model_pts: Vec<(f64, f64)> = b
            .pred
            .times
            .iter()
            .zip(b.pred.mags_median.iter())
            .filter(|(_, m)| m.is_finite())
            .map(|(t, m)| (*t, *m))
            .collect();
        chart
            .draw_series(LineSeries::new(model_pts, color.stroke_width(2)))?
            .label(b.legend_label.clone())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));

        // Error bars
        for ((t, m), err) in b.times_obs.iter().zip(b.mags_obs.iter()).zip(b.mag_errors.iter()) {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(*t, m - err), (*t, m + err)],
                color.stroke_width(1),
            )))?;
        }

        // Observations
        chart.draw_series(b.times_obs.iter().zip(b.mags_obs.iter()).map(|(t, m)| {
            Circle::new((*t, *m), 3, color.filled())
        }))?;
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .margin(20)
        .draw()?;

    root.present()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main processing
// ---------------------------------------------------------------------------

fn read_lightcurve(
    path: &str,
) -> Result<HashMap<String, (Vec<f64>, Vec<f64>, Vec<f64>)>, Box<dyn std::error::Error>> {
    let bands = read_ztf_lightcurve(path, false)?;
    let result = bands
        .into_iter()
        .map(|(filter, bd)| (filter, (bd.times, bd.mags, bd.errors)))
        .collect();
    Ok(result)
}

/// Result from fitting one band of one object.
struct BandFitOutput {
    object: String,
    band: String,
    model: SviModel,
    pso_params: Vec<f64>,
    pso_chi2: f64,
    svi_mu: Vec<f64>,
    svi_log_sigma: Vec<f64>,
    svi_elbo: f64,
    pso_time_s: f64,
    svi_time_s: f64,
    n_obs: usize,
    /// Reduced chi² in magnitude space: sum((mag_obs - mag_pred)² / mag_err²) / N
    mag_chi2: f64,
}

/// Fit a single file. Returns per-band results and optionally generates plots.
fn process_file(
    input_path: &str,
    output_dir: &Path,
    do_plot: bool,
    svi_lr: f64,
    svi_n_steps: usize,
    svi_n_samples: usize,
    retry_enabled: bool,
    force_model: Option<SviModel>,
) -> Result<Vec<BandFitOutput>, Box<dyn std::error::Error>> {
    let object_name = input_path
        .split('/')
        .last()
        .unwrap_or("unknown")
        .trim_end_matches(".csv");

    let bands_raw = read_lightcurve(input_path)?;
    if bands_raw.is_empty() {
        return Ok(vec![]);
    }

    // Prepare per-band data
    let mut band_entries: Vec<(String, BandFitData, Vec<f64>)> = Vec::new();
    for (band_name, (times, fluxes, flux_errs)) in bands_raw.iter() {
        if fluxes.is_empty() {
            continue;
        }
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 {
            continue;
        }
        let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f / peak_flux).collect();
        let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e / peak_flux).collect();

        // Flag upper limits: SNR < 3 in the original (un-normalized) flux
        let is_upper: Vec<bool> = fluxes.iter().zip(flux_errs.iter())
            .map(|(f, e)| *e > 0.0 && (*f / *e) < 3.0)
            .collect();
        // Upper limit ceiling = 3*sigma (normalized)
        let upper_flux: Vec<f64> = flux_errs.iter()
            .map(|e| 3.0 * e / peak_flux)
            .collect();

        let mut frac_noises: Vec<f64> = normalized_flux
            .iter()
            .zip(normalized_err.iter())
            .filter_map(|(f, e)| if *f > 0.0 { Some(e / f) } else { None })
            .collect();
        let noise_frac_median = median_f64(&mut frac_noises).unwrap_or(0.0);

        let mags_obs: Vec<f64> = fluxes.iter().map(|f| flux_to_mag(*f)).collect();

        let fit_data = BandFitData {
            times: times.clone(),
            flux: normalized_flux,
            flux_err: normalized_err,
            is_upper,
            upper_flux,
            noise_frac_median,
            peak_flux_obs: peak_flux,
        };

        band_entries.push((band_name.clone(), fit_data, mags_obs));
    }

    band_entries.sort_by(|a, b| b.1.times.len().cmp(&a.1.times.len()));

    // Build time grid (for plots)
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for (_, data, _) in &band_entries {
        for &t in &data.times {
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }
    }
    let n_pred = 200;
    let duration = (t_max - t_min).max(1.0);
    let times_pred: Vec<f64> = (0..n_pred)
        .map(|i| t_min + (i as f64) * duration / (n_pred - 1) as f64)
        .collect();

    let mut band_plots: Vec<BandPlot> = Vec::new();
    let mut results: Vec<BandFitOutput> = Vec::new();

    for (band_name, data, mags_obs) in &band_entries {
        let fit_start = Instant::now();

        // Step 1: PSO model selection (cascade or forced model)
        let (pso_model, pso_params, pso_chi2) = if let Some(forced) = force_model {
            // Run PSO for just the forced model
            let (lower, upper) = pso_bounds(forced);
            let problem = PsoCost {
                times: data.times.clone(),
                flux: data.flux.clone(),
                flux_err: data.flux_err.clone(),
                is_upper: data.is_upper.clone(),
                upper_flux: data.upper_flux.clone(),
                model: forced,
            };
            let solver = ParticleSwarm::new((lower, upper), 40);
            match Executor::new(problem, solver)
                .configure(|state| state.max_iters(50))
                .run()
            {
                Ok(res) => {
                    let chi2 = res.state().get_cost();
                    let params = res.state().get_best_param().unwrap().position.clone();
                    (forced, params, chi2)
                }
                Err(_) => (forced, vec![], f64::INFINITY),
            }
        } else {
            pso_model_select(data)
        };
        let pso_time = fit_start.elapsed().as_secs_f64();

        // Skip band if PSO failed to find any valid params
        if pso_params.is_empty() {
            eprintln!("  [{}] PSO returned no params, skipping band", band_name);
            continue;
        }

        // Step 2: SVI refinement (with optional retry)
        let svi_start = Instant::now();
        let mut svi_result = svi_fit(pso_model, data, svi_n_steps, svi_n_samples, svi_lr, Some(&pso_params));

        if retry_enabled && svi_result.elbo < -1000.0 {
            // Catastrophic failure — retry with conservative settings
            let retry_lr = if svi_lr > 0.005 { 0.005 } else { svi_lr * 2.0 };
            let retry_result = svi_fit(pso_model, data, svi_n_steps, svi_n_samples.max(8), retry_lr, Some(&pso_params));
            if retry_result.elbo > svi_result.elbo {
                svi_result = retry_result;
            }
        }
        let svi_time = svi_start.elapsed().as_secs_f64();

        // Compute reduced chi² in magnitude space using SVI posterior mean
        let svi_preds = eval_model_batch(pso_model, &svi_result.mu, &data.times);
        let mut mag_chi2_sum = 0.0;
        let mut mag_chi2_n = 0usize;
        for i in 0..data.times.len() {
            let pred_flux = svi_preds[i] * data.peak_flux_obs;
            if pred_flux > 0.0 && data.flux[i] > 0.0 {
                let mag_pred = flux_to_mag(pred_flux);
                let mag_obs = mags_obs[i];
                let mag_err = 1.0857 * data.flux_err[i] / data.flux[i];
                if mag_err > 0.0 {
                    let residual = mag_obs - mag_pred;
                    mag_chi2_sum += residual * residual / (mag_err * mag_err);
                    mag_chi2_n += 1;
                }
            }
        }
        let mag_chi2 = if mag_chi2_n > 0 { mag_chi2_sum / mag_chi2_n as f64 } else { f64::NAN };

        results.push(BandFitOutput {
            object: object_name.to_string(),
            band: band_name.clone(),
            model: pso_model,
            pso_params: pso_params.clone(),
            pso_chi2,
            svi_mu: svi_result.mu.clone(),
            svi_log_sigma: svi_result.log_sigma.clone(),
            svi_elbo: svi_result.elbo,
            pso_time_s: pso_time,
            svi_time_s: svi_time,
            n_obs: data.times.len(),
            mag_chi2,
        });

        if do_plot {
            let pred = posterior_predict(&svi_result, &times_pred, data.peak_flux_obs, 200);
            let mag_errors: Vec<f64> = data
                .flux_err
                .iter()
                .zip(data.flux.iter())
                .map(|(err, flux)| if *flux > 0.0 { 1.0857 * err / flux } else { 0.1 })
                .collect();
            let legend_label = format!(
                "{} SVI-{} (ELBO={:.1}; N={})",
                band_name, svi_result.model.name(), svi_result.elbo, data.times.len()
            );
            band_plots.push(BandPlot {
                times_obs: data.times.clone(),
                mags_obs: mags_obs.clone(),
                mag_errors,
                pred,
                label: band_name.clone(),
                legend_label,
            });
        }
    }

    // Write per-object CSV
    let mut csv_rows = Vec::new();
    for r in &results {
        let param_names = r.model.param_names();
        for (i, name) in param_names.iter().enumerate() {
            let sigma = r.svi_log_sigma[i].exp();
            csv_rows.push(format!(
                "{},{},{},{},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4}",
                r.object, r.band, r.model.name(), name,
                r.pso_params[i], r.svi_mu[i], sigma, r.pso_chi2, r.svi_elbo, r.mag_chi2,
            ));
        }
    }
    let csv_path = output_dir.join(format!("{}_svi_params.csv", object_name));
    let mut csv_content =
        String::from("object,band,model,param,pso_value,svi_mean,svi_std,pso_chi2,svi_elbo,mag_chi2\n");
    for row in &csv_rows {
        csv_content.push_str(row);
        csv_content.push('\n');
    }
    fs::write(&csv_path, csv_content)?;

    if do_plot && !band_plots.is_empty() {
        let output_path = output_dir.join(format!("{}.png", object_name));
        plot_results(&band_plots, &output_path, object_name)?;
    }

    Ok(results)
}

/// Plot PSO vs SVI comparison scatter: one subplot per parameter name.
fn plot_comparison(
    all_results: &[BandFitOutput],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Collect unique parameter names across all models
    let param_order: Vec<&str> = vec![
        "log_a", "b", "beta", "alpha", "alpha1", "alpha2", "n", "log_gamma", "t0",
        "log_tau_rise", "log_tau_fall", "log_tau_m", "log_tau_sd",
        "log_tau_diff", "log_tau_tr", "log_t_b", "logit_f", "log_sigma_extra",
        "log10_mej", "log10_vej", "log10_kappa_r",
    ];

    // Gather (pso_val, svi_mean, svi_std) per parameter name
    let mut param_data: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new();
    for r in all_results {
        let names = r.model.param_names();
        for (i, name) in names.iter().enumerate() {
            let sigma = r.svi_log_sigma[i].exp();
            let pso = r.pso_params[i];
            let svi = r.svi_mu[i];
            if pso.is_finite() && svi.is_finite() && sigma.is_finite() {
                param_data
                    .entry(name.to_string())
                    .or_default()
                    .push((pso, svi, sigma));
            }
        }
    }

    // Only plot params that appear
    let params_to_plot: Vec<&str> = param_order
        .iter()
        .filter(|p| param_data.contains_key(**p))
        .copied()
        .collect();
    let n_panels = params_to_plot.len();
    if n_panels == 0 {
        return Ok(());
    }

    let cols = 3usize;
    let rows = (n_panels + cols - 1) / cols;
    let w = (cols as u32) * 500;
    let h = (rows as u32) * 450;

    let root = BitMapBackend::new(output_path, (w, h)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((rows, cols));

    for (idx, &pname) in params_to_plot.iter().enumerate() {
        let pts = &param_data[pname];
        // Determine axis range from data
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &(pso, svi, sigma) in pts {
            lo = lo.min(pso).min(svi - sigma);
            hi = hi.max(pso).max(svi + sigma);
        }
        let pad = (hi - lo).max(0.1) * 0.1;
        lo -= pad;
        hi += pad;

        let panel = &panels[idx];
        let mut chart = ChartBuilder::on(panel)
            .margin(8)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .caption(pname, ("sans-serif", 20))
            .build_cartesian_2d(lo..hi, lo..hi)?;

        chart
            .configure_mesh()
            .x_desc("PSO")
            .y_desc("SVI mean")
            .x_label_style(("sans-serif", 14))
            .y_label_style(("sans-serif", 14))
            .draw()?;

        // Identity line
        chart.draw_series(LineSeries::new(
            vec![(lo, lo), (hi, hi)],
            BLACK.stroke_width(1),
        ))?;

        // Error bars (SVI std)
        for &(pso, svi, sigma) in pts {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(pso, svi - sigma), (pso, svi + sigma)],
                RGBColor(100, 100, 100).stroke_width(1),
            )))?;
        }

        // Scatter points
        chart.draw_series(pts.iter().map(|&(pso, svi, _)| {
            Circle::new((pso, svi), 2, BLUE.filled())
        }))?;
    }

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let do_plot = !args.iter().any(|a| a == "--no-plot");
    let verbose = !args.iter().any(|a| a == "--quiet");

    // Parse --lr=0.01 (learning rate for SVI)
    let svi_lr: f64 = args.iter()
        .find_map(|a| a.strip_prefix("--lr=").and_then(|v| v.parse().ok()))
        .unwrap_or(0.01);

    // Parse --n-steps=1000 (SVI optimization steps)
    let svi_n_steps: usize = args.iter()
        .find_map(|a| a.strip_prefix("--n-steps=").and_then(|v| v.parse().ok()))
        .unwrap_or(1000);

    // Parse --n-samples=4 (MC samples per SVI step)
    let svi_n_samples: usize = args.iter()
        .find_map(|a| a.strip_prefix("--n-samples=").and_then(|v| v.parse().ok()))
        .unwrap_or(4);

    // Parse --retry (retry with different settings if ELBO is catastrophic)
    let retry_enabled = args.iter().any(|a| a == "--retry");

    // Parse --force-model=Tde (skip PSO cascade, use this model for all bands)
    let force_model: Option<SviModel> = args.iter()
        .find_map(|a| a.strip_prefix("--force-model="))
        .and_then(|name| match name {
            "Bazin" => Some(SviModel::Bazin),
            "Villar" => Some(SviModel::Villar),
            "MetzgerKN" => Some(SviModel::MetzgerKN),
            "Tde" => Some(SviModel::Tde),
            "Arnett" => Some(SviModel::Arnett),
            "Magnetar" => Some(SviModel::Magnetar),
            "ShockCooling" => Some(SviModel::ShockCooling),
            "Afterglow" => Some(SviModel::Afterglow),
            _ => None,
        });

    let mut targets: Vec<String> = Vec::new();
    for arg in args.iter().skip(1) {
        if arg.starts_with("--") { continue; }
        if arg.ends_with(".csv") {
            targets.push(arg.clone());
        } else if Path::new(arg).is_dir() {
            // Scan directory for CSV files
            if let Ok(entries) = fs::read_dir(arg) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                            if let Some(s) = path.to_str() {
                                targets.push(s.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    if targets.is_empty() {
        let dir = Path::new("lightcurves_csv");
        if !dir.exists() {
            eprintln!("Usage: fit_svi_lightcurves [--no-plot] [--quiet] [file.csv ... | dir]");
            eprintln!("  With no files, processes all CSVs in lightcurves_csv/");
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

    let output_dir = Path::new("svi_plots");
    fs::create_dir_all(output_dir)?;

    let mut all_results: Vec<BandFitOutput> = Vec::new();
    let mut n_success = 0usize;
    let mut n_bands_total = 0usize;
    let mut total_pso_time = 0.0f64;
    let mut total_svi_time = 0.0f64;

    let total_start = Instant::now();
    for (idx, t) in targets.iter().enumerate() {
        if verbose {
            eprint!("\r[{}/{}] {}", idx + 1, targets.len(), t);
        }
        match process_file(t, output_dir, do_plot, svi_lr, svi_n_steps, svi_n_samples, retry_enabled, force_model) {
            Ok(band_results) => {
                if !band_results.is_empty() {
                    n_success += 1;
                }
                for r in &band_results {
                    total_pso_time += r.pso_time_s;
                    total_svi_time += r.svi_time_s;
                    n_bands_total += 1;
                }
                all_results.extend(band_results);
            }
            Err(e) => {
                if verbose {
                    eprintln!("\nError processing {}: {}", t, e);
                }
            }
        }
    }
    let total_elapsed = total_start.elapsed().as_secs_f64();
    if verbose {
        eprintln!();
    }

    // Write global CSV
    let global_csv_path = output_dir.join("all_svi_params.csv");
    let mut csv = String::from(
        "object,band,model,param,pso_value,svi_mean,svi_std,pso_chi2,svi_elbo,mag_chi2\n",
    );
    for r in &all_results {
        let names = r.model.param_names();
        for (i, name) in names.iter().enumerate() {
            let sigma = r.svi_log_sigma[i].exp();
            csv.push_str(&format!(
                "{},{},{},{},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4}\n",
                r.object, r.band, r.model.name(), name,
                r.pso_params[i], r.svi_mu[i], sigma, r.pso_chi2, r.svi_elbo, r.mag_chi2,
            ));
        }
    }
    fs::write(&global_csv_path, csv)?;

    // Comparison plot
    if !all_results.is_empty() {
        let cmp_path = output_dir.join("pso_vs_svi_comparison.png");
        plot_comparison(&all_results, &cmp_path)?;
        println!("  Comparison plot: {}", cmp_path.display());
    }

    // Throughput analysis
    println!("\n=== Throughput Analysis ===");
    println!("  Objects processed: {}/{}", n_success, targets.len());
    println!("  Bands fitted:     {}", n_bands_total);
    println!();
    println!("  Total wall time:   {:.2}s", total_elapsed);
    println!("  PSO time (sum):    {:.2}s", total_pso_time);
    println!("  SVI time (sum):    {:.2}s", total_svi_time);
    println!("  Fit time (sum):    {:.2}s  (PSO+SVI, excludes I/O and plotting)", total_pso_time + total_svi_time);
    println!();
    if n_success > 0 {
        let fit_time = total_pso_time + total_svi_time;
        println!("  Objects/sec (wall):   {:.1}", n_success as f64 / total_elapsed);
        println!("  Objects/sec (fit):    {:.1}", n_success as f64 / fit_time);
        println!("  Bands/sec (fit):      {:.1}", n_bands_total as f64 / fit_time);
        println!("  Avg PSO/band:         {:.4}s", total_pso_time / n_bands_total as f64);
        println!("  Avg SVI/band:         {:.4}s", total_svi_time / n_bands_total as f64);
        println!("  Avg total/band:       {:.4}s", fit_time / n_bands_total as f64);
    }
    println!();
    println!("  Global CSV: {}", global_csv_path.display());

    Ok(())
}
