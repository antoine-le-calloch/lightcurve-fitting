#!/usr/bin/env python
"""
Simulation framework for validating lightcurve fitting pipelines.
Uses Redback physical models to generate synthetic ZTF-like lightcurves,
then runs both SVI (parametric) and GP (non-parametric) fitters and compares results.

Usage:
    conda activate origin
    python simulate_and_validate.py
"""

import numpy as np
import os
import csv
import subprocess
import sys
import json
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

# Redback imports
import redback
from redback.transient_models import kilonova_models, supernova_models, tde_models, afterglow_models

# ── ZTF band central frequencies (Hz) ──────────────────────────────────────
ZTF_FREQ = {
    "g": 6.28e14,
    "r": 4.71e14,
    "i": 3.83e14,
}

# Typical ZTF 5-sigma limiting flux (µJy) per band
ZTF_LIMIT_UJY = {"g": 3.0, "r": 3.5, "i": 5.0}

# Reference MJD (arbitrary recent epoch)
MJD0 = 60700.0

# Output directories
SIM_CSV_DIR = Path("simulated_lightcurves_csv")
SIM_PLOTS_DIR = Path("sim_validation")

# Working directory = script directory
SCRIPT_DIR = Path(__file__).parent


# ── Cadence generation ──────────────────────────────────────────────────────
def generate_ztf_cadence(
    duration_days,
    cadence_days=2.5,
    gap_prob=0.3,
    bands=("g", "r", "i"),
    band_probs=(0.35, 0.45, 0.20),
    rng=None,
):
    """Generate realistic ZTF-like observation times and bands."""
    if rng is None:
        rng = np.random.default_rng()

    observations = []
    t = rng.uniform(0.1, 1.0)  # first obs slightly after trigger
    while t < duration_days:
        # Skip this night with some probability (weather/moon)
        if rng.random() > gap_prob:
            # 1-3 observations this night across bands
            n_obs = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            night_bands = rng.choice(
                list(bands), size=n_obs, p=list(band_probs)
            )
            for b in night_bands:
                # Small intra-night time offset (minutes)
                dt = rng.uniform(0, 0.02)
                observations.append((t + dt, b))

        # Next night: cadence with jitter
        t += cadence_days + rng.normal(0, 0.5)

    return observations


# ── Noise model ─────────────────────────────────────────────────────────────
def add_ztf_noise(flux_ujy, band, rng=None):
    """
    Add realistic ZTF-like photometric noise.
    Returns (noisy_flux, flux_err).
    """
    if rng is None:
        rng = np.random.default_rng()

    limit = ZTF_LIMIT_UJY[band]
    # Error is combination of sky noise floor + Poisson-like signal noise
    signal_noise = np.abs(flux_ujy) * 0.05  # ~5% calibration noise
    sky_noise = limit * rng.uniform(0.8, 1.5)
    flux_err = np.sqrt(signal_noise**2 + sky_noise**2)
    noisy_flux = flux_ujy + rng.normal(0, flux_err)
    return noisy_flux, flux_err


# ── Model configurations ───────────────────────────────────────────────────
def get_model_configs():
    """
    Define simulation configurations for each transient class.
    Each config produces ~10 synthetic lightcurves with varied parameters.
    """
    configs = []

    # ─── FAST: Kilonovae (1-10 days) ───
    kn_base = dict(temperature_floor=3000.0)
    for i in range(50):
        rng = np.random.default_rng(seed=1000 + i)
        mej = 10 ** rng.uniform(-2, -0.7)  # 0.01 - 0.2 Msun
        vej = rng.uniform(0.1, 0.4)  # units of c
        kappa = 10 ** rng.uniform(0, 1.2)  # 1-15 cm²/g
        configs.append(
            dict(
                name=f"sim_kn_{i:03d}",
                cls="kilonova",
                model_func=kilonova_models.one_component_kilonova_model,
                redshift=rng.uniform(0.005, 0.05),
                duration=15.0,
                cadence=1.5,
                params=dict(mej=mej, vej=vej, kappa=kappa, **kn_base),
                true_params=dict(mej=mej, vej=vej, kappa=kappa),
            )
        )

    # ─── MEDIUM: Supernovae via Arnett (10-100 days) ───
    for i in range(50):
        rng = np.random.default_rng(seed=2000 + i)
        f_nickel = 10 ** rng.uniform(-2, -0.3)  # 0.01-0.5
        mej = 10 ** rng.uniform(-0.5, 1.5)  # 0.3-30 Msun
        vej = 10 ** rng.uniform(3.3, 4.3)  # 2000-20000 km/s
        kappa = rng.uniform(0.05, 0.4)
        kappa_gamma = 10 ** rng.uniform(-2, 2)
        configs.append(
            dict(
                name=f"sim_sn_{i:03d}",
                cls="supernova",
                model_func=supernova_models.arnett,
                redshift=rng.uniform(0.01, 0.1),
                duration=120.0,
                cadence=3.0,
                params=dict(
                    f_nickel=f_nickel,
                    mej=mej,
                    vej=vej,
                    kappa=kappa,
                    kappa_gamma=kappa_gamma,
                    temperature_floor=3000.0,
                ),
                true_params=dict(
                    f_nickel=f_nickel, mej=mej, vej=vej, kappa=kappa,
                ),
            )
        )

    # ─── SLOW: TDEs (30-500 days) ───
    # tde_analytical requires diffusion params in current redback version
    for i in range(50):
        rng = np.random.default_rng(seed=3000 + i)
        l0 = 10 ** rng.uniform(52, 56)
        t_0_turn = 10 ** rng.uniform(-2, 2)  # 0.01-100 days
        configs.append(
            dict(
                name=f"sim_tde_{i:03d}",
                cls="tde",
                model_func=tde_models.tde_analytical,
                redshift=rng.uniform(0.02, 0.2),
                duration=300.0,
                cadence=4.0,
                params=dict(
                    l0=l0,
                    t_0_turn=t_0_turn,
                    kappa=0.2,
                    kappa_gamma=1.0,
                    mej=1.0,
                    vej=1e4,
                    temperature_floor=3000.0,
                ),
                true_params=dict(l0=l0, t_0_turn=t_0_turn),
            )
        )

    # ─── Phenomenological: Bazin model (for direct SVI comparison) ───
    for i in range(50):
        rng = np.random.default_rng(seed=4000 + i)
        A = 10 ** rng.uniform(1, 3)  # amplitude in µJy
        b = rng.uniform(0, 5)  # baseline
        t0 = rng.uniform(5, 30)  # peak time
        tau_rise = 10 ** rng.uniform(0, 1.5)  # 1-30 days
        tau_fall = 10 ** rng.uniform(1, 2.5)  # 10-300 days
        configs.append(
            dict(
                name=f"sim_bazin_{i:03d}",
                cls="bazin_phenom",
                model_func=None,  # we'll generate analytically
                redshift=0.0,
                duration=150.0,
                cadence=3.0,
                params=dict(A=A, b=b, t0=t0, tau_rise=tau_rise, tau_fall=tau_fall),
                true_params=dict(A=A, b=b, t0=t0, tau_rise=tau_rise, tau_fall=tau_fall),
            )
        )

    # ─── GRB Afterglows (hours to months) ───
    for i in range(50):
        rng = np.random.default_rng(seed=5000 + i)
        thv = rng.uniform(0.0, 0.3)          # viewing angle [rad] (0 = on-axis)
        loge0 = rng.uniform(50, 54)           # log10 isotropic energy [erg]
        thc = rng.uniform(0.05, 0.3)          # jet core half-opening angle [rad]
        logn0 = rng.uniform(-3, 1)            # log10 ISM density [cm^-3]
        p = rng.uniform(2.05, 2.8)            # electron power-law index
        logepse = rng.uniform(-2, -0.3)       # log10 electron energy fraction
        logepsb = rng.uniform(-4, -1)         # log10 magnetic field energy fraction
        g0 = 10 ** rng.uniform(1.5, 3.0)      # initial Lorentz factor
        configs.append(
            dict(
                name=f"sim_ag_{i:03d}",
                cls="afterglow",
                model_func=afterglow_models.tophat_redback,
                redshift=rng.uniform(0.1, 1.0),
                duration=120.0,
                cadence=3.0,
                params=dict(
                    thv=thv, loge0=loge0, thc=thc,
                    logn0=logn0, p=p,
                    logepse=logepse, logepsb=logepsb,
                    g0=g0, xiN=1.0,
                ),
                true_params=dict(
                    thv=thv, loge0=loge0, thc=thc,
                    logn0=logn0, p=p,
                    logepse=logepse, logepsb=logepsb,
                    g0=g0,
                ),
            )
        )

    return configs


def bazin_flux(t, A, b, t0, tau_rise, tau_fall):
    """Evaluate Bazin model in flux space (µJy)."""
    dt = t - t0
    return A * np.exp(-dt / tau_fall) / (1.0 + np.exp(-dt / tau_rise)) + b


# ── Generate a single synthetic lightcurve ──────────────────────────────────
def generate_lightcurve(config, rng=None):
    """
    Generate a synthetic lightcurve for a given model configuration.
    Returns list of (mjd, flux_ujy, flux_err_ujy, band) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    obs = generate_ztf_cadence(
        config["duration"],
        cadence_days=config["cadence"],
        rng=rng,
    )

    rows = []
    for t_day, band in obs:
        freq = ZTF_FREQ[band]
        if config["model_func"] is not None:
            try:
                flux_mjy = config["model_func"](
                    np.array([t_day]),
                    redshift=config["redshift"],
                    frequency=np.array([freq]),
                    output_format="flux_density",
                    **config["params"],
                )
                flux_ujy = float(flux_mjy[0]) * 1000.0  # mJy -> µJy
            except Exception:
                continue
        else:
            flux_ujy = bazin_flux(t_day, **config["params"])

        if not np.isfinite(flux_ujy) or flux_ujy <= 0:
            continue

        noisy_flux, flux_err = add_ztf_noise(flux_ujy, band, rng=rng)
        mjd = MJD0 + t_day
        rows.append((mjd, noisy_flux, flux_err, band))

    return rows


# ── Write lightcurve CSV ────────────────────────────────────────────────────
def write_lightcurve_csv(rows, filepath):
    """Write lightcurve in ZTF CSV format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mjd", "flux", "flux_err", "filter"])
        for mjd, flux, flux_err, band in rows:
            writer.writerow([mjd, flux, flux_err, band])


# ── Run fitting pipelines ──────────────────────────────────────────────────
def run_svi_fitter(csv_dir):
    """Run SVI parametric fitter on simulated lightcurves."""
    print("\n=== Running SVI fitter ===")
    result = subprocess.run(
        [
            "cargo", "run", "--release",
            "--bin", "fit_svi_lightcurves",
            "--", "--quiet", str(csv_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"SVI STDERR: {result.stderr[-1000:]}")
    return result.returncode


def run_gp_fitter(csv_dir):
    """Run GP non-parametric fitter on simulated lightcurves."""
    print("\n=== Running GP fitter ===")
    result = subprocess.run(
        [
            "cargo", "run", "--release",
            "--bin", "fit_nonparametric_lightcurves_sklears",
            "--", str(csv_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"GP STDERR: {result.stderr[-1000:]}")
    return result.returncode


# ── Parse results ───────────────────────────────────────────────────────────
def parse_svi_results(plots_dir="svi_plots"):
    """Parse SVI CSV results from per-object files."""
    results = {}
    plots_path = SCRIPT_DIR / plots_dir
    if not plots_path.exists():
        return results

    for csv_file in plots_path.glob("sim_*_svi_params.csv"):
        obj_name = csv_file.stem.replace("_svi_params", "")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            bands = {}
            for row in reader:
                band = row["band"]
                if band not in bands:
                    bands[band] = {
                        "model": row["model"],
                        "pso_chi2": float(row["pso_chi2"]),
                        "svi_elbo": float(row["svi_elbo"]),
                        "mag_chi2": float(row.get("mag_chi2", "nan")),
                        "params": {},
                    }
                bands[band]["params"][row["param"]] = {
                    "mean": float(row["svi_mean"]),
                    "std": float(row["svi_std"]),
                }
            results[obj_name] = bands
    return results


def parse_gp_results(csv_file="gp_timescale_parameters_sklears.csv"):
    """Parse GP CSV results."""
    results = {}
    csv_path = SCRIPT_DIR / csv_file
    if not csv_path.exists():
        return results

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row["object"]
            if not obj.startswith("sim_"):
                continue  # skip real data results
            band = row["band"]
            if obj not in results:
                results[obj] = {}
            results[obj][band] = {
                "chi2": float(row.get("chi2", "nan")),
                "rise_time": float(row.get("rise_time_days", "nan")),
                "decay_time": float(row.get("decay_time_days", "nan")),
                "peak_mag": float(row.get("peak_mag", "nan")),
                "fwhm": float(row.get("fwhm_days", "nan")),
                "n_obs": int(row.get("n_obs", "0")),
            }
    return results


# ── Validation metrics ──────────────────────────────────────────────────────
def compute_validation_metrics(configs, svi_results, gp_results):
    """Compute and print validation metrics comparing GP and SVI."""

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    # Group by class
    classes = {}
    for cfg in configs:
        cls = cfg["cls"]
        if cls not in classes:
            classes[cls] = []
        classes[cls].append(cfg)

    all_svi_chi2 = []
    all_gp_chi2 = []
    class_stats = {}

    for cls, cfgs in sorted(classes.items()):
        print(f"\n--- {cls.upper()} ({len(cfgs)} objects) ---")
        svi_chi2_list = []
        gp_chi2_list = []
        svi_models = []

        for cfg in cfgs:
            name = cfg["name"]

            # SVI results
            if name in svi_results:
                for band, data in svi_results[name].items():
                    chi2 = data["mag_chi2"]
                    if np.isfinite(chi2):
                        svi_chi2_list.append(chi2)
                        all_svi_chi2.append(chi2)
                    svi_models.append(data["model"])

            # GP results
            if name in gp_results:
                for band, data in gp_results[name].items():
                    chi2 = data["chi2"]
                    if np.isfinite(chi2):
                        gp_chi2_list.append(chi2)
                        all_gp_chi2.append(chi2)

        # Print class-level stats
        if svi_chi2_list:
            svi_arr = np.array(svi_chi2_list)
            print(
                f"  SVI mag_chi2: median={np.median(svi_arr):.3f}, "
                f"mean={np.mean(svi_arr):.3f}, "
                f"<1: {np.mean(svi_arr < 1)*100:.0f}%, "
                f"<5: {np.mean(svi_arr < 5)*100:.0f}%"
            )
            model_counts = Counter(svi_models)
            model_str = ", ".join(
                f"{m}: {c}" for m, c in model_counts.most_common()
            )
            print(f"  SVI models: {model_str}")
        else:
            print("  SVI: no results")

        if gp_chi2_list:
            gp_arr = np.array(gp_chi2_list)
            print(
                f"  GP  chi2:     median={np.median(gp_arr):.3f}, "
                f"mean={np.mean(gp_arr):.3f}, "
                f"<1: {np.mean(gp_arr < 1)*100:.0f}%, "
                f"<5: {np.mean(gp_arr < 5)*100:.0f}%"
            )
        else:
            print("  GP: no results")

        class_stats[cls] = {
            "n_objects": len(cfgs),
            "svi_median_chi2": float(np.median(svi_chi2_list)) if svi_chi2_list else None,
            "gp_median_chi2": float(np.median(gp_chi2_list)) if gp_chi2_list else None,
            "svi_models": dict(Counter(svi_models)) if svi_models else {},
        }

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    if all_svi_chi2:
        svi = np.array(all_svi_chi2)
        print(
            f"SVI: {len(svi)} bands, median chi2={np.median(svi):.3f}, "
            f"<1: {np.mean(svi < 1)*100:.1f}%, <5: {np.mean(svi < 5)*100:.1f}%"
        )
    if all_gp_chi2:
        gp = np.array(all_gp_chi2)
        print(
            f"GP:  {len(gp)} bands, median chi2={np.median(gp):.3f}, "
            f"<1: {np.mean(gp < 1)*100:.1f}%, <5: {np.mean(gp < 5)*100:.1f}%"
        )

    return class_stats


def build_confusion_matrix(configs, svi_results):
    """
    Build and print a confusion matrix: true class (rows) vs SVI-selected model (columns).
    For each object, pick the model selected for the majority of bands.
    """
    # Map true class -> list of selected models (one per object)
    true_classes = ["kilonova", "supernova", "tde", "bazin_phenom", "afterglow"]
    model_names = ["Bazin", "Villar", "MetzgerKN", "Tde", "Arnett", "Magnetar", "ShockCooling", "Afterglow"]

    # For each object, determine the "dominant" model selected across bands
    obj_class = {}
    for cfg in configs:
        obj_class[cfg["name"]] = cfg["cls"]

    # Count: confusion[true_class][selected_model] = count
    confusion = {cls: Counter() for cls in true_classes}
    n_objects_per_class = Counter()

    for obj_name, bands in svi_results.items():
        cls = obj_class.get(obj_name)
        if cls is None:
            continue
        # Pick the best model per object: model with best (lowest) mag_chi2 across bands
        best_model = None
        best_chi2 = float("inf")
        for band, data in bands.items():
            chi2 = data["mag_chi2"]
            if np.isfinite(chi2) and chi2 < best_chi2:
                best_chi2 = chi2
                best_model = data["model"]
        if best_model is None:
            # Fallback: most common model across bands
            model_counts = Counter(d["model"] for d in bands.values())
            best_model = model_counts.most_common(1)[0][0]
        confusion[cls][best_model] += 1
        n_objects_per_class[cls] += 1

    # Print confusion matrix
    print(f"\n{'=' * 80}")
    print("CONFUSION MATRIX: True Class (rows) vs Selected SVI Model (columns)")
    print(f"{'=' * 80}")

    # Header
    col_width = 10
    header = f"{'True \\ Pred':<16}" + "".join(f"{m:>{col_width}}" for m in model_names) + f"{'Total':>{col_width}}"
    print(header)
    print("-" * len(header))

    for cls in true_classes:
        row = f"{cls:<16}"
        total = n_objects_per_class[cls]
        for m in model_names:
            count = confusion[cls].get(m, 0)
            row += f"{count:>{col_width}}"
        row += f"{total:>{col_width}}"
        print(row)

    print("-" * len(header))

    # Accuracy: fraction of objects where the "natural" model was selected
    # kilonova -> MetzgerKN, supernova -> Villar or Bazin, tde -> Tde, bazin_phenom -> Bazin
    natural_map = {
        "kilonova": ["MetzgerKN"],
        "supernova": ["Villar", "Bazin", "Arnett", "Magnetar", "ShockCooling"],
        "tde": ["Tde"],
        "bazin_phenom": ["Bazin"],
        "afterglow": ["Afterglow"],
    }
    total_correct = 0
    total_objects = 0
    print(f"\n{'Class':<16}{'Correct':>10}{'Total':>10}{'Accuracy':>10}")
    print("-" * 46)
    for cls in true_classes:
        correct = sum(confusion[cls].get(m, 0) for m in natural_map[cls])
        total = n_objects_per_class[cls]
        acc = correct / total * 100 if total > 0 else 0
        print(f"{cls:<16}{correct:>10}{total:>10}{acc:>9.1f}%")
        total_correct += correct
        total_objects += total

    overall_acc = total_correct / total_objects * 100 if total_objects > 0 else 0
    print("-" * 46)
    print(f"{'Overall':<16}{total_correct:>10}{total_objects:>10}{overall_acc:>9.1f}%")

    return confusion


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("Lightcurve Fitting Validation Framework")
    print("Using Redback physical models + ZTF-like synthetic observations")
    print("=" * 80)

    # Step 1: Generate synthetic lightcurves
    print("\n[1/4] Generating synthetic lightcurves...")
    sim_dir = SCRIPT_DIR / SIM_CSV_DIR
    sim_dir.mkdir(parents=True, exist_ok=True)

    configs = get_model_configs()
    truth_records = []

    n_success = 0
    for cfg in configs:
        rng = np.random.default_rng(seed=hash(cfg["name"]) % 2**32)
        rows = generate_lightcurve(cfg, rng=rng)
        if len(rows) < 5:
            print(f"  SKIP {cfg['name']}: only {len(rows)} observations")
            continue

        csv_path = sim_dir / f"{cfg['name']}.csv"
        write_lightcurve_csv(rows, csv_path)

        n_bands = len(set(r[3] for r in rows))
        flux_vals = [r[1] for r in rows if r[1] > 0]
        mag_range = ""
        if flux_vals:
            mag_bright = -2.5 * np.log10(max(flux_vals)) + 23.9
            mag_faint = -2.5 * np.log10(min(flux_vals)) + 23.9
            mag_range = f", mag {mag_bright:.1f}-{mag_faint:.1f}"
        print(
            f"  {cfg['name']}: {len(rows)} obs, {n_bands} bands, "
            f"class={cfg['cls']}{mag_range}"
        )
        n_success += 1

        truth_records.append(
            {
                "name": cfg["name"],
                "class": cfg["cls"],
                "redshift": cfg["redshift"],
                "true_params": cfg["true_params"],
                "n_obs": len(rows),
            }
        )

    print(f"\nGenerated {n_success}/{len(configs)} lightcurves in {SIM_CSV_DIR}/")

    # Save ground truth
    truth_path = sim_dir / "ground_truth.json"
    with open(truth_path, "w") as f:
        json.dump(truth_records, f, indent=2, default=str)
    print(f"Ground truth saved to {truth_path}")

    # Clean stale output files (GP appends so old data would contaminate)
    gp_csv = SCRIPT_DIR / "gp_timescale_parameters_sklears.csv"
    if gp_csv.exists():
        gp_csv.unlink()

    # Step 2: Run SVI fitter
    print("\n[2/4] Running SVI parametric fitter...")
    svi_rc = run_svi_fitter(SIM_CSV_DIR)

    # Step 3: Run GP fitter
    print("\n[3/4] Running GP non-parametric fitter...")
    gp_rc = run_gp_fitter(SIM_CSV_DIR)

    # Step 4: Compare results
    print("\n[4/4] Analyzing results...")
    svi_results = parse_svi_results()
    gp_results = parse_gp_results()

    print(f"  Parsed SVI results for {len(svi_results)} objects")
    print(f"  Parsed GP results for {len(gp_results)} objects")

    if not svi_results and not gp_results:
        print("No results to analyze. Check fitter output above.")
        return

    class_stats = compute_validation_metrics(configs, svi_results, gp_results)

    # Build confusion matrix (SVI only)
    if svi_results:
        confusion = build_confusion_matrix(configs, svi_results)

    # Save summary
    out_dir = SCRIPT_DIR / SIM_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(class_stats, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
