# Lightcurve Fitting – Lightweight Enrichment Worker

This repository provides a lightweight version of Boom’s enrichment worker, focused on loading ZTF candidate data,
computing basic photometric properties, and printing the results to the console.
It is designed as a simplified environment for experimentation, debugging, and rapid iteration on light-curve metrics.

## Running the project

All candidate JSON files should be placed inside a candidates/ folder.
To process all candidates, simply run:

```
cargo run
```

This will:
- Load each .json file in candidates/
- Deserialize the ZTF candidate
- Compute the photometric properties
- Print all calculated metrics to standard output

## Input data format

Each JSON file must contain a ZTF alert/candidate in the expected schema (same as Boom’s ZTF format).

## Photometric analysis

The core computations are implemented in `src/lightcurves.rs → analyze_photometry()`

This function handles:
- Filtering photometry points
- Converting magnitudes
- Computing rising/fading rates
- Computing per-band properties
- Building AllBandsProperties and PerBandProperties