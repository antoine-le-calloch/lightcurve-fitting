import json
import pandas as pd
import os

# Load JSON
with open("photometry.json", "r") as f:
    data = json.load(f)

# Output directory
outdir = "lightcurves_csv"
os.makedirs(outdir, exist_ok=True)

for obj_name, obj_data in data.items():
    phot = obj_data.get("photometry", [])
    if len(phot) == 0:
        print(f"Skipping {obj_name} (no photometry)")
        continue

    # Build DataFrame
    df = pd.DataFrame(phot, columns=["mjd", "flux", "flux_err", "filter"])

    # Convert to numeric
    df["mjd"] = df["mjd"].astype(float)
    df["flux"] = df["flux"].astype(float)
    df["flux_err"] = df["flux_err"].astype(float)

    # Optional: sort by time
    df = df.sort_values("mjd")

    # Write CSV
    outpath = os.path.join(outdir, f"{obj_name}.csv")
    df.to_csv(outpath, index=False)

    print(f"Wrote {outpath}")

