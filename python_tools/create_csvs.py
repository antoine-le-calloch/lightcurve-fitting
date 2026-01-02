import json
import pandas as pd
import os

with open("photometry.json", "r") as f:
    data = json.load(f)

outdir = "lightcurves_csv"
os.makedirs(outdir, exist_ok=True)

catalog_rows = []

for obj_name, obj_data in data.items():

    classification = obj_data.get("classification", "")
    probability = obj_data.get("probability", None)

    # Save to catalog
    catalog_rows.append({
        "object": obj_name,
        "classification": classification,
        "probability": probability
    })

    # Light curve CSV
    phot = obj_data.get("photometry", [])
    if len(phot) == 0:
        continue

    df = pd.DataFrame(phot, columns=["mjd", "flux", "flux_err", "filter"])
    df["mjd"] = df["mjd"].astype(float)
    df["flux"] = df["flux"].astype(float)
    df["flux_err"] = df["flux_err"].astype(float)
    df = df.sort_values("mjd")

    df.to_csv(os.path.join(outdir, f"{obj_name}.csv"), index=False)

# Write master classification catalog
catalog_df = pd.DataFrame(catalog_rows)
catalog_df.to_csv("classifications.csv", index=False)

print("Wrote per-object light curves to:", outdir)
print("Wrote classification catalog to fusion_classifications.csv")

