import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="file to plot")
parser.add_argument("-file", type=str)

file_to_plot = f"{parser.parse_args().file}"

if __name__ == "__main__":
    with open(file_to_plot, "r") as file:
        data = json.load(file)

    # Combine present + previous photometry
    photometry = [data["candidate"]] + data.get("prv_candidates", [])

    # --- Build arrays ---
    jd = [x["jd"] for x in photometry]
    mag = [x["magpsf"] for x in photometry]
    mag_err = [x.get("sigmapsf", None) for x in photometry]
    band = [x["band"] for x in photometry]

    # Unique bands for color grouping
    unique_bands = sorted(set(band))

    # Colormap per band
    colors = {
        "g": "green",
        "r": "red",
        "i": "yellow",
        "u": "blue",
        "z": "black",
        "y": "orange",
    }

    plt.figure(figsize=(9, 6))

    # --- Plot band by band ---
    for b in unique_bands:
        jd_b = [jd[i] for i in range(len(jd)) if band[i] == b]
        mag_b = [mag[i] for i in range(len(mag)) if band[i] == b]
        err_b = [mag_err[i] for i in range(len(mag_err)) if band[i] == b]

        plt.errorbar(
            jd_b,
            mag_b,
            yerr=err_b,
            fmt="o",
            ms=6,
            capsize=2,
            label=f"{b}-band",
            color=colors.get(b, "gray"),
        )

    # --- Plot settings ---
    plt.gca().invert_yaxis()  # astronomy magnitudes: brighter = smaller mag
    plt.xlabel("JD")
    plt.ylabel("Magnitude (PSF)")
    plt.title("ZTF Lightcurve")
    plt.legend()
    plt.grid(True)

    plt.show()
