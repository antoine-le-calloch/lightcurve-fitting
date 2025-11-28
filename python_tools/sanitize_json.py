import json
import os

def sanitize_json_file(file_path):
    # Remove entries in 'fp_hists' that do not contain 'magpsf' key
    with open(file_path, "r") as file:
        candidate = json.load(file)

    sanitize_fp_hists = []
    for fp in candidate["fp_hists"]:
        if "magpsf" in fp:
            sanitize_fp_hists.append(fp)

    candidate["fp_hists"] = sanitize_fp_hists
    with open(file_path, "w") as file:
        json.dump(candidate, file, indent=2)

if __name__ == "__main__":
    # Sanitize all candidates json files
    folder = "candidates"
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            file_path_to_sanitize = os.path.join(folder, filename)
            sanitize_json_file(file_path_to_sanitize)

    print("Sanitization complete.")