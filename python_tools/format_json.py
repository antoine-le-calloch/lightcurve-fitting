import json

# Remove entries in 'fp_hists' that do not contain 'magpsf' key
if __name__ == "__main__":
    json_file = "../candidates/ZTF25acdhetm.json"
    with open(json_file, "r") as file:
        candidate = json.load(file)

    sanitize_fp_hists = []
    for fp in candidate["fp_hists"]:
        if "magpsf" in fp:
            sanitize_fp_hists.append(fp)

    candidate["fp_hists"] = sanitize_fp_hists
    with open(json_file, "w") as file:
        json.dump(candidate, file, indent=2)