import json
import random
import argparse
from pathlib import Path
from alerce.core import Alerce

parser = argparse.ArgumentParser(description="Download Data")
parser.add_argument("-n", type=int, default=100)
parser.add_argument("-event", type=str, default="SNIa")

output_path = Path("candidates")
output_path.mkdir(parents=True, exist_ok=True)

alerce = Alerce()

event_type = parser.parse_args().event

num_to_download = parser.parse_args().n
downloaded = 0


def get_band(fid):
    if fid == 1:
        return "g"
    if fid == 2:
        return "r"
    if fid == 3:
        return "i"
    return "g"


def make_candidate_struct(d):
    return {
        "jd": d.get("mjd") + 2400000.5,
        "fid": d.get("fid"),
        "pid": d.get("pid"),
        "diffmaglim": d.get("diffmaglim"),
        "programpi": d.get("programpi"),
        "programid": d.get("programid", 1),
        "candid": int(d.get("candid")),
        "isdiffpos": d.get("isdiffpos", 1) == 1 or d.get("isdiffpos") == "t",
        "ra": d.get("ra", 0.0),
        "dec": d.get("dec", 0.0),
        "magpsf": d.get("magpsf", 0.0),
        "sigmapsf": d.get("sigmapsf", 0.0),
        "ranr": d.get("ranr", 0.0),
        "decnr": d.get("decnr", 0.0),
        "ndethist": d.get("ndethist", 0),
        "ncovhist": d.get("ncovhist", 0),
        "nmtchps": d.get("nmtchps", 0),
        "nid": d.get("nid"),
        "rcid": d.get("rcid"),
        "field": d.get("field"),
        "chipsf": d.get("chipsf"),
        "magap": d.get("magap"),
        "sigmagap": d.get("sigmagap"),
        "distnr": d.get("distnr"),
        "magnr": d.get("magnr"),
        "sigmagnr": d.get("sigmagnr"),
        "chinr": d.get("chinr"),
        "sharpnr": d.get("sharpnr"),
        "sky": d.get("sky"),
        "fwhm": d.get("fwhm"),
        "classtar": d.get("classtar"),
        "mindtoedge": d.get("mindtoedge"),
        "seeratio": d.get("seeratio"),
        "aimage": d.get("aimage"),
        "bimage": d.get("bimage"),
        "elong": d.get("elong"),
        "nneg": d.get("nneg"),
        "nbad": d.get("nbad"),
        "rb": d.get("rb"),
        "ssdistnr": d.get("ssdistnr"),
        "ssmagnr": d.get("ssmagnr"),
        "ssnamenr": d.get("ssnamenr"),
        "sgmag1": d.get("sgmag1"),
        "srmag1": d.get("srmag1"),
        "simag1": d.get("simag1"),
        "szmag1": d.get("szmag1"),
        "sgscore1": d.get("sgscore1"),
        "distpsnr1": d.get("distpsnr1"),
        "jdstarthist": d.get("jdstarthist"),
        "scorr": d.get("scorr"),
        "sgmag2": d.get("sgmag2"),
        "srmag2": d.get("srmag2"),
        "simag2": d.get("simag2"),
        "szmag2": d.get("szmag2"),
        "sgscore2": d.get("sgscore2"),
        "distpsnr2": d.get("distpsnr2"),
        "sgmag3": d.get("sgmag3"),
        "srmag3": d.get("srmag3"),
        "simag3": d.get("simag3"),
        "szmag3": d.get("szmag3"),
        "sgscore3": d.get("sgscore3"),
        "distpsnr3": d.get("distpsnr3"),
        "dsnrms": d.get("dsnrms"),
        "ssnrms": d.get("ssnrms"),
        "dsdiff": d.get("dsdiff"),
        "magzpsci": d.get("magzpsci"),
        "magzpsciunc": d.get("magzpsciunc"),
        "magzpscirms": d.get("magzpscirms"),
        "zpmed": d.get("zpmed"),
        "exptime": d.get("exptime"),
        "drb": d.get("drb"),
        "clrcoeff": d.get("clrcoeff"),
        "clrcounc": d.get("clrcounc"),
        "neargaia": d.get("neargaia"),
        "maggaia": d.get("maggaia"),
        "neargaiabright": d.get("neargaiabright"),
        "maggaiabright": d.get("maggaiabright"),
    }


def make_photometry_mag(d):
    return {
        "jd": d.get("mjd", 0.0) + 2400000.5,
        "magpsf": d.get("magpsf", 0.0),
        "sigmapsf": d.get("sigmapsf", 0.0),
        "band": get_band(d.get("fid", 1)),
    }


def make_fp_hist(d):
    return {
        "jd": d.get("mjd", 0.0) + 2400000.5,
        "magpsf": d.get("diffmaglim", 0.0),
        "sigmapsf": 0.0,
        "band": get_band(d.get("fid", 1)),
    }


while downloaded < num_to_download:
    random_page = random.randint(1, 50)

    params = {
        "classifier": "lc_classifier",
        "class_name": event_type,
        "probability": 0.7,
        "survey": "ztf",
        "page_size": 1,
        "page": random_page,
        "format": "json",
    }

    try:
        objects = alerce.query_objects(**params)

        if isinstance(objects, dict):
            objects = objects.get("items", [])

        for obj in objects:
            if downloaded >= num_to_download:
                break

            obj_id = obj["oid"]
            detections = alerce.query_detections(obj_id, survey="ztf", format="json")

            if not detections:
                continue

            detections.sort(key=lambda x: x.get("mjd", 0.0), reverse=True)
            history_dets = detections[1:]
            detection = detections[0]

            candidate_core = make_candidate_struct(detection)
            sigmapsf = detection.get("sigmapsf", 0.0)
            snr = 1.0857 / sigmapsf if sigmapsf > 0 else 0.0

            non_detections = alerce.query_non_detections(
                obj_id, survey="ztf", format="json"
            )
            if not non_detections:
                non_detections = []
            non_detections.sort(key=lambda x: x.get("mjd", 0.0), reverse=True)

            ztf_candidate_combined = {
                **candidate_core,
                "psfFlux": detection.get("magpsf", 0.0),
                "psfFluxErr": sigmapsf,
                "snr": snr,
                "band": get_band(detection.get("fid", 1)),
            }

            rust_data = {
                "_id": int(detection.get("candid")),
                "objectId": obj_id,
                "candidate": ztf_candidate_combined,
                "prv_candidates": [make_photometry_mag(d) for d in history_dets],
                "fp_hists": [make_fp_hist(d) for d in non_detections],
            }

            file_path = output_path / f"{event_type}_{obj_id}.json"

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(rust_data, f)

            downloaded += 1
            print(f"[{downloaded}/{num_to_download}] Saved {file_path}")

    except Exception as e:
        print(f"Failed to query: {e}")
