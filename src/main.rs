use std::fs::File;
use serde_json::from_reader;
use crate::models::{PerBandProperties, AllBandsProperties};
use crate::ztf_enrichment::{get_alert_properties, ZtfAlertForEnrichment};

mod ztf_enrichment;
mod models;
mod lightcurves;

fn load_ztf_candidate(path: &str) -> anyhow::Result<ZtfAlertForEnrichment> {
    let file = File::open(path)?;
    let alert: ZtfAlertForEnrichment = from_reader(file)?;
    Ok(alert)
}

fn print_properties(photstats: &PerBandProperties, all_bands_props: &AllBandsProperties, path: &std::path::Path) {
    println!("-----------------------------------");
    println!("File: {:?}", path);
    println!("-----------------------------------");
    println!("All bands props");
    println!("-----------------------------------");
    println!("Peak:");
    println!("  peak_jd      : {:.5}", all_bands_props.peak_jd);
    println!("  peak_mag     : {:.3}", all_bands_props.peak_mag);
    println!("  peak_mag_err : {:.3}", all_bands_props.peak_mag_err);
    println!("  peak_band    : {:?}", all_bands_props.peak_band);
    println!("Faintest:");
    println!("  faintest_jd  : {:.5}", all_bands_props.faintest_jd);
    println!("  faintest_mag : {:.3}", all_bands_props.faintest_mag);
    println!();


    let band_list = [
        ("g", &photstats.g),
        ("r", &photstats.r),
        ("i", &photstats.i),
        ("z", &photstats.z),
        ("y", &photstats.y),
        ("u", &photstats.u),
    ];

    for (name, props) in band_list {
        if let Some(bp) = props {
            println!("-----------------------------");
            println!("Band: {}", name);
            println!("-----------------------------");

            println!("Peak:");
            println!("  peak_jd      : {:.5}", bp.peak_jd);
            println!("  peak_mag     : {:.3}", bp.peak_mag);
            println!("  peak_mag_err : {:.3}", bp.peak_mag_err);

            if let Some(rising) = &bp.rising {
                println!("Rising:");
                println!("  rate    : {:.3}", rising.rate);
                println!("  r²      : {:.3}", rising.r_squared);
                println!("  nb_data : {}", rising.nb_data);
            }

            if let Some(fading) = &bp.fading {
                println!("Fading:");
                println!("  rate    : {:.3}", fading.rate);
                println!("  r²      : {:.3}", fading.r_squared);
                println!("  nb_data : {}", fading.nb_data);
            }

            println!();
        }
    }
}

fn main() {
    for entry in std::fs::read_dir("candidates/").expect("Failed to read candidates directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        // process only json files
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
            let alert = load_ztf_candidate(path.to_str().unwrap()).expect("Failed to load ZTF candidate");
            let (alert_props, all_bands_props, _, _) = get_alert_properties(&alert);

            print_properties(&alert_props.photstats, &all_bands_props, &path);
        }
    }
}
