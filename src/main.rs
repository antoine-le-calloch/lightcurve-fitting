use std::fs::File;
use serde_json::from_reader;
use crate::models::{PerBandProperties, AllBandsProperties};
use crate::ztf_enrichment::{get_alert_properties, ZtfAlertForEnrichment};
use std::io::Write;

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

fn save_candidate_properties(candidate_names: &[String], photstats_vec: &[PerBandProperties], filename: &str) {
    let mut file = File::create(filename).expect("Failed to create CSV file");
    
    // Write header
    writeln!(file, "candidate_name,band,peak_jd,peak_mag,peak_mag_err,rising_rate,rising_r_squared,rising_nb_data,fading_rate,fading_r_squared,fading_nb_data").expect("Failed to write header");
    
    // Write data for each band
    for (photstats, candidate_name) in photstats_vec.iter().zip(candidate_names.iter()) {
        let bands = [
            ("g", &photstats.g),
            ("r", &photstats.r),
            ("i", &photstats.i),
            ("z", &photstats.z),
            ("y", &photstats.y),
            ("u", &photstats.u),
        ];
        
        for (band_name, band_props) in bands {
            if let Some(bp) = band_props {
                let rising_rate = bp.rising.as_ref().map(|r| r.rate.to_string()).unwrap_or_default();
                let rising_r2 = bp.rising.as_ref().map(|r| r.r_squared.to_string()).unwrap_or_default();
                let rising_nb = bp.rising.as_ref().map(|r| r.nb_data.to_string()).unwrap_or_default();
                
                let fading_rate = bp.fading.as_ref().map(|f| f.rate.to_string()).unwrap_or_default();
                let fading_r2 = bp.fading.as_ref().map(|f| f.r_squared.to_string()).unwrap_or_default();
                let fading_nb = bp.fading.as_ref().map(|f| f.nb_data.to_string()).unwrap_or_default();
                
                writeln!(file, "{},{},{},{},{},{},{},{},{},{},{}",
                    candidate_name, band_name, bp.peak_jd, bp.peak_mag, bp.peak_mag_err,
                    rising_rate, rising_r2, rising_nb,
                    fading_rate, fading_r2, fading_nb
                ).expect("Failed to write data");
            }
        }
    }
}

fn main() {
    let mut photstats_vec: Vec<PerBandProperties> = Vec::new();
    let mut photstatsul_vec: Vec<PerBandProperties> = Vec::new();
    let mut candidate_names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir("candidates/").expect("Failed to read candidates directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        // process only json files
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
            candidate_names.push(String::from(path.file_stem().unwrap().to_str().unwrap()));
            let alert = load_ztf_candidate(path.to_str().unwrap()).expect("Failed to load ZTF candidate");
            let (alert_props, alert_props_ul, all_bands_props, _, _) = get_alert_properties(&alert);

            print_properties(&alert_props.photstats, &all_bands_props, &path);
            println!("---------USING UPPER LIMITS--------");
            print_properties(&alert_props_ul.photstats, &all_bands_props, &path);
            photstats_vec.push(alert_props.photstats);
            photstatsul_vec.push(alert_props_ul.photstats);
        }
    }
    save_candidate_properties(&candidate_names, &photstats_vec, "candidate_properties.csv");
    save_candidate_properties(&candidate_names, &photstatsul_vec, "candidate_properties_ul.csv");
}
