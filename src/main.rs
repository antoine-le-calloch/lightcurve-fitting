use std::fs::File;
use serde_json::from_reader;
use crate::ztf_enrichment::{get_alert_properties, ZtfAlertForEnrichment};

mod ztf_enrichment;
mod models;
mod lightcurves;

fn load_ztf_candidate(path: &str) -> anyhow::Result<ZtfAlertForEnrichment> {
    let file = File::open(path)?;
    let alert: ZtfAlertForEnrichment = from_reader(file)?;
    Ok(alert)
}

fn main() {
    let alert = load_ztf_candidate("candidates/ZTF25acdhetm.json").expect("Failed to load ZTF candidate");
    let (alert_props, all_bands_props, _, lightcurve) = get_alert_properties(&alert);

    println!("Alert props: {:?}", alert_props);
    println!("All bands props: {:?}", all_bands_props);
    println!("Lightcurve length: {}", lightcurve.len());
}
