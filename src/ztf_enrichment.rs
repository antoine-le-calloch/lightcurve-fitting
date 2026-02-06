use crate::lightcurves::{analyze_photometry, prepare_photometry};
use crate::models::{AllBandsProperties, PerBandProperties, PhotometryMag, ZtfCandidate};

/// ZTF alert structure used to deserialize alerts
/// from the database, used by the enrichment worker
/// to compute features and ML scores
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ZtfAlertForEnrichment {
    #[serde(rename = "_id")]
    pub candid: i64,
    #[serde(rename = "objectId")]
    pub object_id: String,
    pub candidate: ZtfCandidate,
    pub prv_candidates: Vec<PhotometryMag>,
    pub fp_hists: Vec<PhotometryMag>,
}

/// ZTF alert properties computed during enrichment
/// and inserted back into the alert document
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ZtfAlertProperties {
    pub rock: bool,
    pub star: bool,
    pub near_brightstar: bool,
    pub stationary: bool,
    pub photstats: PerBandProperties,
}

pub fn get_alert_properties(alert: &ZtfAlertForEnrichment) -> (
        ZtfAlertProperties,
        AllBandsProperties,
        i32,
        Vec<PhotometryMag>
) {
    let candidate = &alert.candidate.candidate;
    let programid = candidate.programid;
    let ssdistnr = candidate.ssdistnr.unwrap_or(f64::INFINITY);
    let ssmagnr = candidate.ssmagnr.unwrap_or(f64::INFINITY);
    let is_rock = ssdistnr >= 0.0 && ssdistnr < 12.0 && ssmagnr >= 0.0;

    let sgscore1 = candidate.sgscore1.unwrap_or(0.0);
    let sgscore2 = candidate.sgscore2.unwrap_or(0.0);
    let sgscore3 = candidate.sgscore3.unwrap_or(0.0);
    let distpsnr1 = candidate.distpsnr1.unwrap_or(f64::INFINITY);
    let distpsnr2 = candidate.distpsnr2.unwrap_or(f64::INFINITY);
    let distpsnr3 = candidate.distpsnr3.unwrap_or(f64::INFINITY);

    let srmag1 = candidate.srmag1.unwrap_or(f64::INFINITY);
    let srmag2 = candidate.srmag2.unwrap_or(f64::INFINITY);
    let srmag3 = candidate.srmag3.unwrap_or(f64::INFINITY);
    let sgmag1 = candidate.sgmag1.unwrap_or(f64::INFINITY);
    let simag1 = candidate.simag1.unwrap_or(f64::INFINITY);
    let szmag1 = candidate.szmag1.unwrap_or(f64::INFINITY);

    let neargaiabright = candidate.neargaiabright.unwrap_or(f64::INFINITY);
    let maggaiabright = candidate.maggaiabright.unwrap_or(f64::INFINITY);

    let is_star = (sgscore1 > 0.76 && distpsnr1 >= 0.0 && distpsnr1 <= 2.0)
        || (sgscore1 > 0.2
        && distpsnr1 >= 0.0
        && distpsnr1 <= 1.0
        && srmag1 > 0.0
        && ((szmag1 > 0.0 && srmag1 - szmag1 > 3.0)
        || (simag1 > 0.0 && srmag1 - simag1 > 3.0)));

    let is_near_brightstar = (neargaiabright >= 0.0
        && neargaiabright <= 20.0
        && maggaiabright > 0.0
        && maggaiabright <= 12.0)
        || (sgscore1 > 0.49 && distpsnr1 <= 20.0 && srmag1 > 0.0 && srmag1 <= 15.0)
        || (sgscore2 > 0.49 && distpsnr2 <= 20.0 && srmag2 > 0.0 && srmag2 <= 15.0)
        || (sgscore3 > 0.49 && distpsnr3 <= 20.0 && srmag3 > 0.0 && srmag3 <= 15.0)
        || (sgscore1 == 0.5
        && distpsnr1 < 0.5
        && (sgmag1 < 17.0 || srmag1 < 17.0 || simag1 < 17.0));

    let prv_candidates = alert.prv_candidates.clone();
    let fp_hists = alert.fp_hists.clone();

    // lightcurve is prv_candidates + fp_hists, no need for parse_photometry here
    let mut lightcurve = [prv_candidates, fp_hists].concat();

    prepare_photometry(&mut lightcurve);
    let (photstats, all_bands_properties, stationary) = analyze_photometry(&lightcurve);

    (
        ZtfAlertProperties {
            rock: is_rock,
            star: is_star,
            near_brightstar: is_near_brightstar,
            stationary,
            photstats,
        },
        all_bands_properties,
        programid,
        lightcurve,
    )
}