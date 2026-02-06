use schemars::_private::serde_json;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, skip_serializing_none};

fn deserialize_ssnamenr<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    match opt {
        Some(s) if s.trim().is_empty() => Ok(None),
        _ => Ok(opt),
    }
}

fn deserialize_isdiffpos_option<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<serde_json::Value>::deserialize(deserializer)?;
    match opt {
        Some(serde_json::Value::String(s)) => {
            if s.eq_ignore_ascii_case("t")
                || s.eq_ignore_ascii_case("true")
                || s.eq_ignore_ascii_case("1")
            {
                Ok(Some(true))
            } else {
                Ok(Some(false))
            }
        }
        Some(serde_json::Value::Number(n)) => Ok(Some(
            n.as_i64().ok_or(serde::de::Error::custom(
                "Failed to convert isdiffpos to i64",
            ))? == 1,
        )),
        Some(serde_json::Value::Bool(b)) => Ok(Some(b)),
        _ => Ok(None),
    }
}

fn deserialize_isdiffpos<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_isdiffpos_option(deserializer).map(|x| x.unwrap())
}

/// avro alert schema
#[serde_as]
#[skip_serializing_none]
#[derive(Debug, PartialEq, Clone, Deserialize, Serialize, Default, schemars::JsonSchema)]
#[serde(default)]
pub struct Candidate {
    pub jd: f64,
    pub fid: i32,
    pub pid: i64,
    pub diffmaglim: Option<f64>,
    pub programpi: Option<String>,
    pub programid: i32,
    pub candid: i64,
    #[serde(deserialize_with = "deserialize_isdiffpos")]
    pub isdiffpos: bool,
    pub nid: Option<i32>,
    pub rcid: Option<i32>,
    pub field: Option<i32>,
    pub ra: f64,
    pub dec: f64,
    pub magpsf: f64,
    pub sigmapsf: f64,
    pub chipsf: Option<f64>,
    pub magap: Option<f64>,
    pub sigmagap: Option<f64>,
    pub distnr: Option<f64>,
    pub magnr: Option<f64>,
    pub sigmagnr: Option<f64>,
    pub chinr: Option<f64>,
    pub sharpnr: Option<f64>,
    pub sky: Option<f64>,
    pub fwhm: Option<f64>,
    pub classtar: Option<f64>,
    pub mindtoedge: Option<f64>,
    pub seeratio: Option<f64>,
    pub aimage: Option<f64>,
    pub bimage: Option<f64>,
    pub elong: Option<f64>,
    pub nneg: Option<i32>,
    pub nbad: Option<i32>,
    pub rb: Option<f64>,
    pub ssdistnr: Option<f64>,
    pub ssmagnr: Option<f64>,
    #[serde(deserialize_with = "deserialize_ssnamenr")]
    pub ssnamenr: Option<String>,
    pub ranr: f64,
    pub decnr: f64,
    pub sgmag1: Option<f64>,
    pub srmag1: Option<f64>,
    pub simag1: Option<f64>,
    pub szmag1: Option<f64>,
    pub sgscore1: Option<f64>,
    pub distpsnr1: Option<f64>,
    pub ndethist: i32,
    pub ncovhist: i32,
    pub jdstarthist: Option<f64>,
    pub scorr: Option<f64>,
    pub sgmag2: Option<f64>,
    pub srmag2: Option<f64>,
    pub simag2: Option<f64>,
    pub szmag2: Option<f64>,
    pub sgscore2: Option<f64>,
    pub distpsnr2: Option<f64>,
    pub sgmag3: Option<f64>,
    pub srmag3: Option<f64>,
    pub simag3: Option<f64>,
    pub szmag3: Option<f64>,
    pub sgscore3: Option<f64>,
    pub distpsnr3: Option<f64>,
    pub nmtchps: i32,
    pub dsnrms: Option<f64>,
    pub ssnrms: Option<f64>,
    pub dsdiff: Option<f64>,
    pub magzpsci: Option<f64>,
    pub magzpsciunc: Option<f64>,
    pub magzpscirms: Option<f64>,
    pub zpmed: Option<f64>,
    pub exptime: Option<f64>,
    pub drb: Option<f64>,

    pub clrcoeff: Option<f64>,
    pub clrcounc: Option<f64>,
    pub neargaia: Option<f64>,
    pub maggaia: Option<f64>,
    pub neargaiabright: Option<f64>,
    pub maggaiabright: Option<f64>,
}

#[derive(Debug, PartialEq, Clone, Deserialize, Serialize, Eq, Hash, schemars::JsonSchema)]
pub enum Band {
    #[serde(rename = "g")]
    G,
    #[serde(rename = "r")]
    R,
    #[serde(rename = "i")]
    I,
    #[serde(rename = "z")]
    Z,
    #[serde(rename = "y")]
    Y,
    #[serde(rename = "u")]
    U,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PhotometryMag {
    #[serde(alias = "jd")]
    pub time: f64,
    #[serde(alias = "magpsf")]
    pub mag: f64,
    #[serde(alias = "sigmapsf")]
    pub mag_err: f64,
    pub band: Band,
}

#[serde_as]
#[skip_serializing_none]
#[derive(Debug, PartialEq, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ZtfCandidate {
    #[serde(flatten)]
    pub candidate: Candidate,
    #[serde(rename = "psfFlux")]
    pub psf_flux: f64,
    #[serde(rename = "psfFluxErr")]
    pub psf_flux_err: f64,
    pub snr: f64,
    pub band: Band,
}

#[serde_as]
#[skip_serializing_none]
#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub struct BandRateProperties {
    pub rate: f64,
    pub r_squared: f64,
    pub nb_data: i32,
}

#[serde_as]
#[skip_serializing_none]
#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub struct BandProperties {
    pub peak_jd: f64,
    pub peak_mag: f64,
    pub peak_mag_err: f64,
    pub rising: Option<BandRateProperties>,
    pub fading: Option<BandRateProperties>,
}

#[serde_as]
#[skip_serializing_none]
#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)] pub struct PerBandProperties {
    pub g: Option<BandProperties>,
    pub r: Option<BandProperties>,
    pub i: Option<BandProperties>,
    pub z: Option<BandProperties>,
    pub y: Option<BandProperties>,
    pub u: Option<BandProperties>,
}

#[serde_as]
#[skip_serializing_none]
#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub struct AllBandsProperties {
    pub peak_jd: f64,
    pub peak_mag: f64,
    pub peak_mag_err: f64,
    pub peak_band: Band,
    pub faintest_jd: f64,
    pub faintest_mag: f64,
    pub faintest_mag_err: f64,
    pub faintest_band: Band,
    pub first_jd: f64,
    pub last_jd: f64,
}

pub struct LinearFitResult {
    pub slope: f64,
    pub r_squared: f64,
    pub nb_data: i32,
}
