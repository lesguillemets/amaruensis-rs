use opencv::imgproc::{threshold, ThresholdTypes};
use opencv::prelude::*;

use crate::consts;

const DEBUG: bool = true;

pub fn to_bw(m: &Mat) -> Mat {
    let mut dest = Mat::default();
    threshold(
        m,
        &mut dest,
        consts::BLACK_WHITE_THRESH,
        consts::WHITE,
        ThresholdTypes::THRESH_BINARY.into(),
    )
    .unwrap();
    dest
}

pub fn to_bw_ohtsu(m: &Mat) -> Mat {
    let mut dest = Mat::default();
    let found_thresh = threshold(
        m,
        &mut dest,
        consts::BLACK_WHITE_THRESH,
        consts::WHITE,
        ThresholdTypes::THRESH_OTSU.into(),
    )
    .unwrap();
    if DEBUG {
        eprintln!("Fount threshold: {found_thresh}");
    }
    dest
}
