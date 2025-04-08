use opencv::core::{DMatch, Vector};
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

/// Take n best matches, taking these with least distances
/// Accepts: &Vector<DMatch>, supposedly the result of
/// DescriptorMatcherTraitConst::train_match or something
pub fn gather_good_matches_take_n(matches: &Vector<DMatch>, n: usize) -> Vector<DMatch> {
    let mut good_matches: Vec<DMatch> = matches.iter().collect();
    good_matches.sort_by(|ma, mb| ma.distance.partial_cmp(&mb.distance).unwrap());
    good_matches.into_iter().take(n).collect()
}

/// Use Lowe's ratio test with threshold to get the better matches
/// Accepts: &Vector<Vector<DMatch>>, supposedly the result of
/// DescriptorMatcherTraitConst::knn_train_match or something
pub fn gather_good_matches_lowe(
    matches: &Vector<Vector<DMatch>>,
    ratio_threshold: f32,
) -> Vector<DMatch> {
    let mut good_matches: Vector<DMatch> = Vector::new();
    for m in matches.iter() {
        let better = m.get(0).unwrap();
        if m.get(1).unwrap().distance * ratio_threshold > better.distance {
            good_matches.push(better);
        }
    }
    good_matches
}
