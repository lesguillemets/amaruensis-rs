//! Here, we deal with taking the diff of the source image and scanned image,
//! after they are grayscaled and transformed.

use opencv::core::absdiff;
use opencv::imgproc::median_blur;
use opencv::prelude::*;

use crate::base::to_bw;
use crate::consts;

pub trait DiffMethod {
    fn diff(&self, source: &Mat, scanned: &Mat) -> Mat;
}

/// Subtract, then apply median blurs repeatedly using given ksize
pub struct DiffThenMedianBlur {
    ksizes: Vec<i32>,
}

impl DiffThenMedianBlur {
    pub fn use_ksizes(ksizes: Vec<i32>) -> Self {
        DiffThenMedianBlur { ksizes: { ksizes } }
    }
}

impl DiffMethod for DiffThenMedianBlur {
    fn diff(&self, source: &Mat, scanned: &Mat) -> Mat {
        let mut diff = (source - scanned).into_result().unwrap().to_mat().unwrap();
        for ksize in &self.ksizes {
            let current = diff.clone();
            median_blur(&current, &mut diff, *ksize).unwrap();
        }
        diff
    }
}

/// Just gets the absolute difference:
/// Currently exploring plain difference (source - scanned)
/// because there's not so much information when scanned is brighter.
pub struct PlainAbsDiff;

impl DiffMethod for PlainAbsDiff {
    fn diff(&self, source: &Mat, scanned: &Mat) -> Mat {
        let mut diff = Mat::default();
        absdiff(source, scanned, &mut diff).unwrap();
        diff
    }
}

/// Absolute difference AFTER thresholding to binary.
pub struct ThresholdAndAbsDiff;
impl DiffMethod for ThresholdAndAbsDiff {
    fn diff(&self, source: &Mat, scanned: &Mat) -> Mat {
        let bw_source = to_bw(source, consts::BLACK_WHITE_THRESH);
        let bw_scanned = to_bw(scanned, consts::BLACK_WHITE_THRESH);
        let mut diff = Mat::default();
        absdiff(&bw_source, &bw_scanned, &mut diff).unwrap();
        diff
    }
}
