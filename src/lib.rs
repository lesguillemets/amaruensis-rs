mod consts;

const DEBUG: bool = true;
use opencv::core::{no_array, KeyPoint, Ptr, Range, Vector, CV_VERSION};
use opencv::features2d::{draw_keypoints_def, draw_matches_def, FlannBasedMatcher, ORB};
use opencv::flann;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, ImreadModes};
use opencv::imgproc::{threshold, ThresholdTypes};
use opencv::prelude::*;

use consts::*;

pub fn do_main() {
    eprint_opencv_version();
    let pair = PaperPair::from_files(EXAMPLE_PAPER_PATH, EXAMPLE_SCANNED_PATH, false);
    pair.detect_transform();
}

fn eprint_opencv_version() {
    eprintln!("Currently using opencv {CV_VERSION}");
}

/// Pair of the original document and scanned (filled) document
#[derive(Clone, Debug)]
pub struct PaperPair {
    source: Mat,
    scanned: Mat,
}

impl PaperPair {
    pub fn from_files(source: &str, scanned: &str, use_otsu: bool) -> Self {
        let paper = imread(EXAMPLE_PAPER_PATH, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
        let scanned = imread(EXAMPLE_SCANNED_PATH, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
        if DEBUG {
            eprintln!("Loaded paper: {paper:?}");
            eprintln!("Loaded scanned: {scanned:?}");
        }
        let mut p = PaperPair {
            source: paper,
            scanned,
        };
        p.fit_sizes();
        if use_otsu {
            p.apply(to_bw_ohtsu);
        } else {
            p.apply(to_bw);
        }
        p
    }
    pub fn apply(&self, f: fn(&Mat) -> Mat) -> Self {
        let new_source = f(&self.source);
        let new_scanned = f(&self.scanned);
        PaperPair {
            source: new_source,
            scanned: new_scanned,
        }
    }
    pub fn apply_inplace(&mut self, f: fn(&Mat) -> Mat) {
        self.source = f(&self.source);
        self.scanned = f(&self.scanned);
    }
    pub fn apply_into(self, f: fn(Mat) -> Mat) -> Self {
        let new_source = f(self.source);
        let new_scanned = f(self.scanned);
        PaperPair {
            source: new_source,
            scanned: new_scanned,
        }
    }
    /// Adjust the sizes of the images by dropping pixels.
    /// (assuming they don't differ too much)
    pub fn fit_sizes(&mut self) {
        let the_row = std::cmp::min(self.source.rows(), self.scanned.rows());
        let the_col = std::cmp::min(self.source.cols(), self.scanned.cols());
        Range::new(0, the_col).unwrap();
        let cropped_source = self
            .source
            .rowscols(
                // doesn't impl Clone
                Range::new(0, the_row).unwrap(),
                Range::new(0, the_col).unwrap(),
            )
            .unwrap();
        let cropped_scanned = self
            .scanned
            .rowscols(
                Range::new(0, the_row).unwrap(),
                Range::new(0, the_col).unwrap(),
            )
            .unwrap();
        self.source = cropped_source.clone_pointee();
        self.scanned = cropped_scanned.clone_pointee();
    }

    pub fn detect_transform(&self) {
        let (source_keypoints, source_descriptors) = detect_and_compute_orb(&self.source);
        let (scan_keypoints, scan_descriptors) = detect_and_compute_orb(&self.scanned);
        let flann_matcher = FlannBasedMatcher::new(
            &Ptr::new(flann::AutotunedIndexParams::new_def().unwrap().into()),
            &Ptr::new(flann::SearchParams::new_def().unwrap()),
        )
        .unwrap();
        let mut matches = Vector::new();
        flann_matcher
            .knn_train_match_def(&scan_descriptors, &source_descriptors, &mut matches, 3)
            .unwrap();
        if DEBUG {
            let mut result = Mat::default();
            draw_matches_def(
                &self.source,
                &source_keypoints,
                &self.scanned,
                &scan_keypoints,
                &matches.iter().flatten().collect(),
                &mut result,
            )
            .unwrap();
            imshow("Matching result", &result).unwrap();
            wait_key(0).unwrap();
        }
    }
}

fn detect_and_compute_orb(m: &Mat) -> (Vector<KeyPoint>, Mat) {
    let mut orb = ORB::create_def().unwrap();
    let mut keypoints = Vector::new();
    let mut descriptors = Mat::default();
    // TODO: 「ここには回答は来ない場所」を mask として使ってもよいかもしれない
    orb.detect_and_compute_def(&m, &no_array(), &mut keypoints, &mut descriptors)
        .unwrap();
    if DEBUG {
        let mut drawn = Mat::default();
        draw_keypoints_def(&m, &keypoints, &mut drawn).unwrap();
        imshow("Keypoints", &drawn).unwrap();
        wait_key(0).unwrap();
    }
    (keypoints, descriptors)
}

fn to_bw(m: &Mat) -> Mat {
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

fn to_bw_ohtsu(m: &Mat) -> Mat {
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
