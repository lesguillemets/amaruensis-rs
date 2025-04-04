mod consts;

const DEBUG: bool = true;
use opencv::core::{Range, CV_VERSION};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, ImreadModes};
use opencv::imgproc::{threshold, ThresholdTypes};
use opencv::prelude::*;

use consts::*;

pub fn do_main() {
    eprint_opencv_version();
    let pair = PaperPair::from_files(EXAMPLE_PAPER_PATH, EXAMPLE_SCANNED_PATH, false);
    let scanned = pair.scanned;
    let paper = pair.source;
    let diff = (&scanned - &paper).into_result().unwrap();
    println!("{paper:?}");
    println!("{scanned:?}");
    imshow("paper", &paper).unwrap();
    wait_key(0).unwrap();
    imshow("scanned", &scanned).unwrap();
    wait_key(0).unwrap();
    imshow("diff", &diff).unwrap();
    wait_key(0).unwrap();
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
