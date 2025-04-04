mod consts;

use opencv::core::CV_VERSION;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, ImreadModes};
use opencv::imgproc::{threshold, ThresholdTypes};
use opencv::prelude::*;

use consts::*;

pub fn do_main() {
    eprint_opencv_version();
    let pair = load_images();
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

fn load_images() -> PaperPair {
    let paper = imread(EXAMPLE_PAPER_PATH, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
    let scanned = imread(EXAMPLE_SCANNED_PATH, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
    println!("{paper:?}");
    println!("{scanned:?}");
    let mut loaded = PaperPair {
        source: paper,
        scanned,
    };
    loaded.apply_both(to_bw);
    loaded
}

/// Pair of the original document and scanned (filled) document
#[derive(Debug)]
pub struct PaperPair {
    source: Mat,
    scanned: Mat,
}

impl PaperPair {
    pub fn apply_both(&mut self, f: fn(&Mat) -> Mat) {
        self.source = f(&self.source);
        self.scanned = f(&self.scanned);
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
