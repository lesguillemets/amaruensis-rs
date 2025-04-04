mod consts;

use opencv::core::CV_VERSION;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, ImreadModes};
use opencv::imgproc::{threshold, ThresholdTypes};
use opencv::prelude::*;

use consts::*;

pub fn do_main() {
    eprint_opencv_version();
    let (paper, scanned) = load_images();
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

fn load_images() -> (Mat, Mat) {
    let paper = imread(EXAMPLE_PAPER_PATH, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
    let scanned = imread(EXAMPLE_SCANNED_PATH, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
    println!("{paper:?}");
    println!("{scanned:?}");
    let mut p = Mat::default();
    let mut s = Mat::default();
    threshold(
        &paper,
        &mut p,
        consts::BLACK_WHITE_THRESH,
        consts::WHITE,
        ThresholdTypes::THRESH_BINARY.into(),
    )
    .unwrap();
    threshold(
        &scanned,
        &mut s,
        consts::BLACK_WHITE_THRESH,
        consts::WHITE,
        ThresholdTypes::THRESH_BINARY.into(),
    )
    .unwrap();
    (p, s)
}
