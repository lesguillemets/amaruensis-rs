mod base;
mod consts;
mod orb_flann;
mod paper_pair;

use opencv::core::CV_VERSION;

use consts::*;
use orb_flann::ORBFlann;
use paper_pair::PaperPair;

pub fn do_main() {
    eprint_opencv_version();
    let pair = PaperPair::from_files(EXAMPLE_PAPER_PATH, EXAMPLE_SCANNED_PATH, false);
    pair.detect_transform();
}

fn eprint_opencv_version() {
    eprintln!("Currently using opencv {CV_VERSION}");
}
