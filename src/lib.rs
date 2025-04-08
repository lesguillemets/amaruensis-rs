mod base;
pub mod consts;
mod orb_flann;
mod paper_pair;
pub mod sheet;

use opencv::core::CV_VERSION;

use orb_flann::ORBFlann;
use paper_pair::PaperPair;

pub fn do_main(f0: String, f1: String, fs: String) {
    eprint_opencv_version();
    let pair = PaperPair::from_files(&f0, &f1, &fs, false);
    pair.calc_diff();
}

fn eprint_opencv_version() {
    eprintln!("Currently using opencv {CV_VERSION}");
}
