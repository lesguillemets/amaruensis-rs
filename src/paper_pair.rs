use opencv::core::Range;
use opencv::imgcodecs::{imread, ImreadModes};
use opencv::prelude::*;

use crate::base::{to_bw, to_bw_ohtsu};

const DEBUG: bool = true;

/// Pair of the original document and scanned (filled) document
#[derive(Clone, Debug)]
pub struct PaperPair {
    pub source: Mat,
    pub scanned: Mat,
}

impl PaperPair {
    pub fn from_files(source: &str, scanned: &str, use_otsu: bool) -> Self {
        let paper = imread(source, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
        let scanned = imread(scanned, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
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
