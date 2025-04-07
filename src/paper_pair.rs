use opencv::core::Range;
use opencv::imgcodecs::{imread, ImreadModes};
use opencv::prelude::*;

use crate::base::{to_bw, to_bw_ohtsu};
use crate::sheet::SheetData;

use std::fs::File;
use std::io::BufReader;

const DEBUG: bool = true;

/// Pair of the original document and scanned (filled) document
#[derive(Clone, Debug)]
pub struct PaperPair {
    pub source: Mat,
    pub sheet_data: SheetData,
    pub scanned: Mat,
}

impl PaperPair {
    pub fn from_files(source: &str, scanned: &str, sheet: &str, use_otsu: bool) -> Self {
        let paper = imread(source, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
        let scanned = imread(scanned, ImreadModes::IMREAD_GRAYSCALE.into()).unwrap();
        let sheet_file = File::open(sheet).unwrap();
        let reader = BufReader::new(sheet_file);
        let sheet_data: SheetData = serde_json::from_reader(reader).unwrap();
        if DEBUG {
            eprintln!("Loaded paper: {paper:?}");
            eprintln!("Loaded scanned: {scanned:?}");
            eprintln!("Loaded sheet: {sheet_data:?}");
        }
        let mut p = PaperPair {
            source: paper,
            sheet_data,
            scanned,
        };
        // p.fit_sizes();
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
            sheet_data: self.sheet_data.clone(),
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
            sheet_data: self.sheet_data,
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
