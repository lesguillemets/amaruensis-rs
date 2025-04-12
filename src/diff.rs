use opencv::imgproc::median_blur;
/// Here, we deal with taking the diff of the source image and scanned image,
/// after they are grayscaled and transformed.
use opencv::prelude::*;

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
