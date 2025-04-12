use opencv::calib3d::{find_homography, RANSAC};
use opencv::core::{
    no_array, DMatch, KeyPoint, Point2f, Ptr, Size, VecN, Vector, BORDER_CONSTANT, CV_32F,
};
use opencv::features2d::{draw_keypoints_def, draw_matches_def, FlannBasedMatcher, ORB};
use opencv::flann;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imwrite, ImwriteFlags};
use opencv::imgproc::{warp_perspective, INTER_LINEAR, WARP_INVERSE_MAP};
use opencv::prelude::*;

use crate::base::{gather_good_matches_lowe, gather_good_matches_take_n};
use crate::consts::{LOWE_DEFAULT_THRESHOLD, ORB_ENLARGE_RECT_BY, ORB_FLANN_SHOW_N_BEST_MATCHES};
use crate::diff::{DiffMethod, DiffThenMedianBlur};
use crate::paper_pair::PaperPair;
use crate::sheet::SheetData;

const DEBUG: bool = true;
const PREFER_LOWE: bool = true;

pub trait ORBFlann {
    fn detect_transform(&self) -> Mat;
    fn calc_diff(&self);
}

fn detect_transform_orb_flann(
    source: &Mat,
    scanned: &Mat,
    sheet_data: &SheetData,
    enlarge_by: i32,
) -> Mat {
    // find keypoints
    let (source_keypoints, source_descriptors) =
        detect_and_compute_orb(source, Some(sheet_data.gen_detect_mask(source).unwrap()));
    // for scanned images, the corresponding points should be near that area.
    // We will search an area expanded a little (by ORB_ENLARGE_RECT_BY).
    let (scan_keypoints, scan_descriptors) = detect_and_compute_orb(
        scanned,
        Some(
            sheet_data
                .gen_enlarged_detect_mask(scanned, enlarge_by, enlarge_by)
                .unwrap(),
        ),
    );

    // use flann matcher to find matches
    let autoindexparams = flann::AutotunedIndexParams::new_def().expect("autotunedindexparams");
    let flann_matcher = FlannBasedMatcher::new(
        &Ptr::new(autoindexparams.into()),
        &Ptr::new(flann::SearchParams::new_def().unwrap()),
    )
    .expect("Creating FlannBasedMatcher");
    let mut src_desc = Mat::default();
    let mut scan_desc = Mat::default();
    // https://stackoverflow.com/a/29695032
    source_descriptors
        .convert_to_def(&mut src_desc, CV_32F)
        .unwrap();
    scan_descriptors
        .convert_to_def(&mut scan_desc, CV_32F)
        .unwrap();
    // use the best matches in the visualisation
    let bm: Vector<DMatch> = if PREFER_LOWE {
        let mut matches = Vector::new();
        flann_matcher
            .knn_train_match_def(&scan_desc, &src_desc, &mut matches, 2)
            .unwrap();
        gather_good_matches_lowe(&matches, LOWE_DEFAULT_THRESHOLD)
    } else {
        let mut matches = Vector::new();
        flann_matcher
            .train_match_def(&scan_desc, &src_desc, &mut matches)
            .unwrap();
        gather_good_matches_take_n(&matches, ORB_FLANN_SHOW_N_BEST_MATCHES)
    };
    if DEBUG {
        let mut result = Mat::default();
        draw_matches_def(
            &source,
            &source_keypoints,
            &scanned,
            &scan_keypoints,
            &bm,
            &mut result,
        )
        .unwrap();
        imshow("Matching result", &result).unwrap();
        wait_key(0).unwrap();
        imwrite(
            "result.png",
            &result,
            &Vector::from_slice(&[ImwriteFlags::IMWRITE_PNG_COMPRESSION.into(), 9]),
        )
        .unwrap();
    }

    // find the transformation matrix
    let mut from_points: Vector<KeyPoint> = Vector::with_capacity(bm.len());
    let mut to_points: Vector<KeyPoint> = Vector::with_capacity(bm.len());
    for kp in bm.iter() {
        from_points.push(
            // these unwraps will never fail
            source_keypoints
                .get(kp.train_idx.try_into().unwrap())
                .unwrap(),
        );
        to_points.push(
            // these unwraps will never fail
            scan_keypoints
                .get(kp.query_idx.try_into().unwrap())
                .unwrap(),
        );
    }
    let mut from_p2f: Vector<Point2f> = Vector::with_capacity(bm.len());
    KeyPoint::convert_def(&from_points, &mut from_p2f).unwrap();
    let mut to_p2f: Vector<Point2f> = Vector::with_capacity(bm.len());
    KeyPoint::convert_def(&to_points, &mut to_p2f).unwrap();
    let mut result_mask = Mat::default();
    find_homography(&from_p2f, &to_p2f, &mut result_mask, RANSAC, 10.0).unwrap()
}

fn calc_diff(source: &Mat, scanned: &Mat, homography: &Mat) {
    // use that matrix to transform
    let mut transformed_scanned = Mat::default();
    warp_perspective(
        &scanned,
        &mut transformed_scanned,
        &homography,
        Size {
            width: source.cols(),
            height: source.rows(),
        },
        INTER_LINEAR + WARP_INVERSE_MAP,
        BORDER_CONSTANT,
        VecN::default(),
    )
    .unwrap();
    let diff_blurred =
        DiffThenMedianBlur::use_ksizes(vec![5, 5, 5]).diff(source, &transformed_scanned);
    if DEBUG {
        imshow("transformed", &transformed_scanned).unwrap();
        wait_key(0).unwrap();
        imwrite(
            "transformed.png",
            &transformed_scanned,
            &Vector::from_slice(&[ImwriteFlags::IMWRITE_PNG_COMPRESSION.into(), 9]),
        )
        .unwrap();
        imshow("blurred_diff", &diff_blurred).unwrap();
        wait_key(0).unwrap();
        imwrite(
            "diff_blurred.png",
            &diff_blurred,
            &Vector::from_slice(&[ImwriteFlags::IMWRITE_PNG_COMPRESSION.into(), 9]),
        )
        .unwrap();
    }
}

impl ORBFlann for PaperPair {
    fn detect_transform(&self) -> Mat {
        detect_transform_orb_flann(
            &self.source,
            &self.scanned,
            &self.sheet_data,
            ORB_ENLARGE_RECT_BY,
        )
    }
    fn calc_diff(&self) {
        let homography = self.detect_transform();
        calc_diff(&self.source, &self.scanned, &homography);
    }
}

fn detect_and_compute_orb(m: &Mat, mask: Option<Mat>) -> (Vector<KeyPoint>, Mat) {
    let mut orb = ORB::create_def().unwrap();
    let mut keypoints = Vector::new();
    let mut descriptors = Mat::default();
    // TODO: 「ここには回答は来ない場所」を mask として使ってもよいかもしれない
    if let Some(mask) = mask {
        orb.detect_and_compute_def(&m, &mask, &mut keypoints, &mut descriptors)
            .unwrap();
    } else {
        orb.detect_and_compute_def(&m, &no_array(), &mut keypoints, &mut descriptors)
            .unwrap();
    }
    if DEBUG {
        let mut drawn = Mat::default();
        draw_keypoints_def(&m, &keypoints, &mut drawn).unwrap();
        imshow("Keypoints", &drawn).unwrap();
        wait_key(0).unwrap();
        let file_name = format!(
            "result_keypoints_{:?}.jpg",
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        );
        imwrite(&file_name, &drawn, &Vector::from_slice(&[])).unwrap();
    }
    (keypoints, descriptors)
}
