use opencv::calib3d::{find_homography, RANSAC};
use opencv::core::{
    no_array, DMatch, KeyPoint, Point2f, Ptr, Size, VecN, Vector, BORDER_CONSTANT, CV_32F,
};
use opencv::features2d::{
    draw_keypoints_def, draw_matches_def, FlannBasedMatcher, KeyPointsFilter, ORB,
};
use opencv::flann;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imwrite, ImwriteFlags};
use opencv::imgproc::{warp_perspective, warp_perspective_def, INTER_LINEAR, WARP_INVERSE_MAP};
use opencv::prelude::*;

use crate::consts::{ORB_ENLARGE_RECT_BY, ORB_FLANN_SHOW_N_BEST_MATCHES};

use crate::paper_pair::PaperPair;

const DEBUG: bool = true;

pub trait ORBFlann {
    fn detect_transform(&self);
}

impl ORBFlann for PaperPair {
    fn detect_transform(&self) {
        // find keypoints
        let (source_keypoints, source_descriptors) = detect_and_compute_orb(
            &self.source,
            Some(self.sheet_data.gen_detect_mask(&self.source).unwrap()),
        );
        // for scanned images, the corresponding points should be near that area.
        // We will search an area expanded a little (by ORB_ENLARGE_RECT_BY).
        let (scan_keypoints, scan_descriptors) = detect_and_compute_orb(
            &self.scanned,
            Some(
                self.sheet_data
                    .gen_enlarged_detect_mask(
                        &self.scanned,
                        ORB_ENLARGE_RECT_BY,
                        ORB_ENLARGE_RECT_BY,
                    )
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
        let mut matches = Vector::new();
        let mut src_desc = Mat::default();
        let mut scan_desc = Mat::default();
        // https://stackoverflow.com/a/29695032
        source_descriptors
            .convert_to_def(&mut src_desc, CV_32F)
            .unwrap();
        scan_descriptors
            .convert_to_def(&mut scan_desc, CV_32F)
            .unwrap();
        flann_matcher
            .train_match_def(&src_desc, &scan_desc, &mut matches)
            .unwrap();
        // use the best matches in the visualisation
        let mut best_matches: Vec<DMatch> = matches.iter().collect();
        best_matches.sort_by(|ma, mb| ma.distance.partial_cmp(&mb.distance).unwrap());
        best_matches = best_matches
            .into_iter()
            .take(ORB_FLANN_SHOW_N_BEST_MATCHES)
            .collect();
        let bm: Vector<DMatch> = best_matches.into_iter().collect();
        if DEBUG {
            let mut result = Mat::default();
            draw_matches_def(
                &self.source,
                &source_keypoints,
                &self.scanned,
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
        let homography =
            find_homography(&from_p2f, &to_p2f, &mut result_mask, RANSAC, 10.0).unwrap();

        // use that matrix to transform
        let mut transformed_source = Mat::default();
        warp_perspective(
            &self.source,
            &mut transformed_source,
            &homography,
            Size {
                width: self.source.cols(),
                height: self.source.rows(),
            },
            INTER_LINEAR,
            BORDER_CONSTANT,
            VecN::default(),
        )
        .unwrap();
        if DEBUG {
            imshow("transformed", &transformed_source).unwrap();
            wait_key(0).unwrap();
            imwrite(
                "transformed.png",
                &transformed_source,
                &Vector::from_slice(&[ImwriteFlags::IMWRITE_PNG_COMPRESSION.into(), 9]),
            )
            .unwrap();
        }
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
