use opencv::core::{Rect, Scalar_, CV_8UC1};
use opencv::error;
use opencv::prelude::*;
use serde::{Deserialize, Serialize};
use std::convert::From;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SheetData {
    pub name: String,
    pub img_path: String,
    pub detect_rects: Vec<Rect_>,
}

impl SheetData {
    pub fn gen_detect_mask(&self, img: &Mat) -> error::Result<Mat> {
        self.gen_enlarged_detect_mask(img, 0, 0)
    }

    pub fn gen_enlarged_detect_mask(&self, img: &Mat, by_x: i32, by_y: i32) -> error::Result<Mat> {
        let mut m = Mat::new_rows_cols_with_default(
            img.rows(),
            img.cols(),
            CV_8UC1,
            Scalar_::new(0.0, 0.0, 0.0, 0.0),
        )?;
        for r in &self.detect_rects {
            let mut roi = m.roi_mut(r.enlarge(by_x, by_y).into())?;
            // FIXME : not sure
            roi.set_to_def(&Scalar_::new(255.0, 255.0, 255.0, 255.0))?;
        }
        Ok(m)
    }
}

#[derive(Default)]
pub struct SheetDataBuilder {
    name: Option<String>,
    img_path: Option<String>,
    detect_rects: Option<Vec<Rect_>>,
}

impl SheetDataBuilder {
    pub fn new() -> Self {
        SheetDataBuilder::default()
    }
    pub fn new_with_name(n: &str) -> Self {
        SheetDataBuilder::new().name(n.to_string())
    }
    pub fn name(mut self, n: String) -> Self {
        self.name = Some(n);
        self
    }
    pub fn img_path(mut self, p: String) -> Self {
        self.img_path = Some(p);
        self
    }
    pub fn detect_rects(mut self, dr: Vec<Rect_>) -> Self {
        self.detect_rects = Some(dr);
        self
    }

    pub fn build(self) -> Option<SheetData> {
        Some(SheetData {
            name: self.name?,
            img_path: self.img_path?,
            detect_rects: self.detect_rects?,
        })
    }
}

// reimplemented just for serder
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default)]
pub struct Rect_ {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect_ {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Rect_ {
            x,
            y,
            width,
            height,
        }
    }
    /// 両側に広げた Rect_ を返す
    pub fn enlarge(&self, by_x: i32, by_y: i32) -> Self {
        let x = std::cmp::max(self.x - by_x, 0);
        let y = std::cmp::max(self.y - by_y, 0);
        let rightmost = self.x + self.width + by_x;
        let bottommost = self.y + self.height + by_y;
        Rect_ {
            x,
            y,
            width: rightmost - x,
            height: bottommost - y,
        }
    }
}

impl From<Rect> for Rect_ {
    fn from(r: Rect) -> Self {
        Rect_ {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

impl From<Rect_> for Rect {
    fn from(r: Rect_) -> Self {
        Rect {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}
