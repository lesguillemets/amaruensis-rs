use opencv::core::Rect;
use serde::{Deserialize, Serialize};
use std::convert::From;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SheetData {
    pub name: String,
    pub img_path: String,
    pub detect_rects: Vec<Rect_>,
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
