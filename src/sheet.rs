use opencv::core::Rect;
use serde::{Deserialize, Serialize};
use std::convert::{AsRef, From};

#[derive(Serialize, Deserialize, Debug)]
struct Sheet {
    name: String,
    img_path: String,
    detect_rects: Vec<Rect_>,
}

// reimplemented just for serder
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
struct Rect_ {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
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
