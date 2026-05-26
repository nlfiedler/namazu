//! `POST /synthetic/:blobId` handler.

use std::io;
use std::path::Path;

use actix_web::{HttpResponse, web};
use log::warn;
use serde_json::json;

use super::engine::{ProcessError, SyntheticEngine};
use crate::{blob_path, is_video};

pub async fn post_synthetic(
    info: web::Path<String>,
    engine: web::Data<SyntheticEngine>,
) -> actix_web::Result<HttpResponse> {
    let Ok(filepath) = blob_path(&info) else {
        return Ok(HttpResponse::BadRequest().finish());
    };

    match tokio::fs::metadata(&filepath).await {
        Ok(m) if !m.is_file() => return Ok(HttpResponse::NotFound().finish()),
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok(HttpResponse::NotFound().finish());
        }
        Err(e) => return Err(e.into()),
    }

    if !is_image(&filepath) {
        return Ok(HttpResponse::NoContent().finish());
    }

    let engine = engine.clone();
    let filepath_for_block = filepath.clone();
    let result =
        web::block(move || engine.process(&filepath_for_block)).await?;

    match result {
        Ok(response) => match serde_json::to_vec(&response) {
            Ok(body) => Ok(HttpResponse::Ok().content_type("application/json").body(body)),
            Err(e) => {
                warn!("serialize synthetic response failed: {e}");
                Ok(error_response(
                    HttpResponse::InternalServerError(),
                    "could not serialize response",
                    "internal_error",
                ))
            }
        },
        Err(ProcessError::Decode(e)) => {
            warn!("synthetic decode failed for {}: {e}", info);
            Ok(error_response(
                HttpResponse::BadRequest(),
                "image could not be decoded",
                "decode_failed",
            ))
        }
        Err(ProcessError::TooLarge(w, h)) => Ok(error_response(
            HttpResponse::PayloadTooLarge(),
            &format!("image dimensions {w}x{h} exceed processing limit"),
            "too_large",
        )),
        Err(ProcessError::Inference(e)) => {
            warn!("synthetic inference failed for {}: {e}", info);
            Ok(error_response(
                HttpResponse::BadRequest(),
                "inference failed",
                "inference_failed",
            ))
        }
        Err(ProcessError::ThumbnailEncode(e)) => {
            warn!("synthetic thumbnail encode failed for {}: {e}", info);
            Ok(error_response(
                HttpResponse::InternalServerError(),
                "could not encode face thumbnail",
                "thumbnail_encode_failed",
            ))
        }
    }
}

fn error_response(
    mut builder: actix_web::HttpResponseBuilder,
    message: &str,
    code: &str,
) -> HttpResponse {
    let body = json!({ "error": message, "code": code });
    let bytes = serde_json::to_vec(&body).unwrap_or_else(|_| b"{}".to_vec());
    builder.content_type("application/json").body(bytes)
}

/// Conservative image-extension whitelist. Anything else (including videos
/// and unknown types) yields a 204.
fn is_image(path: &Path) -> bool {
    if is_video(path) {
        return false;
    }
    let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "jpe" | "png" | "gif" | "webp" | "bmp" | "tif" | "tiff"
    )
}
