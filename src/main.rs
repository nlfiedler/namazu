//
// Copyright (c) 2025 Nathan Fiedler
//
use actix_files::NamedFile;
use actix_web::http::header;
use actix_web::{App, Either, HttpMessage, HttpRequest, HttpResponse, HttpServer, Responder, web};
use anyhow::{Error, anyhow};
use base64::{Engine as _, engine::general_purpose};
use log::info;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

#[cfg(test)]
static DEFAULT_ASSETS_PATH: &str = "tests/blobs";
#[cfg(not(test))]
static DEFAULT_ASSETS_PATH: &str = "tmp/blobs";

pub static ASSETS_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    let path = env::var("ASSETS_PATH").unwrap_or_else(|_| DEFAULT_ASSETS_PATH.to_owned());
    PathBuf::from(path)
});

/// Convert the identifier to a file path within the assets store.
fn blob_path(encoded: &str) -> Result<PathBuf, Error> {
    let decoded = general_purpose::STANDARD.decode(encoded)?;
    let as_string = str::from_utf8(&decoded)?;
    // Windows will raise unexpected errors if the path has any trailing EOL
    // characters which is very easy to do with base64 encoding.
    let rel_path = Path::new(as_string.trim());
    if rel_path.has_root() {
        return Err(anyhow!("root path not allowed"));
    }
    // Note also that on Windows both slash and backslash can be used in the
    // same path without any issues. Only the trailing EOL causes problems.
    let mut full_path = ASSETS_PATH.to_path_buf();
    full_path.push(rel_path);
    Ok(full_path)
}

/// Produce a thumbnail for the given asset (assumed to be an image) that fits
/// within the bounds given while maintaining aspect ratio.
///
/// If the asset is not an image, or any other problem arises, returns an error.
fn create_thumbnail(filepath: &Path, width: u32, height: u32) -> Result<Vec<u8>, Error> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    // The image crate does not recognize .jpe extension as jpeg, so use the
    // format guessing code based on the first few bytes.
    let mut img = image::ImageReader::open(filepath)?
        .with_guessed_format()?
        .decode()?;
    match get_image_orientation(filepath) {
        Ok(orientation) => {
            // c.f. https://magnushoff.com/articles/jpeg-orientation/
            if orientation > 4 {
                // image is sideways, need to swap new width/height
                img = img.thumbnail(height, width);
            } else {
                img = img.thumbnail(width, height);
            }
            img = correct_orientation(orientation, img);
        }
        _ => {
            img = img.thumbnail(width, height);
        }
    }
    // The image crate's JpegEncoder will use a quality factor of 75 by default,
    // which yields very good results (libvips uses the same default).
    img.write_to(&mut cursor, image::ImageFormat::Jpeg)?;
    let buffer: Vec<u8> = cursor.into_inner();
    Ok(buffer)
}

/// Extract the EXIF orientation value from the asset, if any.
fn get_image_orientation(filepath: &Path) -> Result<u16, Error> {
    let file = std::fs::File::open(filepath)?;
    let mut buffer = std::io::BufReader::new(&file);
    let reader = exif::Reader::new();
    let exif = reader.read_from_container(&mut buffer)?;
    let field = exif
        .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .ok_or_else(|| anyhow!("no orientation field"))?;
    if let exif::Value::Short(data) = &field.value {
        return Ok(data[0]);
    }
    Err(anyhow!("not an image"))
}

/// Flip and/or rotate the image to have the correct rendering.
///
/// The orientation value should be as read from the EXIF header.
fn correct_orientation(orientation: u16, img: image::DynamicImage) -> image::DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.flipv().rotate90(),
        6 => img.rotate90(),
        7 => img.fliph().rotate90(),
        8 => img.rotate270(),
        _ => img,
    }
}

async fn index(_req: HttpRequest) -> &'static str {
    "See the README.md file for more information."
}

async fn get_asset(info: web::Path<String>) -> actix_web::Result<impl Responder> {
    if let Ok(filepath) = blob_path(&info) {
        if filepath.exists() {
            // NamedFile will generate an ETag and respond to If-None-Match with
            // 304 Not Modified, and respond to Range requests with 206 Partial
            // Content and a Content-Range header.
            let named_file = NamedFile::open(filepath)?;
            let responder = named_file
                .customize()
                .insert_header(("Cache-Control", "public, max-age=31536000"));
            Ok(Either::Left(responder))
        } else {
            Ok(Either::Right(HttpResponse::NotFound().finish()))
        }
    } else {
        // if path conversion fails, probably client error
        Ok(Either::Right(HttpResponse::BadRequest().finish()))
    }
}

// Produce a thumbnail for the asset of the requested size.
async fn get_thumbnail(req: HttpRequest) -> actix_web::Result<HttpResponse> {
    // => /thumbnail/{w}/{h}/{id}
    let width: u32 = req.match_info().get("w").unwrap().parse().unwrap();
    let height: u32 = req.match_info().get("h").unwrap().parse().unwrap();
    let identifier: String = req.match_info().get("id").unwrap().to_owned();
    let etag_value = format!("{}:{}:{}", width, height, &identifier);
    let etag: header::EntityTag = header::EntityTag::new_strong(etag_value);
    if none_match(&etag, &req) {
        if let Ok(filepath) = blob_path(&identifier) {
            if filepath.exists() {
                let result = web::block(move || create_thumbnail(&filepath, width, height)).await?;
                match result {
                    Ok(data) => Ok(HttpResponse::Ok()
                        .content_type("image/jpeg")
                        .append_header((header::CONTENT_LENGTH, data.len() as u64))
                        .append_header((header::ETAG, etag))
                        .body(data)),
                    Err(_) => Ok(HttpResponse::TemporaryRedirect()
                        .append_header((header::LOCATION, "/public/placeholder.svg"))
                        .finish()),
                }
            } else {
                Ok(HttpResponse::NotFound().finish())
            }
        } else {
            // if path conversion fails, probably client error
            Ok(HttpResponse::BadRequest().finish())
        }
    } else {
        Ok(HttpResponse::NotModified().finish())
    }
}

// Returns true if `req` does not have an `If-None-Match` header matching `etag`.
fn none_match(etag: &header::EntityTag, req: &HttpRequest) -> bool {
    match req.get_header::<header::IfNoneMatch>() {
        Some(header::IfNoneMatch::Any) => false,
        Some(header::IfNoneMatch::Items(ref items)) => {
            for item in items {
                if item.weak_eq(etag) {
                    return false;
                }
            }
            true
        }
        None => true,
    }
}

// Receive the incoming file content as the body of the PUT request.
async fn put_asset(
    info: web::Path<String>,
    mut payload: web::Payload,
) -> actix_web::Result<HttpResponse> {
    use futures::StreamExt;
    use std::io::Write;

    if let Ok(filepath) = blob_path(&info) {
        if filepath.exists() {
            Ok(HttpResponse::Conflict().finish())
        } else {
            // store the file content to the given path
            let fp_clone = filepath.clone();
            // file operations are blocking, use threadpool
            let mut file = web::block(move || {
                let parent_dir = fp_clone.parent().expect("no parent");
                std::fs::create_dir_all(parent_dir)?;
                std::fs::File::create(fp_clone)
            })
            .await??;
            // the body is a stream of Bytes objects
            while let Some(chunk) = payload.next().await {
                let data = chunk?;
                // file operations are blocking, use threadpool
                file = web::block(move || file.write_all(&data).map(|_| file)).await??;
            }
            // ensure file is readable by backup programs and the like
            #[cfg(target_family = "unix")]
            {
                use std::fs::Permissions;
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&filepath, Permissions::from_mode(0o644))?;
            }
            info!("asset {} saved to {}", info, filepath.display());
            Ok(HttpResponse::Created().finish())
        }
    } else {
        // if path conversion fails, probably client error
        Ok(HttpResponse::BadRequest().finish())
    }
}

async fn delete_asset(info: web::Path<String>) -> actix_web::Result<HttpResponse> {
    if let Ok(filepath) = blob_path(&info) {
        if filepath.exists() {
            std::fs::remove_file(filepath)?;
            Ok(HttpResponse::Ok().finish())
        } else {
            Ok(HttpResponse::NotFound().finish())
        }
    } else {
        // if path conversion fails, probably client error
        Ok(HttpResponse::BadRequest().finish())
    }
}

#[actix_web::get("favicon.ico")]
async fn favicon() -> actix_web::Result<actix_files::NamedFile> {
    Ok(actix_files::NamedFile::open("./public/favicon.ico")?)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();
    std::fs::create_dir_all(ASSETS_PATH.as_path())?;
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_owned());
    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_owned());
    let addr = format!("{}:{}", host, port);
    info!("listening on {}", addr);
    HttpServer::new(move || {
        App::new()
            .wrap(actix_web::middleware::Logger::default())
            .service(
                web::resource("/liveness")
                    .route(web::get().to(HttpResponse::Ok))
                    .route(web::head().to(HttpResponse::Ok)),
            )
            .service(
                actix_files::Files::new("/public", "./public")
                    .use_etag(true)
                    .use_last_modified(true),
            )
            .service(favicon)
            .route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail))
            .route("/assets/{id}", web::get().to(get_asset))
            .route("/assets/{id}", web::head().to(get_asset))
            .route("/assets/{id}", web::put().to(put_asset))
            .route("/assets/{id}", web::delete().to(delete_asset))
            .route("/", web::get().to(index))
    })
    .bind(addr)?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::http::header::{self, ContentType};
    use actix_web::{App, test};

    #[actix_web::test]
    async fn test_index_get() {
        let app = test::init_service(App::new().route("/", web::get().to(index))).await;
        let req = test::TestRequest::default()
            .insert_header(ContentType::plaintext())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_get_asset_bad_encoding() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        let uri = "/assets/thisisnotbase64";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        )
    }

    #[actix_web::test]
    async fn test_get_asset_root_paths() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;

        if cfg!(unix) {
            let uri = "/assets/L3Jvb3Qvc2VjcmV0L2ZpbGVz"; // /root/secret/files
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }

        if cfg!(windows) {
            let uri = "/assets/XHdpbmRvd3M="; // \windows
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/Yzpcd2luZG93cw=="; // c:\windows
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/XFxzZXJ2ZXJcc2hhcmU="; // \\server\share
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }
    }

    #[actix_web::test]
    async fn test_get_asset_missing_file() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let uri = "/assets/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw==";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        )
    }

    #[actix_web::test]
    async fn test_get_asset_trailing_newline() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2YxdC5qcGcK is 2019-04-15/0830/f1t.jpg\n
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2YxdC5qcGcK";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");
    }

    #[actix_web::test]
    async fn test_get_asset_find_jpeg() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc= is 2019-04-15/0830/f1t.jpg
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc=";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");

        // retrieve a second time with the same ETag to get a 304 response
        let etag = resp.headers().get(header::ETAG).unwrap();
        let req = test::TestRequest::with_uri(uri)
            .insert_header(("If-None-Match", etag))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_MODIFIED
        );
    }

    #[actix_web::test]
    async fn test_get_asset_plain_text() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA==";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "text/plain; charset=utf-8");
    }

    #[actix_web::test]
    async fn test_get_asset_range_request() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA==";
        let req = test::TestRequest::with_uri(uri)
            .insert_header(("Range", "bytes=500-999"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::PARTIAL_CONTENT
        );
        let crange = resp.headers().get(header::CONTENT_RANGE).unwrap();
        #[cfg(unix)]
        assert_eq!(crange, "bytes 500-999/3129");
        #[cfg(windows)]
        assert_eq!(crange, "bytes 500-999/3138"); // Windows EOL
    }

    #[actix_web::test]
    async fn test_thumbnail_bad_encoding() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        let uri = "/thumbnail/480/320/thisisnotbase64";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        )
    }

    #[actix_web::test]
    async fn test_thumbnail_missing_file() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let uri = "/thumbnail/480/320/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw==";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        )
    }

    #[actix_web::test]
    async fn test_thumbnail_not_image() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/thumbnail/480/320/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA==";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.headers().get(header::LOCATION).unwrap(),
            "/public/placeholder.svg"
        );
    }

    #[actix_web::test]
    async fn test_placeholder_image() {
        let app = test::init_service(
            App::new().service(
                actix_files::Files::new("/public", "./public")
                    .use_etag(true)
                    .use_last_modified(true),
            ),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/public/placeholder.svg";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/svg+xml");
        // actix_files::Files does not return content-length header?
    }

    #[actix_web::test]
    async fn test_thumbnail_valid_image() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn is 2019-04-15/0830/fighting_kittens.jpg
        let uri = "/thumbnail/480/320/MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");

        // retrieve a second time with the same ETag to get a 304 response
        let etag = resp.headers().get(header::ETAG).unwrap();
        let req = test::TestRequest::with_uri(uri)
            .insert_header(("If-None-Match", etag))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_MODIFIED
        );
    }

    #[actix_web::test]
    async fn test_put_asset_bad_encoding() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;
        let req = test::TestRequest::with_uri("/assets/thisisnotbase64")
            .method(actix_web::http::Method::PUT)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        );
    }

    #[actix_web::test]
    async fn test_put_asset_root_paths() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;

        if cfg!(unix) {
            let uri = "/assets/L3Jvb3Qvc2VjcmV0L2ZpbGVz"; // /root/secret/files
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }

        if cfg!(windows) {
            let uri = "/assets/XHdpbmRvd3M="; // \windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/Yzpcd2luZG93cw=="; // c:\windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/XFxzZXJ2ZXJcc2hhcmU="; // \\server\share
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }
    }

    #[actix_web::test]
    async fn test_put_asset_conflict() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;
        let req = test::TestRequest::with_uri("/assets/MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc=")
            .method(actix_web::http::Method::PUT)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::CONFLICT
        );
    }

    fn checksum_file(infile: &Path) -> std::io::Result<String> {
        use sha2::{Digest, Sha256};
        let mut file = std::fs::File::open(infile)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let digest = hasher.finalize();
        Ok(format!("{:x}", digest))
    }

    #[actix_web::test]
    async fn test_put_asset_ok() {
        // clean up from previous failed runs
        let expected_path = Path::new("./tests/blobs/2019-04-15/0830/f2t.jpg");
        if expected_path.exists() {
            std::fs::remove_file(expected_path).expect("delete file");
        }

        //
        // request should look something like this:
        //
        // PUT /assets/BASE64-ENCODED-PATH HTTP/1.1
        // Content-Type: xxx/yyy
        // Content-Length: nnn
        //
        // [raw data of the incoming file]
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;
        let filepath = "./tests/fixtures/f2t.jpg";
        let payload = std::fs::read(filepath).expect("file read");

        // MjAxOS0wNC0xNS8wODMwL2YydC5qcGc= is 2019-04-15/0830/f2t.jpg
        let req = test::TestRequest::with_uri("/assets/MjAxOS0wNC0xNS8wODMwL2YydC5qcGc=")
            .method(actix_web::http::Method::PUT)
            .append_header((header::CONTENT_TYPE, "image/jpeg"))
            .append_header((header::CONTENT_LENGTH, payload.len()))
            .set_payload(payload)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::CREATED);
        assert!(expected_path.exists());
        let digest = checksum_file(expected_path).expect("checksum");
        assert_eq!(
            digest,
            "c52b9501d1037c50c8d20969a36a888b71310ff90ee557f813330144d8377b18"
        );
        std::fs::remove_file(expected_path).expect("delete file");
    }

    #[actix_web::test]
    async fn test_delete_asset_bad_encoding() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;
        let req = test::TestRequest::with_uri("/assets/thisisnotbase64")
            .method(actix_web::http::Method::DELETE)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        );
    }

    #[actix_web::test]
    async fn test_delete_asset_root_paths() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;

        if cfg!(unix) {
            let uri = "/assets/L3Jvb3Qvc2VjcmV0L2ZpbGVz"; // /root/secret/files
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }

        if cfg!(windows) {
            let uri = "/assets/XHdpbmRvd3M="; // \windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/Yzpcd2luZG93cw=="; // c:\windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/XFxzZXJ2ZXJcc2hhcmU="; // \\server\share
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }
    }

    #[actix_web::test]
    async fn test_delete_asset_missing_file() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let req = test::TestRequest::with_uri("/assets/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw==")
            .method(actix_web::http::Method::DELETE)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        );
    }

    #[actix_web::test]
    async fn test_delete_asset_ok() {
        let source_path = Path::new("./tests/fixtures/f2t.jpg");
        // use a different destination to avoid Windows "used by another process" error
        let target_path = Path::new("./tests/blobs/2019-04-15/0830/f3t.jpg");
        std::fs::copy(source_path, target_path).expect("file copy");

        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;
        // MjAxOS0wNC0xNS8wODMwL2YzdC5qcGc= is 2019-04-15/0830/f3t.jpg
        let req = test::TestRequest::with_uri("/assets/MjAxOS0wNC0xNS8wODMwL2YzdC5qcGc=")
            .method(actix_web::http::Method::DELETE)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        assert!(!target_path.exists());
    }
}
