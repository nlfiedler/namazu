use http_body_util::Empty;
use hyper::Request;
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use std::env;
use std::process::exit;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let port = env::var("PORT").unwrap_or("3000".into());
    let path = env::var("HEALTHCHECK_PATH").unwrap_or("/".into());
    let url = format!("http://localhost:{port}{path}").parse::<hyper::Uri>()?;
    let authority = url.authority().unwrap().clone();
    let req = Request::builder()
        .uri(url)
        .method("HEAD")
        .header(hyper::header::HOST, authority.as_str())
        .body(Empty::<Bytes>::new())?;
    let client = Client::builder(TokioExecutor::new()).build_http();
    let response = client.request(req).await?;
    let status_code = response.status();
    if status_code.is_client_error() || status_code.is_server_error() {
        exit(1)
    }
    exit(0)
}
