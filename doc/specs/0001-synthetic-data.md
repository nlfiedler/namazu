# Namazu: Synthetic Data Endpoint

This spec describes a new feature for the **Namazu** blob store: an HTTP endpoint that runs **image classification** (label tagging) and **face recognition** on stored image blobs and returns the results as JSON.

This is a handoff document — Namazu is implemented in Rust and is developed independently from its primary consumer (Tanuki, a TypeScript digital-asset-management server). The endpoint defined here is the contract Tanuki will call. The implementer should have working knowledge of Rust, axum/actix/whatever HTTP framework Namazu currently uses, and basic familiarity with ML inference (CPU-only, no GPU assumed).

## Background

Namazu stores image and video blobs and serves them by id over HTTP. Tanuki already calls a metadata endpoint (`GET /metadata/:blobId`) on Namazu to extract EXIF / `ffprobe` data without streaming bytes back to Tanuki. This spec adds an analogous endpoint for ML-derived synthetic data: classification labels and detected faces (with feature embeddings).

Pushing inference into Namazu, instead of streaming image bytes to Tanuki and running inference there, avoids large transfers and lets the operator scale inference independently from the application server.

## Goals

- One endpoint per asset that returns **both** classification labels and face detections + embeddings in a single response.
- CPU-only inference. **No GPU dependency.**
- Stable response shape across model upgrades — Tanuki should not require a redeploy when Namazu swaps in a new model, as long as the JSON contract holds.
- Bounded response size (count caps + byte cap) so a pathological input can never return megabytes.

## Non-goals

- Identity assignment / clustering. Namazu returns raw face embeddings; Tanuki does the clustering and naming.
- Persistence. Namazu does not cache results; each call re-runs inference. (If caching becomes necessary, that's a future addition.)
- Video. Only image blobs are processed. Non-images return `204 No Content`.
- User-facing UI. Namazu is a service; the UI lives in Tanuki.
- Object detection / bounding boxes for non-face content. Tanuki uses an image classifier (MobileNetV2), not a detector, for label tagging — see below. There is therefore no need for object bboxes.

## Endpoint

```
POST /synthetic/:blobId
```

`:blobId` is the same id used by existing Namazu routes (e.g. `/blobs/:blobId`, `/metadata/:blobId`).

### Request

No body. No query parameters. The endpoint is idempotent: calling it twice produces functionally equivalent results (modulo nondeterminism in the underlying models).

### Response: 200 OK

`Content-Type: application/json`

```json
{
  "labels": [
    { "name": "beach", "score": 0.91 },
    { "name": "palm tree", "score": 0.62 }
  ],
  "faces": [
    {
      "bbox": [x, y, w, h],
      "embedding": "<base64-encoded little-endian Float32Array, 512 floats>",
      "thumbnail": "<base64-encoded JPEG>",
      "score": 0.97,
      "model_version": "mobilefacenet-v1"
    }
  ],
  "model_versions": {
    "labels": "mobilenetv2-v1",
    "faces": "mobilefacenet-v1"
  },
  "truncated": false
}
```

#### Field semantics

- `labels[].name` — **curated display label** (post-mapping, see "Label vocabulary and curation" below). Lowercase English, may contain spaces (`"palm tree"`).
- `labels[].score` — classifier confidence, `[0.0, 1.0]`. After de-duplication (multiple raw ImageNet labels collapsing to one display label), the surviving score is the maximum raw score among the merged entries.
- `faces[].bbox` — `[x, y, width, height]` in **displayed-orientation** pixel coordinates of the source image (i.e. after applying EXIF orientation; see "Orientation" below). Floating-point allowed.
- `faces[].embedding` — base64 of a raw little-endian `f32` array. **MobileFaceNet output: 512 floats = 2048 bytes raw → ~2732 bytes base64.** The vector **must be L2-normalized** so consumers can use cosine similarity by simple dot product.
- `faces[].thumbnail` — base64-encoded JPEG of the cropped, aligned face, **~128 px on the long edge** (target 128, accept ±32). The crop is the same aligned face that was fed to MobileFaceNet (5-point landmark warp to the standard ArcFace reference template), not just a raw bbox crop. JPEG quality ~85.
- `faces[].score` — face detector confidence, `[0.0, 1.0]`.
- `faces[].model_version` — the model that produced this embedding. Must agree with `model_versions.faces` (e.g. `mobilefacenet-v1`). Embeddings from different `model_version` values are not directly comparable.
- `model_versions.labels` / `model_versions.faces` — version identifiers for the active label/face models. These should change whenever the model weights change in a way that affects output semantics. Use a stable, short string (e.g. `mobilenetv2-v1`, `mobilefacenet-v1`). Tanuki uses these as opaque tags and as part of the contract that face embeddings produced by Namazu and Tanuki's local detector are byte-comparable when their `model_version` values match.
- `truncated` — `true` if either list was clipped by the count caps below; otherwise `false`. When `true`, the items that remain are the top-N by descending `score`.

#### Ordering

- `labels` and `faces` should be returned **sorted by descending `score`**. Tanuki picks the primary label as `labels[0].name`.

#### Sort stability for ties

If two faces have identical scores, break ties by larger bbox area. If two labels have identical scores, break ties alphabetically by `name`. This produces deterministic output for fixture-based testing.

### Response: 204 No Content

Returned when the blob is not an image — i.e. its detected media type is not `image/*`. No body. Tanuki treats this as "nothing to do."

### Response: 404 Not Found

Returned when no blob exists for the given `:blobId`.

### Response: 4xx Bad Request

Returned when the blob exists and is an image but inference fails (corrupt JPEG, unsupported pixel format, etc.). Body:

```json
{ "error": "human-readable description", "code": "decode_failed" }
```

Suggested `code` values (extend as needed):

- `decode_failed` — image could not be decoded
- `inference_failed` — model crashed or timed out
- `too_large` — image dimensions exceed processing limit (see Limits)

### Response: 5xx

Reserved for genuine server errors (out of memory, model not loaded, etc.). The body should match the 4xx shape but with appropriate `code` like `model_unavailable`.

## Label vocabulary and curation

The image classifier (MobileNetV2) emits raw **ImageNet-1000** labels. Many of these are too granular, redundant, or noisy for end users ("tabby cat" vs "Egyptian cat", "Granny Smith", "comic book"). Namazu **must apply a curation mapping** before returning `labels[]`.

### The labels-map file

A JSON file, `labels-map.json`, ships with Namazu (provided by Tanuki — see "Deployment notes"). Each entry has three fields:

- `raw` — the original ImageNet class name (string)
- `label` — the curated display label (string), or `null` to drop this class entirely
- `category` — one of: `animal`, `plant`, `food`, `nature`, `person`, `vehicle`, `building`, `clothing`, `furniture`, `instrument`, `electronics`, `tool`, `weapon`, `container`, `sport`, `household`, `decoration`, `object`

The same file is used by the Tanuki local detector. Both backends produce identical output for the same image.

If `labels-map.json` is missing or unparseable at startup, treat it as a fatal startup error.

### Curation pipeline

Apply these steps in order to the raw classifier output:

1. Take all 1000 softmax scores.
2. **Drop scores below 0.05.** This floors out the long noise tail before any further work; without it, low-confidence ImageNet classes that happen to map to common display labels would inflate those labels' final scores via the dedup-max step.
3. Look up each surviving raw class in `labels-map.json`.
4. Drop entries whose `label` is `null`.
5. **Drop entries with `category == "person"`.** The three person entries (`baseball player`, `groom`, `diver`) overlap semantically with the face-recognition output and would just create noise in the labels list.
6. De-duplicate by display label, keeping the maximum raw score per display label.
7. Sort by score descending. Apply the label count cap (see "Limits").

The `category` field is **internal** in v1 — used only for the person filter in step 5. It is **not** included in the response. The field is reserved for future use (e.g. a category-grouped Labels page in Tanuki).

### MobileNetV2 input preprocessing

The canonical ONNX Model Zoo preprocessing recipe is required, byte-for-byte. Both backends apply the same transform; deviation causes silent accuracy loss and breaks cross-backend consistency.

- Decode to RGB.
- Resize so the shorter edge is **256 px**, preserving aspect ratio.
- Center-crop to **224 × 224**.
- Convert to float32 in `[0, 1]`.
- Normalize per-channel: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Layout: NCHW, shape `[1, 3, 224, 224]`.

## Orientation

Source images may carry an EXIF `Orientation` tag (1–8). All face bounding boxes and crops in the response **must be expressed in displayed orientation** — that is, the orientation a user sees when the image is rendered correctly. For orientations 5, 6, 7, 8, the source image is rotated/flipped before inference, or equivalently the bounding boxes are transformed post-inference. Tanuki stores `width`/`height` in displayed orientation (see Tanuki spec 0003), and bbox coordinates must align with those dimensions.

Image classification output (`labels[]`) is not orientation-sensitive — labels describe content regardless of rotation — but the input image should still be rotated to displayed orientation before classification, because some models are sensitive to upside-down inputs.

## Limits

- **Per-image processing timeout**: 30 seconds. Beyond that, return `408 Request Timeout` (or treat as `inference_failed` with a long timeout — implementer's choice).
- **Maximum input dimensions**: 8000×8000 pixels. Larger images return `413` with `code: "too_large"`. (Tanuki can downscale before posting if necessary, but at present it does not.)
- **Label count cap**: keep the top **20** curated labels by score; drop the rest.
- **Face count cap**: keep the top **20** detections by score; drop the rest.
- **Total response size cap**: **1 MB**. If the response would exceed 1 MB after the count caps are applied, drop additional faces (which carry the largest payload — the thumbnails) until the response fits. If even one face cannot fit, return the response with zero faces and `truncated: true`.

Whenever the count caps or the byte cap cause drops, set `"truncated": true`. Tanuki uses this to know that more detections existed but were trimmed.

## Models

The model files are **fixed** — both Namazu and Tanuki's local detector load the same ONNX files so labels and face embeddings are comparable across backends. The Tanuki repo is the canonical source and hosts the files via GitHub Releases (see "Model file management" below).

### Image classification (labels)

- **MobileNetV2** (`mobilenet_v2.onnx`, ~14 MB) — ImageNet-1000-trained, ships as an official ONNX from the ONNX Model Zoo. Roughly comparable to PhotoPrism's NASNet-Mobile in accuracy for our purposes, materially smaller, and requires no model-conversion ceremony.

Suggested Rust inference crates (pure-Rust, CPU-only):

- [`tract`](https://crates.io/crates/tract) — pure-Rust ONNX inference. Recommended.
- [`candle-core`](https://crates.io/crates/candle-core) — Hugging Face's Rust ML framework; alternative.
- [`ort`](https://crates.io/crates/ort) — Rust bindings to ONNX Runtime, if you want the official ORT implementation (adds a native dep).

### Face detection + embedding

**The exact model files are fixed** — both Namazu and Tanuki's local detector must use the same ONNX files so that embeddings are byte-comparable across backends:

- **Face detection**: `scrfd_2.5g.onnx` (~3 MB). Produces face bounding boxes and 5-point landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner).
- **Face embedding**: `mobilefacenet.onnx` (~14 MB, ArcFace-trained). Produces 512-dimensional L2-normalized embeddings from aligned face crops.

The pipeline per detected face is: SCRFD → 5 landmarks → affine warp to the standard ArcFace 112×112 reference template → MobileFaceNet → 512-dim embedding (L2-normalized).

**SCRFD inference parameters** (InsightFace defaults — both backends agree):

- Detection score threshold: **0.5** (discard candidates below this confidence).
- NMS IoU threshold: **0.4**.

These are well-characterized defaults; do not tune in v1. They may be promoted to env-var knobs later if real usage shows they need to be.

Suggested Rust inference crates (pure-Rust or near-pure-Rust, CPU-only):

- [`tract`](https://crates.io/crates/tract) — pure-Rust ONNX inference. Recommended.
- [`ort`](https://crates.io/crates/ort) — Rust bindings to ONNX Runtime, if you want the official ORT implementation (adds a native dep).
- [`face_id`](https://crates.io/crates/face_id) — may be usable as a higher-level wrapper if it accepts custom ONNX model paths (i.e. lets you supply `scrfd_2.5g.onnx` and `mobilefacenet.onnx` instead of its bundled defaults). If it can't, drop down to `tract`/`ort` directly.

For image decoding and the affine warp, use [`image`](https://crates.io/crates/image) plus a small affine-transform routine (the reference 112×112 template coordinates are well-known and fixed; the warp is a standard `imageproc`-style operation).

Bundle all model weights into the Namazu deployment artifact (Docker image, etc.) so that first-run does not require a network fetch.

### Model file management

Model weights are **not committed to the Namazu repo** and are not assumed to be supplied via Docker volumes. Instead, the repo carries a small `model-manifest.json` that lists each file by URL + SHA256, and `build.rs` downloads any missing or stale files into a workspace `models/` directory (which is gitignored).

The manifest is **byte-identical to the corresponding manifest in the Tanuki repo**:

```json
{
  "version": "models-v1",
  "files": [
    { "name": "mobilenet_v2.onnx",   "url": "https://github.com/<owner>/tanuki/releases/download/models-v1/mobilenet_v2.onnx",   "sha256": "<hex>", "bytes": 14000000 },
    { "name": "scrfd_2.5g.onnx",     "url": "https://github.com/<owner>/tanuki/releases/download/models-v1/scrfd_2.5g.onnx",     "sha256": "<hex>", "bytes": 3200000 },
    { "name": "mobilefacenet.onnx",  "url": "https://github.com/<owner>/tanuki/releases/download/models-v1/mobilefacenet.onnx",  "sha256": "<hex>", "bytes": 5100000 },
    { "name": "labels-map.json",     "url": "https://github.com/<owner>/tanuki/releases/download/models-v1/labels-map.json",     "sha256": "<hex>", "bytes": 80000 }
  ]
}
```

The Tanuki repo is the canonical source; Namazu's copy of the manifest must match. When Tanuki cuts a new `models-vN` release, the Namazu manifest is updated in the same coordinated change. SHA256 verification on both sides is the early-warning signal that catches drift — if the two manifests fall out of sync, the fetch will fail loudly on whichever side is stale.

#### `build.rs`

`build.rs` should:

1. Read `model-manifest.json`.
2. For each entry, check `models/<name>` on disk:
   - If present and SHA256 matches, skip.
   - Otherwise, download (using `reqwest` or `ureq`) to `models/<name>.tmp`, verify SHA256, atomic-rename to `models/<name>`.
3. Emit `cargo:rerun-if-changed=model-manifest.json` so manifest changes trigger a fresh fetch.
4. Fail the build with a clear error if any download or hash check fails.

The runtime code loads ONNX files from `models/` relative to `CARGO_MANIFEST_DIR` (development) or alongside the binary (release). Tests pick up the same files.

#### `.gitignore`

```
/models/
```

#### CI caching

GitHub Actions / equivalent should cache `models/` keyed by the manifest's hash so the fetch happens once per `models-vN` revision across runs.

### Initialization

Load all models and `labels-map.json` once at process start. Failing to load any required artifact should be a fatal startup error — better to fail fast than to discover the problem on the first request.

## Determinism

The pixel-level outputs of these models are not bit-stable across hardware (BLAS implementations vary), but the high-level outputs (top label set, presence of faces, approximate bboxes) should be stable. Tests should tolerate small float deltas rather than exact equality.

## Concurrency

Inference is CPU-bound. Cap concurrent inferences by available cores (e.g. `num_cpus::get()` workers behind a semaphore). Beyond that, queue requests; a long queue is preferable to thrashing the CPU.

## Testing

Tests to add:

1. **Golden-file label classification** — known input images produce expected curated label sets (modulo ordering and small score wobble). Suggested fixtures: one beach scene, one indoor portrait, one ambiguous image (low-confidence outputs).
2. **Curation behavior** — given a fixed `labels-map.json`, verify that raw ImageNet labels map to expected display labels, that `null`-mapped raw labels are dropped, and that duplicate display labels are merged by max score.
3. **Golden-file face detection** — known portraits return the expected number of faces with bboxes within tolerance.
4. **Embedding stability** — the same input image produces embeddings that are equal within a small tolerance across runs (verify L2-normalization while you're at it).
5. **Non-image input** — POST against a video blob returns 204; against a PDF returns 204.
6. **Unknown blob** — POST against a random id returns 404.
7. **Corrupt image** — POST against a truncated JPEG returns 4xx with `code: "decode_failed"`.
8. **Truncation** — synthesize a pathological case (more than 20 labels or 20 faces — easiest: feed the truncation routine more than the cap directly in a unit test) and verify `truncated: true` + correct count.
9. **Size cap** — verify that responses near the 1 MB cap have additional faces dropped (not labels — labels are tiny), and `truncated: true` is set.
10. **Orientation** — verify that face bboxes returned for an EXIF-rotated input align with the displayed image. Suggested fixture: same image with `Orientation=1` and `Orientation=6` should produce equivalent bboxes when both are interpreted in displayed orientation.

## Integration with the existing `/metadata/:blobId` route

`/synthetic/:blobId` is a sibling of `/metadata/:blobId`, not a replacement. The two are called independently by Tanuki and have unrelated request/response shapes.

## Deployment notes

- Add model-weight files to the Docker image (or document the volume mount):
  - `mobilenet_v2.onnx` for label classification
  - `scrfd_2.5g.onnx` for face detection
  - `mobilefacenet.onnx` for face embedding
- **Ship `labels-map.json` with the deployment artifact.** Tanuki provides the canonical file (located in the Tanuki repo at `server/data/synthetic/labels-map.json`); the Namazu build script should copy it in, or it can be baked into the Docker image. If both sides drift, label outputs will diverge.
- The face ONNX files are likewise shared with Tanuki and must match byte-for-byte across the two deployments to preserve cross-backend embedding compatibility. Tanuki repo will hold the canonical copies.
- Document any new system packages required by the chosen inference crate. `tract` is pure-Rust and needs none; `ort` needs `libstdc++` and a recent `glibc`.
- Update `Cargo.toml` with the chosen ML crates; rebuild verifies they link CPU-only.
- The endpoint adds a meaningful steady-state memory baseline (model weights live in RAM). Expect ~60 MB extra resident memory total (MobileNetV2 ≈ 14 MB + SCRFD-2.5g ≈ 3 MB + MobileFaceNet ≈ 14 MB plus the ORT runtime and per-request working buffers); document this in the README so operators are not surprised.

## Out-of-scope but worth flagging

- **Result caching.** Tanuki currently treats each call as fresh inference. If repeated calls for the same blob become common, a small LRU cache keyed by blob id is reasonable, but is not required by Tanuki at this time.
- **Streaming / chunked responses.** All current sizes fit comfortably under 1 MB; no need for chunking.
- **Authentication.** Whatever auth scheme protects `/metadata/:blobId` applies to `/synthetic/:blobId` unchanged.

## Acceptance

The work is done when:

- `POST /synthetic/:blobId` returns the documented JSON for a representative test fixture set.
- All response codes (200, 204, 404, 4xx) are reachable and tested.
- Curation mapping is applied and verified by tests.
- Truncation behavior is verified.
- Models and `labels-map.json` are bundled with the deployment artifact so a fresh deploy serves requests immediately.
- A short README section documents the new endpoint, the memory footprint, and any new system dependencies.

Tanuki will then enable its Namazu detector path by setting `NAMAZU_URL` and verify end-to-end that the People and Labels pages populate correctly.
