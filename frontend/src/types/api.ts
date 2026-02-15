/**
 * TypeScript types for API request/response shapes.
 *
 * These mirror the Pydantic models on the backend for
 * type-safe frontend-backend communication.
 */

/** Standard health/status response from /api/health. */
export interface StatusResponse {
  status: string
  version: string
}

/** Standard error response body. */
export interface ErrorResponse {
  error: string
  detail?: string
}

/** Response from 501 stub endpoints. */
export interface NotImplementedResponse {
  error: string
  detail: string
}

/** System config response from /api/system/config. */
export interface SystemConfig {
  data_dir: string
  models_dir: string
  capture_interval: number
  negative_ratio: number
  confidence_threshold: number
  training_epochs: number
  training_batch_size: number
  training_image_size: number
}

// --- Feed API ---

/** Feed info as returned by GET /api/feeds. */
export interface FeedInfoResponse {
  feed_id: string
  feed_type: string
  source: string
  name: string
  status: string
  fps: number
  resolution: [number, number] | null
  frame_count: number
}

/** Request body for POST /api/feeds. */
export interface CreateFeedRequest {
  feed_type: string
  source: string
  name?: string
  buffer_size?: number
}

/** Response for GET /api/feeds. */
export interface FeedListResponse {
  feeds: FeedInfoResponse[]
  count: number
}

// --- Dataset API ---

/** Response for a single dataset's metadata. */
export interface DatasetResponse {
  name: string
  path: string
  classes: string[]
  num_images: Record<string, number>
  created_at: number
  modified_at: number
}

/** Response for GET /api/datasets. */
export interface DatasetListResponse {
  datasets: DatasetResponse[]
  count: number
}

/** Response for a single image's metadata. */
export interface ImageResponse {
  filename: string
  split: string
  width: number
  height: number
  size_bytes: number
  has_labels: boolean
}

/** Response for GET /api/datasets/{name}/images. */
export interface ImageListResponse {
  images: ImageResponse[]
  count: number
}

/** Single YOLO-format annotation from the API. */
export interface AnnotationResponse {
  class_id: number
  x: number
  y: number
  width: number
  height: number
}

/** Response for GET /api/datasets/{name}/labels/{split}/{file}. */
export interface LabelsResponse {
  filename: string
  split: string
  annotations: AnnotationResponse[]
}

/** Response for GET /api/datasets/{name}/prompts. */
export interface PromptsResponse {
  prompts: Record<number, string[]>
}

// --- Inference API ---

/** Request body for POST /api/inference/start. */
export interface StartInferenceRequest {
  source_feed_id: string
  model_name?: string
  model_type?: string
  prompts?: string[]
  confidence_threshold?: number
}

/** Response for POST /api/inference/start. */
export interface StartInferenceResponse {
  source_feed_id: string
  output_feed_id: string
  model_name: string
  model_type: string
  status: string
}

/** Single inference session status. */
export interface InferenceSessionStatus {
  source_feed_id: string
  output_feed_id: string
  model_name: string
  model_type: string
  prompts: string[] | null
  confidence_threshold: number
  frames_processed: number
  avg_inference_ms: number
  last_inference_ms: number
  status: string
}

/** Response for GET /api/inference/status. */
export interface InferenceStatusResponse {
  sessions: InferenceSessionStatus[]
  count: number
}

// --- Capture API ---

/** Response for GET /api/capture/status and POST /api/capture/start. */
export interface CaptureStatusResponse {
  status: string
  feed_id: string | null
  dataset_name: string | null
  split: string | null
  mode: 'raw' | 'inference' | null
  config: {
    capture_interval: number
    negative_ratio: number
    confidence_threshold: number
  } | null
  stats: {
    total_captures: number
    positive_captures: number
    negative_captures: number
  } | null
}

/** Response for POST /api/capture/trigger. */
export interface ManualCaptureResponse {
  filename: string
  split: string
  dataset_name: string
  num_detections: number
}

// --- Training API ---

/**
 * Data augmentation overrides for YOLO training.
 * All fields optional — omitted fields use ultralytics defaults.
 */
export interface AugmentationConfig {
  // Color space
  hsv_h?: number   // 0.0-1.0, default 0.015
  hsv_s?: number   // 0.0-1.0, default 0.7
  hsv_v?: number   // 0.0-1.0, default 0.4
  // Geometric
  degrees?: number     // 0-180, default 0.0
  translate?: number   // 0.0-1.0, default 0.1
  scale?: number       // 0.0-1.0, default 0.5
  shear?: number       // -180 to 180, default 0.0
  perspective?: number // 0.0-0.001, default 0.0
  flipud?: number      // 0.0-1.0, default 0.0
  fliplr?: number      // 0.0-1.0, default 0.5
  bgr?: number         // 0.0-1.0, default 0.0
  // Advanced
  mosaic?: number      // 0.0-1.0, default 1.0
  mixup?: number       // 0.0-1.0, default 0.0
  copy_paste?: number  // 0.0-1.0, default 0.0
  erasing?: number     // 0.0-0.9, default 0.4
  auto_augment?: 'randaugment' | 'autoaugment' | 'augmix' | null
}

/** Response for training status. */
export interface TrainingStatusResponse {
  job_id: string
  status: string
  current_epoch: number
  total_epochs: number
  progress_pct: number
  metrics: Record<string, number>
  dataset_name: string
  base_model: string
  model_name: string
  started_at: number | null
  completed_at: number | null
  error: string | null
}

/** Single entry in training history. */
export interface TrainingHistoryEntry {
  job_id: string
  model_name: string
  dataset_name: string
  base_model: string
  status: string
  epochs_completed: number
  best_map50: number | null
  started_at: number | null
  completed_at: number | null
}

/** Response for GET /api/training/history. */
export interface TrainingHistoryResponse {
  jobs: TrainingHistoryEntry[]
  count: number
}

/** Response for a single trained model. */
export interface ModelResponse {
  name: string
  base_model: string
  dataset_name: string
  created_at: number
  epochs_completed: number
  best_map50: number | null
  is_active: boolean
  path: string
}

/** Response for GET /api/models. */
export interface ModelListResponse {
  models: ModelResponse[]
  count: number
}

/** Response for POST /api/models/import (same shape as ModelResponse). */
export type ImportModelResponse = ModelResponse

/** Request body for POST /api/models/pretrained. */
export interface DownloadPretrainedRequest {
  model_id: string
  name?: string
}

/** Request body for PUT /api/datasets/{name}. */
export interface UpdateDatasetRequest {
  classes: string[]
}

/** Generic message response. */
export interface MessageResponse {
  message: string
}
