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
