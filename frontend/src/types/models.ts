/**
 * TypeScript types for domain models.
 *
 * Shared data structures used across stores, composables,
 * and components.
 */

/**
 * A single object detection result.
 *
 * Uses bbox [x1, y1, x2, y2] in pixel coordinates,
 * matching the backend InferenceFrame detection format.
 */
export interface Detection {
  classId: number
  className: string
  confidence: number
  bbox: [number, number, number, number]
}

/** Image metadata in a dataset. */
export interface ImageInfo {
  filename: string
  split: string
  width: number
  height: number
  sizeBytes: number
  hasLabels: boolean
}

/** A single YOLO annotation (normalized 0-1 center format). */
export interface Annotation {
  classId: number
  x: number
  y: number
  width: number
  height: number
}

/** Training job configuration. */
export interface TrainingConfig {
  datasetName: string
  modelBase: string
  epochs: number
  batchSize: number
  imageSize: number
}

/** Feed information. */
export interface FeedInfo {
  id: string
  name: string
  type: 'raw' | 'inference'
  status: 'active' | 'inactive' | 'error'
  sourceFeedId?: string
}

/** Trained model information. */
export interface ModelInfo {
  name: string
  baseModel: string
  datasetName: string
  createdAt: number
  epochsCompleted: number
  bestMap50: number | null
  isActive: boolean
  path: string
}
