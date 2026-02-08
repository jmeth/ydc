/**
 * TypeScript types for domain models.
 *
 * Shared data structures used across stores, composables,
 * and components.
 */

/** A single object detection result. */
export interface Detection {
  classId: number
  className: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
}

/** Image metadata in a dataset. */
export interface ImageInfo {
  filename: string
  split: string
  width: number
  height: number
  annotated: boolean
  reviewStatus?: 'pending' | 'accepted' | 'rejected'
}

/** A single YOLO annotation. */
export interface Annotation {
  classId: number
  x: number
  y: number
  width: number
  height: number
  confidence?: number
  auto: boolean
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
  id: string
  name: string
  datasetName: string
  epochs: number
  metrics: Record<string, number>
  createdAt: string
  active: boolean
}
