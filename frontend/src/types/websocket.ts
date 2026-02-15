/**
 * TypeScript types for WebSocket message protocol.
 *
 * Defines the shape of messages sent between the Vue frontend
 * and the FastAPI WebSocket endpoints.
 */

/** Base shape for all WebSocket messages from the server. */
export interface WsMessage {
  type: string
}

/** Video frame message from /ws/video. */
export interface WsFrameMessage extends WsMessage {
  type: 'frame'
  feed_id: string
  data: string
  detections: WsDetection[]
  timestamp: number
}

/**
 * Detection within a video frame.
 *
 * Matches the backend's InferenceFrame detection format:
 * class_name, confidence, bbox [x1,y1,x2,y2] in pixel coords, class_id.
 */
export interface WsDetection {
  class_id: number
  class_name: string
  confidence: number
  bbox: [number, number, number, number]
}

/** Subscription confirmation from /ws/video. */
export interface WsSubscribedMessage extends WsMessage {
  type: 'subscribed'
  feed_id: string
}

/** Pong response from /ws/events. */
export interface WsPongMessage extends WsMessage {
  type: 'pong'
}

/** Training progress event from /ws/events. */
export interface WsTrainingProgressMessage extends WsMessage {
  type: 'training.progress'
  current_epoch: number
  total_epochs: number
  progress_pct: number
  metrics: Record<string, number>
  loss: number
}

/** Training completed event from /ws/events. */
export interface WsTrainingCompletedMessage extends WsMessage {
  type: 'training.completed'
  model_name: string
  epochs_completed: number
  best_map50: number | null
}

/** Training error event from /ws/events. */
export interface WsTrainingErrorMessage extends WsMessage {
  type: 'training.error'
  error: string
}

/** Capture event from /ws/events. */
export interface WsCaptureEventMessage extends WsMessage {
  type: 'capture.event'
  capture_type: 'positive' | 'negative'
  image: string
  dataset: string
}

/** Notification pushed from the server via /ws/events. */
export interface WsNotificationMessage extends WsMessage {
  type: 'notification'
  id: string
  notification_type: 'toast' | 'banner' | 'alert' | 'status'
  level: 'info' | 'success' | 'warning' | 'error'
  category: string
  title: string
  message: string
  timestamp: string
  data: Record<string, unknown> | null
}

/** Client-to-server action messages. */
export interface WsClientAction {
  action: string
  [key: string]: unknown
}
