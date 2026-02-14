/**
 * Composable for subscribing to a video feed via WebSocket.
 *
 * Provides reactive refs for the current frame (base64), FPS,
 * and detection results. Automatically subscribes/unsubscribes
 * when the feedId ref changes.
 *
 * Detections use the backend bbox format: [x1, y1, x2, y2] in
 * pixel coordinates, with class_name, confidence, and class_id.
 *
 * @param feedId - Reactive ref to the feed ID to subscribe to
 */

import { ref, watch, type Ref } from 'vue'
import { useWebSocket } from './useWebSocket'
import type { Detection } from '@/types/models'
import type { WsDetection } from '@/types/websocket'

export function useVideoStream(feedId: Ref<string | null>) {
  const frame = ref<string | null>(null)
  const fps = ref(0)
  const detections = ref<Detection[]>([])

  const { lastMessage, send, isConnected } = useWebSocket('/ws/video')

  // Subscribe to feed changes
  watch(
    feedId,
    (newId, oldId) => {
      if (oldId) {
        send({ action: 'unsubscribe', feed_id: oldId })
      }
      if (newId) {
        send({ action: 'subscribe', feed_id: newId })
      }
    },
  )

  // Re-subscribe when reconnected
  watch(isConnected, (connected) => {
    if (connected && feedId.value) {
      send({ action: 'subscribe', feed_id: feedId.value })
    }
  })

  // Process incoming frames
  watch(lastMessage, (msg: unknown) => {
    const message = msg as Record<string, unknown> | null
    if (message?.type === 'frame' && message.feed_id === feedId.value) {
      frame.value = message.data as string
      fps.value = (message.fps as number) || 0

      // Map WsDetection (snake_case) to Detection (camelCase)
      const rawDetections = (message.detections as WsDetection[]) || []
      detections.value = rawDetections.map((d) => ({
        classId: d.class_id,
        className: d.class_name,
        confidence: d.confidence,
        bbox: d.bbox,
      }))
    }
  })

  return { frame, fps, detections, isConnected }
}
