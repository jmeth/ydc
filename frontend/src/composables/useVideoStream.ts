/**
 * Composable for subscribing to a video feed via WebSocket.
 *
 * Provides reactive refs for the current frame (base64), FPS,
 * and detection results. Automatically subscribes/unsubscribes
 * when the feedId ref changes.
 *
 * @param feedId - Reactive ref to the feed ID to subscribe to
 */

import { ref, watch, type Ref } from 'vue'
import { useWebSocket } from './useWebSocket'
import type { Detection } from '@/types/models'

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
      detections.value = (message.detections as Detection[]) || []
    }
  })

  return { frame, fps, detections, isConnected }
}
