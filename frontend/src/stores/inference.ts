/**
 * Inference state store.
 *
 * Tracks inference session status, the source/output feed IDs,
 * active model, prompts, and real-time performance stats.
 */

import { defineStore } from 'pinia'

interface InferenceStats {
  fps: number
  detections: number
  inferenceTimeMs: number
}

interface InferenceState {
  status: 'idle' | 'running' | 'error'
  sourceFeedId: string | null
  outputFeedId: string | null
  modelId: string | null
  prompts: Record<number, string[]>
  stats: InferenceStats
}

export const useInferenceStore = defineStore('inference', {
  state: (): InferenceState => ({
    status: 'idle',
    sourceFeedId: null,
    outputFeedId: null,
    modelId: null,
    prompts: {},
    stats: { fps: 0, detections: 0, inferenceTimeMs: 0 },
  }),

  actions: {
    /** Update real-time inference stats from WebSocket. */
    updateStats(stats: Partial<InferenceStats>) {
      Object.assign(this.stats, stats)
    },

    /** Reset store to idle state. */
    reset() {
      this.status = 'idle'
      this.sourceFeedId = null
      this.outputFeedId = null
      this.stats = { fps: 0, detections: 0, inferenceTimeMs: 0 }
    },
  },
})
