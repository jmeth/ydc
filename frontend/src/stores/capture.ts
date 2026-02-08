/**
 * Capture state store.
 *
 * Tracks capture session status, the inference feed being subscribed to,
 * target dataset, capture configuration, and capture statistics.
 */

import { defineStore } from 'pinia'

interface CaptureConfig {
  captureInterval: number
  negativeRatio: number
  confidenceThreshold: number
}

interface CaptureStats {
  totalCaptures: number
  positiveCaptures: number
  negativeCaptures: number
}

interface CaptureState {
  status: 'idle' | 'running' | 'paused'
  inferenceFeedId: string | null
  datasetName: string | null
  config: CaptureConfig
  stats: CaptureStats
}

export const useCaptureStore = defineStore('capture', {
  state: (): CaptureState => ({
    status: 'idle',
    inferenceFeedId: null,
    datasetName: null,
    config: {
      captureInterval: 2.0,
      negativeRatio: 0.2,
      confidenceThreshold: 0.3,
    },
    stats: {
      totalCaptures: 0,
      positiveCaptures: 0,
      negativeCaptures: 0,
    },
  }),

  actions: {
    /** Update capture stats from WebSocket event. */
    updateStats(stats: Partial<CaptureStats>) {
      Object.assign(this.stats, stats)
    },

    /** Reset store to idle state. */
    reset() {
      this.status = 'idle'
      this.inferenceFeedId = null
      this.stats = { totalCaptures: 0, positiveCaptures: 0, negativeCaptures: 0 }
    },
  },
})
