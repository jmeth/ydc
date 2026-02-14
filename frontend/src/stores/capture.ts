/**
 * Capture state store.
 *
 * Tracks capture session status, the inference feed being subscribed to,
 * target dataset, capture configuration, and capture statistics.
 * Provides actions for starting/stopping capture, manual trigger,
 * and config updates via the API.
 */

import { defineStore } from 'pinia'
import { useApi } from '@/composables/useApi'
import type { CaptureStatusResponse, ManualCaptureResponse } from '@/types/api'

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
  split: string
  mode: 'raw' | 'inference' | null
  config: CaptureConfig
  stats: CaptureStats
}

export const useCaptureStore = defineStore('capture', {
  state: (): CaptureState => ({
    status: 'idle',
    inferenceFeedId: null,
    datasetName: null,
    split: 'train',
    mode: null,
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
      this.datasetName = null
      this.mode = null
      this.stats = { totalCaptures: 0, positiveCaptures: 0, negativeCaptures: 0 }
    },

    /**
     * Start a capture session.
     *
     * @param feedId - Feed ID to capture from
     * @param datasetName - Target dataset name
     * @param split - Target split (train/val/test)
     */
    async startCapture(feedId: string, datasetName: string, split = 'train') {
      const { post } = useApi()
      const res = await post<CaptureStatusResponse>('/capture/start', {
        feed_id: feedId,
        dataset_name: datasetName,
        split,
        capture_interval: this.config.captureInterval,
        negative_ratio: this.config.negativeRatio,
        confidence_threshold: this.config.confidenceThreshold,
      })
      this._applyStatus(res)
    },

    /** Stop the current capture session. */
    async stopCapture() {
      const { post } = useApi()
      await post('/capture/stop')
      this.reset()
    },

    /** Fetch current capture status from the backend. */
    async fetchStatus() {
      const { get } = useApi()
      try {
        const res = await get<CaptureStatusResponse>('/capture/status')
        this._applyStatus(res)
      } catch (err) {
        console.error('Failed to fetch capture status:', err)
      }
    },

    /** Manually trigger a single frame capture. */
    async triggerManualCapture(): Promise<ManualCaptureResponse> {
      const { post } = useApi()
      return post<ManualCaptureResponse>('/capture/trigger')
    },

    /**
     * Update capture configuration on the running session.
     *
     * @param config - Partial config fields to update
     */
    async updateConfig(config: Partial<CaptureConfig>) {
      const { put } = useApi()
      const body: Record<string, number> = {}
      if (config.captureInterval !== undefined) body.capture_interval = config.captureInterval
      if (config.negativeRatio !== undefined) body.negative_ratio = config.negativeRatio
      if (config.confidenceThreshold !== undefined) body.confidence_threshold = config.confidenceThreshold

      await put('/capture/config', body)
      Object.assign(this.config, config)
    },

    /**
     * Apply a CaptureStatusResponse to the store state.
     * Maps snake_case API fields to camelCase store fields.
     */
    _applyStatus(res: CaptureStatusResponse) {
      this.status = res.status as CaptureState['status']
      this.inferenceFeedId = res.feed_id
      this.datasetName = res.dataset_name
      this.split = res.split || 'train'
      this.mode = res.mode
      if (res.config) {
        this.config = {
          captureInterval: res.config.capture_interval,
          negativeRatio: res.config.negative_ratio,
          confidenceThreshold: res.config.confidence_threshold,
        }
      }
      if (res.stats) {
        this.stats = {
          totalCaptures: res.stats.total_captures,
          positiveCaptures: res.stats.positive_captures,
          negativeCaptures: res.stats.negative_captures,
        }
      }
    },
  },
})
