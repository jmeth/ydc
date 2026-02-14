/**
 * Inference state store.
 *
 * Tracks inference session status, the source/output feed IDs,
 * active model, prompts, and real-time performance stats.
 * Provides actions for starting/stopping inference, updating
 * prompts, and switching models via the API.
 */

import { defineStore } from 'pinia'
import { useApi } from '@/composables/useApi'
import type {
  StartInferenceResponse,
  InferenceStatusResponse,
} from '@/types/api'

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
      this.modelId = null
      this.prompts = {}
      this.stats = { fps: 0, detections: 0, inferenceTimeMs: 0 }
    },

    /**
     * Start an inference session on a feed.
     *
     * @param feedId - Source feed ID
     * @param modelPath - Optional model weights path
     * @param prompts - Optional YOLO-World class prompts
     */
    async startInference(feedId: string, modelPath?: string, prompts?: Record<number, string[]>) {
      const { post } = useApi()
      try {
        const res = await post<StartInferenceResponse>('/inference/start', {
          feed_id: feedId,
          model_path: modelPath,
          prompts,
        })
        this.status = 'running'
        this.sourceFeedId = res.source_feed_id
        this.outputFeedId = res.output_feed_id
        this.modelId = res.model
        if (prompts) this.prompts = prompts
      } catch (err) {
        this.status = 'error'
        throw err
      }
    },

    /** Stop the current inference session. */
    async stopInference() {
      const { post } = useApi()
      try {
        await post('/inference/stop')
        this.reset()
      } catch (err) {
        console.error('Failed to stop inference:', err)
        throw err
      }
    },

    /** Fetch current inference status from the backend. */
    async fetchStatus() {
      const { get } = useApi()
      try {
        const res = await get<InferenceStatusResponse>('/inference/status')
        const session = res.sessions[0]
        if (session) {
          this.status = session.status === 'running' ? 'running' : 'idle'
          this.sourceFeedId = session.source_feed_id
          this.outputFeedId = session.output_feed_id
          this.modelId = session.model
          this.prompts = session.prompts || {}
        } else {
          this.reset()
        }
      } catch (err) {
        console.error('Failed to fetch inference status:', err)
      }
    },

    /**
     * Update YOLO-World prompts on the active session.
     *
     * @param prompts - Updated class prompts mapping
     */
    async updatePrompts(prompts: Record<number, string[]>) {
      const { put } = useApi()
      await put('/inference/prompts', { prompts })
      this.prompts = prompts
    },

    /**
     * Switch the model on the active inference session.
     *
     * @param modelPath - Path to the new model weights
     */
    async switchModel(modelPath: string) {
      const { put } = useApi()
      await put('/inference/model', { model_path: modelPath })
      this.modelId = modelPath
    },
  },
})
