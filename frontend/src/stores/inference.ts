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
     * @param modelName - Optional model name or path
     * @param prompts - Optional YOLO-World class prompts
     */
    async startInference(feedId: string, modelName?: string, prompts?: Record<number, string[]>) {
      const { post } = useApi()
      try {
        // Convert prompts from Record<number, string[]> to flat string[]
        const flatPrompts = prompts
          ? Object.values(prompts).flat()
          : undefined
        const res = await post<StartInferenceResponse>('/inference/start', {
          source_feed_id: feedId,
          model_name: modelName,
          model_type: flatPrompts ? 'yolo_world' : 'fine_tuned',
          prompts: flatPrompts,
        })
        this.status = 'running'
        this.sourceFeedId = res.source_feed_id
        this.outputFeedId = res.output_feed_id
        this.modelId = res.model_name
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
        await post('/inference/stop', {
          output_feed_id: this.outputFeedId,
        })
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
          this.modelId = session.model_name
          this.prompts = {}
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
      await put('/inference/prompts', {
        output_feed_id: this.outputFeedId,
        prompts: Object.values(prompts).flat(),
      })
      this.prompts = prompts
    },

    /**
     * Switch the model on the active inference session.
     *
     * @param modelName - Model name or identifier
     */
    async switchModel(modelName: string) {
      const { put } = useApi()
      await put('/inference/model', {
        output_feed_id: this.outputFeedId,
        model_name: modelName,
      })
      this.modelId = modelName
    },
  },
})
