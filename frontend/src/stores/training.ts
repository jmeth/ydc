/**
 * Training state store.
 *
 * Tracks training job status, progress updates from WebSocket,
 * configuration, error state, job history, and model list.
 * Provides actions for training lifecycle and model management via the API.
 */

import { defineStore } from 'pinia'
import { useApi } from '@/composables/useApi'
import type { TrainingConfig, ModelInfo } from '@/types/models'
import type {
  TrainingStatusResponse,
  TrainingHistoryResponse,
  TrainingHistoryEntry,
  ModelListResponse,
  ModelResponse,
  MessageResponse,
  DownloadPretrainedRequest,
  AugmentationConfig,
  ImportModelResponse,
} from '@/types/api'

interface TrainingProgress {
  epoch: number
  totalEpochs: number
  progressPct: number
  loss: number
  eta: string | null
  metrics: Record<string, number>
}

interface TrainingState {
  status: 'idle' | 'training' | 'completed' | 'error'
  jobId: string | null
  progress: TrainingProgress
  config: TrainingConfig | null
  error: string | null
  history: TrainingHistoryEntry[]
  models: ModelInfo[]
}

export const useTrainingStore = defineStore('training', {
  state: (): TrainingState => ({
    status: 'idle',
    jobId: null,
    progress: {
      epoch: 0,
      totalEpochs: 100,
      progressPct: 0,
      loss: 0,
      eta: null,
      metrics: {},
    },
    config: null,
    error: null,
    history: [],
    models: [],
  }),

  actions: {
    /** Update training progress from WebSocket event. */
    updateProgress(progress: Partial<TrainingProgress>) {
      Object.assign(this.progress, progress)
    },

    /** Reset store to idle state. */
    reset() {
      this.status = 'idle'
      this.jobId = null
      this.progress = { epoch: 0, totalEpochs: 100, progressPct: 0, loss: 0, eta: null, metrics: {} }
      this.error = null
    },

    /**
     * Start a training job.
     *
     * @param config - Training configuration
     */
    async startTraining(config: {
      datasetName: string
      baseModel: string
      epochs?: number
      batchSize?: number
      imageSize?: number
      patience?: number
      freezeLayers?: number
      lr0?: number
      lrf?: number
      modelName?: string
      augmentation?: AugmentationConfig
    }) {
      const { post } = useApi()

      // Build augmentation payload — only include non-undefined fields
      let augmentation: AugmentationConfig | undefined
      if (config.augmentation) {
        const filtered: Record<string, unknown> = {}
        for (const [k, v] of Object.entries(config.augmentation)) {
          if (v !== undefined && v !== null && v !== '') filtered[k] = v
        }
        if (Object.keys(filtered).length > 0) {
          augmentation = filtered as AugmentationConfig
        }
      }

      const res = await post<TrainingStatusResponse>('/training/start', {
        dataset_name: config.datasetName,
        base_model: config.baseModel,
        epochs: config.epochs,
        batch_size: config.batchSize,
        image_size: config.imageSize,
        patience: config.patience,
        freeze_layers: config.freezeLayers,
        lr0: config.lr0,
        lrf: config.lrf,
        model_name: config.modelName,
        augmentation: augmentation,
      })
      this._applyStatus(res)
    },

    /** Stop the current training job. */
    async stopTraining() {
      const { post } = useApi()
      await post('/training/stop')
      this.status = 'idle'
    },

    /** Fetch current training status from the backend. */
    async fetchStatus() {
      const { get } = useApi()
      try {
        const res = await get<TrainingStatusResponse>('/training/status')
        this._applyStatus(res)
      } catch (err) {
        console.error('Failed to fetch training status:', err)
      }
    },

    /** Fetch training job history. */
    async fetchHistory() {
      const { get } = useApi()
      try {
        const res = await get<TrainingHistoryResponse>('/training/history')
        this.history = res.jobs
      } catch (err) {
        console.error('Failed to fetch training history:', err)
      }
    },

    /** Fetch all trained models. */
    async fetchModels() {
      const { get } = useApi()
      try {
        const res = await get<ModelListResponse>('/models')
        this.models = res.models.map(this._mapModel)
      } catch (err) {
        console.error('Failed to fetch models:', err)
      }
    },

    /**
     * Delete a trained model.
     *
     * @param modelName - Model name to delete
     */
    async deleteModel(modelName: string) {
      const { del } = useApi()
      await del<MessageResponse>(`/models/${modelName}`)
      await this.fetchModels()
    },

    /**
     * Set a model as the active inference model.
     *
     * @param modelName - Model name to activate
     */
    async activateModel(modelName: string) {
      const { put } = useApi()
      await put<MessageResponse>(`/models/${modelName}/activate`, {})
      await this.fetchModels()
    },

    /**
     * Export a trained model as a zip download.
     *
     * Opens a browser download via direct navigation — no fetch needed
     * since the backend returns a FileResponse.
     *
     * @param modelName - Model name to export
     */
    exportModel(modelName: string) {
      window.location.href = `/api/models/${encodeURIComponent(modelName)}/export`
    },

    /**
     * Import a trained model from a zip file upload.
     *
     * @param file - Zip file containing model weights and metadata
     * @param name - Optional override name for the imported model
     */
    async importModel(file: File, name?: string) {
      const { upload } = useApi()
      const path = name
        ? `/models/import?name=${encodeURIComponent(name)}`
        : '/models/import'
      await upload<ImportModelResponse>(path, file)
      await this.fetchModels()
    },

    /**
     * Download a pretrained YOLO model and register it.
     *
     * @param modelId - Ultralytics model identifier (e.g. 'yolo11n.pt')
     * @param name - Optional display name (defaults to modelId minus .pt)
     */
    async downloadPretrained(modelId: string, name?: string) {
      const { post } = useApi()
      const body: DownloadPretrainedRequest = { model_id: modelId }
      if (name) body.name = name
      await post<ModelResponse>('/models/pretrained', body)
      await this.fetchModels()
    },

    /** Apply a TrainingStatusResponse to the store state. */
    _applyStatus(res: TrainingStatusResponse) {
      this.jobId = res.job_id
      this.status = res.status as TrainingState['status']
      this.progress = {
        epoch: res.current_epoch,
        totalEpochs: res.total_epochs,
        progressPct: res.progress_pct,
        loss: res.metrics?.loss || 0,
        eta: null,
        metrics: res.metrics || {},
      }
      this.error = res.error || null
    },

    /** Map API model response to domain ModelInfo. */
    _mapModel(m: ModelResponse): ModelInfo {
      return {
        name: m.name,
        baseModel: m.base_model,
        datasetName: m.dataset_name,
        createdAt: m.created_at,
        epochsCompleted: m.epochs_completed,
        bestMap50: m.best_map50,
        isActive: m.is_active,
        path: m.path,
      }
    },
  },
})
