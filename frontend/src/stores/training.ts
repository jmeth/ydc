/**
 * Training state store.
 *
 * Tracks training job status, progress updates from WebSocket,
 * configuration, and error state.
 */

import { defineStore } from 'pinia'
import type { TrainingConfig } from '@/types/models'

interface TrainingProgress {
  epoch: number
  totalEpochs: number
  loss: number
  eta: string | null
}

interface TrainingState {
  status: 'idle' | 'training' | 'completed' | 'error'
  progress: TrainingProgress
  config: TrainingConfig | null
  error: string | null
}

export const useTrainingStore = defineStore('training', {
  state: (): TrainingState => ({
    status: 'idle',
    progress: {
      epoch: 0,
      totalEpochs: 100,
      loss: 0,
      eta: null,
    },
    config: null,
    error: null,
  }),

  actions: {
    /** Update training progress from WebSocket event. */
    updateProgress(progress: Partial<TrainingProgress>) {
      Object.assign(this.progress, progress)
    },

    /** Reset store to idle state. */
    reset() {
      this.status = 'idle'
      this.progress = { epoch: 0, totalEpochs: 100, loss: 0, eta: null }
      this.error = null
    },
  },
})
