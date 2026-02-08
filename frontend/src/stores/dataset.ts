/**
 * Dataset state store.
 *
 * Manages the currently loaded dataset, image list, annotations,
 * and filtering/review state for the Dataset view.
 */

import { defineStore } from 'pinia'
import type { ImageInfo, Annotation } from '@/types/models'

interface DatasetFilter {
  split: string
  annotated: string
  search: string
}

interface DatasetState {
  currentDataset: string | null
  images: ImageInfo[]
  annotations: Annotation[]
  filter: DatasetFilter
}

export const useDatasetStore = defineStore('dataset', {
  state: (): DatasetState => ({
    currentDataset: null,
    images: [],
    annotations: [],
    filter: {
      split: 'all',
      annotated: 'all',
      search: '',
    },
  }),

  getters: {
    /** Filter images based on current filter state. */
    filteredImages: (state) => {
      return state.images.filter((img) => {
        if (state.filter.split !== 'all' && img.split !== state.filter.split) return false
        if (state.filter.search && !img.filename.includes(state.filter.search)) return false
        return true
      })
    },
  },

  actions: {
    /** Reset store when switching datasets. */
    reset() {
      this.currentDataset = null
      this.images = []
      this.annotations = []
      this.filter = { split: 'all', annotated: 'all', search: '' }
    },
  },
})
