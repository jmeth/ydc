/**
 * Dataset state store.
 *
 * Manages the currently loaded dataset, image list, annotations,
 * and filtering state for the Dataset view. Provides actions for
 * all dataset CRUD, image management, and label operations via the API.
 */

import { defineStore } from 'pinia'
import { useApi } from '@/composables/useApi'
import type { ImageInfo, Annotation } from '@/types/models'
import type {
  DatasetResponse,
  DatasetListResponse,
  ImageListResponse,
  LabelsResponse,
  AnnotationResponse,
} from '@/types/api'

interface DatasetFilter {
  split: string
  annotated: string
  search: string
}

interface DatasetState {
  currentDataset: string | null
  currentDatasetInfo: DatasetResponse | null
  datasets: DatasetResponse[]
  images: ImageInfo[]
  annotations: Annotation[]
  selectedImage: ImageInfo | null
  filter: DatasetFilter
  loading: boolean
}

export const useDatasetStore = defineStore('dataset', {
  state: (): DatasetState => ({
    currentDataset: null,
    currentDatasetInfo: null,
    datasets: [],
    images: [],
    annotations: [],
    selectedImage: null,
    filter: {
      split: 'all',
      annotated: 'all',
      search: '',
    },
    loading: false,
  }),

  getters: {
    /** Filter images based on current filter state. */
    filteredImages: (state) => {
      return state.images.filter((img) => {
        if (state.filter.split !== 'all' && img.split !== state.filter.split) return false
        if (state.filter.annotated === 'yes' && !img.hasLabels) return false
        if (state.filter.annotated === 'no' && img.hasLabels) return false
        if (state.filter.search && !img.filename.toLowerCase().includes(state.filter.search.toLowerCase())) return false
        return true
      })
    },
  },

  actions: {
    /** Reset store when switching datasets. */
    reset() {
      this.currentDataset = null
      this.currentDatasetInfo = null
      this.images = []
      this.annotations = []
      this.selectedImage = null
      this.filter = { split: 'all', annotated: 'all', search: '' }
    },

    /** Fetch all datasets from the backend. */
    async fetchDatasets() {
      const { get } = useApi()
      const res = await get<DatasetListResponse>('/datasets')
      this.datasets = res.datasets
    },

    /**
     * Load a dataset's info and image list.
     *
     * @param name - Dataset name to load
     */
    async loadDataset(name: string) {
      const { get } = useApi()
      this.loading = true
      try {
        this.currentDataset = name
        const info = await get<DatasetResponse>(`/datasets/${name}`)
        this.currentDatasetInfo = info
        await this.loadImages()
      } finally {
        this.loading = false
      }
    },

    /** Load images for the current dataset, with optional split filter. */
    async loadImages() {
      if (!this.currentDataset) return
      const { get } = useApi()
      const splitParam = this.filter.split !== 'all' ? `?split=${this.filter.split}` : ''
      const res = await get<ImageListResponse>(`/datasets/${this.currentDataset}/images${splitParam}`)
      this.images = res.images.map((img) => ({
        filename: img.filename,
        split: img.split,
        width: img.width,
        height: img.height,
        sizeBytes: img.size_bytes,
        hasLabels: img.has_labels,
      }))
    },

    /**
     * Load annotation labels for a specific image.
     *
     * @param split - Image split
     * @param filename - Image filename
     * @returns List of annotations
     */
    async loadLabels(split: string, filename: string): Promise<Annotation[]> {
      if (!this.currentDataset) return []
      const { get } = useApi()
      const stem = filename.replace(/\.[^.]+$/, '')
      const res = await get<LabelsResponse>(`/datasets/${this.currentDataset}/labels/${split}/${stem}`)
      this.annotations = res.annotations.map(this._mapAnnotation)
      return this.annotations
    },

    /**
     * Save annotation labels for a specific image.
     *
     * @param split - Image split
     * @param filename - Image filename
     * @param annotations - YOLO annotations to save
     */
    async saveLabels(split: string, filename: string, annotations: Annotation[]) {
      if (!this.currentDataset) return
      const { put } = useApi()
      const stem = filename.replace(/\.[^.]+$/, '')
      await put(`/datasets/${this.currentDataset}/labels/${split}/${stem}`, {
        annotations: annotations.map((a) => ({
          class_id: a.classId,
          x: a.x,
          y: a.y,
          width: a.width,
          height: a.height,
        })),
      })
      this.annotations = annotations
    },

    /**
     * Create a new dataset.
     *
     * @param name - Dataset name
     * @param classes - Class name list
     */
    async createDataset(name: string, classes: string[]): Promise<DatasetResponse> {
      const { post } = useApi()
      const res = await post<DatasetResponse>('/datasets', { name, classes })
      await this.fetchDatasets()
      return res
    },

    /** Delete a dataset by name. */
    async deleteDataset(name: string) {
      const { del } = useApi()
      await del(`/datasets/${name}`)
      if (this.currentDataset === name) this.reset()
      await this.fetchDatasets()
    },

    /**
     * Delete an image from the current dataset.
     *
     * @param split - Image split
     * @param filename - Image filename
     */
    async deleteImage(split: string, filename: string) {
      if (!this.currentDataset) return
      const { del } = useApi()
      await del(`/datasets/${this.currentDataset}/images/${split}/${filename}`)
      this.images = this.images.filter((i) => !(i.filename === filename && i.split === split))
    },

    /**
     * Move an image to a different split.
     *
     * @param filename - Image filename
     * @param fromSplit - Current split
     * @param toSplit - Destination split
     */
    async changeSplit(filename: string, fromSplit: string, toSplit: string) {
      if (!this.currentDataset) return
      const { put } = useApi()
      await put(`/datasets/${this.currentDataset}/split/${fromSplit}/${filename}`, {
        to_split: toSplit,
      })
      const img = this.images.find((i) => i.filename === filename && i.split === fromSplit)
      if (img) img.split = toSplit
    },

    /**
     * Upload an image to the current dataset.
     *
     * @param file - File to upload
     * @param split - Target split
     */
    async uploadImage(file: File, split = 'train') {
      if (!this.currentDataset) return
      const { upload } = useApi()
      await upload(`/datasets/${this.currentDataset}/images?split=${split}`, file)
      await this.loadImages()
    },

    /**
     * Export dataset as a zip file download.
     * Triggers browser download via the export endpoint.
     */
    async exportDataset() {
      if (!this.currentDataset) return
      const link = document.createElement('a')
      link.href = `/api/datasets/${this.currentDataset}/export`
      link.download = `${this.currentDataset}.zip`
      link.click()
    },

    /** Map API annotation response to domain Annotation. */
    _mapAnnotation(a: AnnotationResponse): Annotation {
      return {
        classId: a.class_id,
        x: a.x,
        y: a.y,
        width: a.width,
        height: a.height,
      }
    },
  },
})
