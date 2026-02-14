/**
 * Global application state store.
 *
 * Tracks the currently selected dataset, WebSocket connection status,
 * active mode, and shared lists of feeds and datasets fetched from
 * the API for use across multiple views.
 */

import { defineStore } from 'pinia'
import { useApi } from '@/composables/useApi'
import type { FeedInfo } from '@/types/models'
import type { FeedListResponse, DatasetListResponse, DatasetResponse } from '@/types/api'

export const useAppStore = defineStore('app', {
  state: () => ({
    /** Currently selected dataset name, null if none */
    currentDataset: null as string | null,
    /** Whether the WebSocket event connection is live */
    isConnected: false,
    /** Active application mode matching route names */
    activeMode: 'scan' as 'scan' | 'dataset' | 'train' | 'model',
    /** All available feeds from the backend */
    feeds: [] as FeedInfo[],
    /** All available datasets from the backend */
    datasets: [] as DatasetResponse[],
  }),

  actions: {
    setDataset(name: string | null) {
      this.currentDataset = name
    },
    setConnected(connected: boolean) {
      this.isConnected = connected
    },
    setMode(mode: 'scan' | 'dataset' | 'train' | 'model') {
      this.activeMode = mode
    },

    /**
     * Fetch all feeds from the backend and update state.
     * Maps snake_case API response to camelCase FeedInfo.
     */
    async fetchFeeds() {
      const { get } = useApi()
      try {
        const res = await get<FeedListResponse>('/feeds')
        this.feeds = res.feeds.map((f) => ({
          id: f.id,
          name: f.name,
          type: f.type as FeedInfo['type'],
          status: f.status as FeedInfo['status'],
          sourceFeedId: f.source_feed_id,
        }))
      } catch (err) {
        console.error('Failed to fetch feeds:', err)
      }
    },

    /**
     * Fetch all datasets from the backend and update state.
     */
    async fetchDatasets() {
      const { get } = useApi()
      try {
        const res = await get<DatasetListResponse>('/datasets')
        this.datasets = res.datasets
      } catch (err) {
        console.error('Failed to fetch datasets:', err)
      }
    },
  },
})
