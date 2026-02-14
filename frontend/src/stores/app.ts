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
import type {
  FeedListResponse,
  FeedInfoResponse,
  CreateFeedRequest,
  DatasetListResponse,
  DatasetResponse,
} from '@/types/api'

/**
 * Map a snake_case FeedInfoResponse from the API to a camelCase FeedInfo.
 *
 * @param f - API response feed object
 * @returns FeedInfo domain model
 */
function mapFeed(f: FeedInfoResponse): FeedInfo {
  return {
    id: f.feed_id,
    name: f.name,
    feedType: f.feed_type,
    source: f.source,
    status: f.status,
    fps: f.fps,
    resolution: f.resolution,
    frameCount: f.frame_count,
  }
}

export const useAppStore = defineStore('app', {
  state: () => ({
    /** Currently selected dataset name, null if none */
    currentDataset: null as string | null,
    /** Whether the WebSocket event connection is live */
    isConnected: false,
    /** Active application mode matching route names */
    activeMode: 'scan' as 'feeds' | 'scan' | 'dataset' | 'train' | 'model',
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
    setMode(mode: 'feeds' | 'scan' | 'dataset' | 'train' | 'model') {
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
        this.feeds = res.feeds.map(mapFeed)
      } catch (err) {
        console.error('Failed to fetch feeds:', err)
      }
    },

    /**
     * Create a new feed via the API and refresh the feed list.
     *
     * @param request - Feed creation params (feed_type, source, name, buffer_size)
     * @returns The created FeedInfo, or null on failure
     */
    async addFeed(request: CreateFeedRequest): Promise<FeedInfo | null> {
      const { post } = useApi()
      try {
        const res = await post<FeedInfoResponse>('/feeds', request)
        await this.fetchFeeds()
        return mapFeed(res)
      } catch (err) {
        console.error('Failed to add feed:', err)
        throw err
      }
    },

    /**
     * Delete a feed by ID and refresh the feed list.
     *
     * @param feedId - The feed to remove
     */
    async removeFeed(feedId: string): Promise<void> {
      const { del } = useApi()
      try {
        await del(`/feeds/${feedId}`)
        await this.fetchFeeds()
      } catch (err) {
        console.error('Failed to remove feed:', err)
        throw err
      }
    },

    /**
     * Pause a feed by ID and refresh the feed list.
     *
     * @param feedId - The feed to pause
     */
    async pauseFeed(feedId: string): Promise<void> {
      const { post } = useApi()
      try {
        await post(`/feeds/${feedId}/pause`)
        await this.fetchFeeds()
      } catch (err) {
        console.error('Failed to pause feed:', err)
        throw err
      }
    },

    /**
     * Resume a paused feed by ID and refresh the feed list.
     *
     * @param feedId - The feed to resume
     */
    async resumeFeed(feedId: string): Promise<void> {
      const { post } = useApi()
      try {
        await post(`/feeds/${feedId}/resume`)
        await this.fetchFeeds()
      } catch (err) {
        console.error('Failed to resume feed:', err)
        throw err
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
