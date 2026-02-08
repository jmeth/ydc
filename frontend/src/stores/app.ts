/**
 * Global application state store.
 *
 * Tracks the currently selected dataset, WebSocket connection status,
 * and which application mode is active.
 */

import { defineStore } from 'pinia'

export const useAppStore = defineStore('app', {
  state: () => ({
    /** Currently selected dataset name, null if none */
    currentDataset: null as string | null,
    /** Whether the WebSocket event connection is live */
    isConnected: false,
    /** Active application mode matching route names */
    activeMode: 'scan' as 'scan' | 'dataset' | 'train' | 'model',
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
  },
})
