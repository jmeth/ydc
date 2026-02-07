# Frontend Application (Vue.js)

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Browser-based UI for all modes of operation, built with Vue.js 3 and Composition API.

## Technology

- **Vue.js 3** - Reactive component framework
- **Vite** - Build tool with fast HMR
- **Vue Router** - Client-side routing for modes
- **Pinia** - State management (Vue 3 native)
- **TypeScript** - Type safety throughout

## Component Structure

```
frontend/
├── src/
│   ├── main.ts                 # App entry point
│   ├── App.vue                 # Root component
│   ├── router/
│   │   └── index.ts            # Route definitions
│   ├── stores/                 # Pinia stores
│   │   ├── app.ts              # Global app state
│   │   ├── inference.ts        # Inference state (produces output feeds)
│   │   ├── capture.ts          # Capture state (subscribes to inference feeds)
│   │   ├── dataset.ts          # Dataset mode state
│   │   ├── training.ts         # Training mode state
│   │   └── notifications.ts    # Notification state
│   ├── composables/            # Reusable composition functions
│   │   ├── useApi.ts           # API client
│   │   ├── useWebSocket.ts     # WebSocket connection
│   │   └── useVideoStream.ts   # Video frame handling
│   ├── views/                  # Top-level route views
│   │   ├── ScanView.vue
│   │   ├── DatasetView.vue
│   │   ├── TrainView.vue
│   │   └── ModelView.vue
│   ├── components/
│   │   ├── common/             # Shared components
│   │   │   ├── AppHeader.vue
│   │   │   ├── AppNav.vue
│   │   │   ├── StatusBar.vue
│   │   │   ├── Modal.vue
│   │   │   └── Toast.vue
│   │   ├── inference/          # Inference components (shared by Scan/Model modes)
│   │   │   ├── VideoPlayer.vue       # Subscribes to inference output feed
│   │   │   ├── DetectionOverlay.vue  # Renders detection boxes
│   │   │   ├── PromptEditor.vue      # YOLO-World prompt configuration
│   │   │   └── InferenceStats.vue    # FPS, inference time
│   │   ├── capture/            # Capture components (Scan mode specific)
│   │   │   ├── CaptureControls.vue   # Start/stop capture, manual trigger
│   │   │   └── CaptureStats.vue      # Positive/negative counts
│   │   ├── dataset/            # Dataset mode components
│   │   │   ├── ImageGrid.vue
│   │   │   ├── ImageEditor.vue
│   │   │   ├── AnnotationCanvas.vue
│   │   │   ├── AnnotationList.vue
│   │   │   ├── BulkActions.vue
│   │   │   └── ReviewQueue.vue
│   │   ├── training/           # Training mode components
│   │   │   ├── TrainingConfig.vue
│   │   │   ├── ProgressDisplay.vue
│   │   │   └── ModelList.vue
│   │   └── inference/          # Model mode components
│   │       ├── InferencePlayer.vue
│   │       ├── ModelSelector.vue
│   │       └── DetectionStats.vue
│   ├── types/                  # TypeScript type definitions
│   │   ├── api.ts
│   │   ├── models.ts
│   │   └── websocket.ts
│   └── assets/
│       └── styles/
│           ├── main.css
│           └── variables.css
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

## Navigation

```
┌─────────────────────────────────────────────────────────────┐
│  [Scan] [Dataset] [Train] [Model]    Dataset: ▼    [⚙️]    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                     Mode-specific UI                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Status: Scanning... | Captures: 45 | Training: 23/100     │
└─────────────────────────────────────────────────────────────┘
```

## State Management (Pinia Stores)

```typescript
// stores/inference.ts
import { defineStore } from 'pinia'

interface InferenceState {
  status: 'idle' | 'running' | 'error'
  sourceFeedId: string | null      // Raw feed being processed
  outputFeedId: string | null      // Inference output feed ID
  modelId: string | null
  prompts: Record<number, string[]>
  stats: {
    fps: number
    detections: number
    inferenceTimeMs: number
  }
}

export const useInferenceStore = defineStore('inference', {
  state: (): InferenceState => ({
    status: 'idle',
    sourceFeedId: null,
    outputFeedId: null,
    modelId: null,
    prompts: {},
    stats: { fps: 0, detections: 0, inferenceTimeMs: 0 }
  }),

  actions: {
    async startInference(sourceFeedId: string, modelId: string, prompts?: Record<number, string[]>) { ... },
    async stopInference() { ... },
    async updatePrompts(prompts: Record<number, string[]>) { ... },
    updateStats(stats: Partial<InferenceState['stats']>) { ... }
  }
})

// stores/capture.ts
interface CaptureState {
  status: 'idle' | 'running' | 'paused'
  inferenceFeedId: string | null   // Inference feed we're subscribed to
  datasetName: string | null
  config: {
    captureInterval: number
    negativeRatio: number
    confidenceThreshold: number
  }
  stats: {
    totalCaptures: number
    positiveCaptures: number
    negativeCaptures: number
  }
}

export const useCaptureStore = defineStore('capture', {
  state: (): CaptureState => ({
    status: 'idle',
    inferenceFeedId: null,
    datasetName: null,
    config: {
      captureInterval: 2.0,
      negativeRatio: 0.2,
      confidenceThreshold: 0.3
    },
    stats: {
      totalCaptures: 0,
      positiveCaptures: 0,
      negativeCaptures: 0
    }
  }),

  actions: {
    async startCapture(inferenceFeedId: string, datasetName: string, config?: Partial<CaptureState['config']>) { ... },
    async stopCapture() { ... },
    async triggerManualCapture() { ... },
    updateStats(stats: Partial<CaptureState['stats']>) { ... }
  }
})

// stores/dataset.ts
export const useDatasetStore = defineStore('dataset', {
  state: () => ({
    currentDataset: null as string | null,
    images: [] as ImageInfo[],
    reviewQueue: [] as ReviewItem[],
    currentImage: null as ImageInfo | null,
    annotations: [] as Annotation[],
    filter: {
      split: 'all',
      annotated: 'all',
      search: ''
    }
  }),

  getters: {
    filteredImages: (state) => { ... },
    pendingReviewCount: (state) => state.reviewQueue.length
  },

  actions: {
    async loadDataset(name: string) { ... },
    async saveAnnotations() { ... },
    async bulkAction(imageIds: string[], action: string) { ... }
  }
})

// stores/training.ts
export const useTrainingStore = defineStore('training', {
  state: () => ({
    status: 'idle' as 'idle' | 'training' | 'completed' | 'error',
    progress: {
      epoch: 0,
      totalEpochs: 100,
      loss: 0,
      eta: null as string | null
    },
    config: null as TrainingConfig | null,
    error: null as string | null
  }),

  actions: {
    async startTraining(config: TrainingConfig) { ... },
    async stopTraining() { ... },
    updateProgress(progress: Partial<TrainingProgress>) { ... }
  }
})

// stores/notifications.ts
export const useNotificationStore = defineStore('notifications', {
  state: () => ({
    toasts: [] as Toast[],
    banners: [] as Banner[],
    history: [] as Notification[]
  }),

  actions: {
    showToast(toast: Omit<Toast, 'id'>) { ... },
    showBanner(banner: Omit<Banner, 'id'>) { ... },
    dismiss(id: string) { ... }
  }
})
```

## Composables

```typescript
// composables/useWebSocket.ts
import { ref, onMounted, onUnmounted } from 'vue'

export function useWebSocket(url: string) {
  const socket = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const lastMessage = ref<any>(null)

  const connect = () => {
    socket.value = new WebSocket(url)

    socket.value.onopen = () => {
      isConnected.value = true
    }

    socket.value.onmessage = (event) => {
      lastMessage.value = JSON.parse(event.data)
    }

    socket.value.onclose = () => {
      isConnected.value = false
      // Auto-reconnect after 2 seconds
      setTimeout(connect, 2000)
    }
  }

  const send = (data: any) => {
    if (socket.value?.readyState === WebSocket.OPEN) {
      socket.value.send(JSON.stringify(data))
    }
  }

  onMounted(connect)
  onUnmounted(() => socket.value?.close())

  return { isConnected, lastMessage, send }
}

// composables/useVideoStream.ts
export function useVideoStream(feedId: Ref<string | null>) {
  const frame = ref<string | null>(null)  // Base64 frame
  const fps = ref(0)

  // Subscribe to video frames via WebSocket
  const { lastMessage } = useWebSocket('/ws/video')

  watch(lastMessage, (msg) => {
    if (msg?.type === 'frame' && msg.feedId === feedId.value) {
      frame.value = msg.data
      fps.value = msg.fps
    }
  })

  return { frame, fps }
}

// composables/useApi.ts
export function useApi() {
  const baseUrl = '/api'

  const get = async <T>(path: string): Promise<T> => {
    const res = await fetch(`${baseUrl}${path}`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  const post = async <T>(path: string, data?: any): Promise<T> => {
    const res = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: data ? JSON.stringify(data) : undefined
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  // ... put, delete, upload methods

  return { get, post, put, delete: del, upload }
}
```

## Example Component

```vue
<!-- components/inference/VideoPlayer.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import { useInferenceStore } from '@/stores/inference'
import { useVideoStream } from '@/composables/useVideoStream'
import DetectionOverlay from './DetectionOverlay.vue'

const inferenceStore = useInferenceStore()
// Subscribe to the inference output feed (which includes detections)
const { frame, fps, detections } = useVideoStream(
  computed(() => inferenceStore.outputFeedId)
)
</script>

<template>
  <div class="video-player">
    <div class="video-container">
      <img
        v-if="frame"
        :src="`data:image/jpeg;base64,${frame}`"
        alt="Video feed"
      />
      <div v-else class="no-feed">
        No video feed
      </div>
      <DetectionOverlay :detections="detections" />
    </div>
    <div class="video-stats">
      <span>{{ fps.toFixed(1) }} FPS</span>
      <span>{{ detections.length }} detections</span>
      <span>{{ inferenceStore.stats.inferenceTimeMs.toFixed(0) }}ms</span>
    </div>
  </div>
</template>

<style scoped>
.video-player {
  position: relative;
  background: #1a1a1a;
}

.video-container {
  position: relative;
  width: 100%;
  aspect-ratio: 16/9;
}

.video-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-stats {
  display: flex;
  gap: 1rem;
  padding: 0.5rem;
  font-size: 0.875rem;
  color: #888;
}
</style>
```
