<script setup lang="ts">
/**
 * Table of past training jobs.
 *
 * Displays model name, dataset, base model, status badge, epochs
 * completed, best mAP@50, and date for each historical training run.
 */
import { computed } from 'vue'
import { useTrainingStore } from '@/stores/training'

const trainingStore = useTrainingStore()

/** Format a unix timestamp to a readable date string. */
function formatDate(ts: number | null): string {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** Map status to badge class. */
function statusBadge(status: string): string {
  switch (status) {
    case 'completed': return 'badge-success'
    case 'error': return 'badge-error'
    case 'cancelled': return 'badge-warning'
    default: return 'badge-info'
  }
}

const history = computed(() => trainingStore.history)
</script>

<template>
  <div class="card">
    <div class="card-header">Training History</div>

    <div v-if="history.length === 0" class="empty-state">
      <div class="empty-state-text">No training history</div>
    </div>

    <table v-else class="history-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Dataset</th>
          <th>Base</th>
          <th>Status</th>
          <th>Epochs</th>
          <th>mAP@50</th>
          <th>Date</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="job in history" :key="job.job_id">
          <td class="model-name">{{ job.model_name }}</td>
          <td>{{ job.dataset_name }}</td>
          <td>{{ job.base_model }}</td>
          <td><span class="badge" :class="statusBadge(job.status)">{{ job.status }}</span></td>
          <td class="num">{{ job.epochs_completed }}</td>
          <td class="num">{{ job.best_map50 !== null ? (job.best_map50 * 100).toFixed(1) + '%' : '-' }}</td>
          <td>{{ formatDate(job.started_at) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8125rem;
}

.history-table th {
  text-align: left;
  padding: 0.5rem 0.5rem;
  border-bottom: 1px solid var(--color-border);
  font-size: 0.6875rem;
  font-weight: 600;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.history-table td {
  padding: 0.375rem 0.5rem;
  border-bottom: 1px solid var(--color-border);
}

.model-name {
  font-weight: 600;
}

.num {
  font-family: var(--font-mono);
  text-align: right;
}
</style>
