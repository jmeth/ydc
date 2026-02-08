<script setup lang="ts">
/**
 * Generic modal dialog component.
 *
 * Renders a centered overlay with a content slot. Closes on
 * backdrop click or Escape key. Use v-if to show/hide.
 *
 * @prop title - Modal header text
 * @emits close - Fired when the modal should be dismissed
 */

defineProps<{
  title: string
}>()

const emit = defineEmits<{
  close: []
}>()

function onBackdropClick(event: MouseEvent) {
  if ((event.target as HTMLElement).classList.contains('modal-backdrop')) {
    emit('close')
  }
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    emit('close')
  }
}
</script>

<template>
  <Teleport to="body">
    <div class="modal-backdrop" @click="onBackdropClick" @keydown="onKeydown" tabindex="-1">
      <div class="modal-content" role="dialog" :aria-label="title">
        <div class="modal-header">
          <h2>{{ title }}</h2>
          <button class="modal-close" @click="emit('close')" aria-label="Close">&times;</button>
        </div>
        <div class="modal-body">
          <slot />
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--color-bg-surface);
  border-radius: 8px;
  min-width: 400px;
  max-width: 90vw;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--color-border);
}

.modal-header h2 {
  margin: 0;
  font-size: 1.125rem;
}

.modal-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--color-text-muted);
  line-height: 1;
}

.modal-body {
  padding: 1.25rem;
}
</style>
