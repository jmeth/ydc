# Training Subsystem

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Manage model training jobs, monitor resources, and handle background execution. **Delegates model storage to the [Persistence Layer](persistence.md).**

## Components

```
Training Subsystem
├── Training Runner
│   ├── Job Queue (single job for MVP)
│   ├── YOLO Training Wrapper (ultralytics)
│   ├── Progress Monitor
│   └── Output Parser (loss, metrics)
├── Model Manager
│   ├── Training orchestration
│   ├── Model registration (uses ModelStore)
│   └── Active model tracking
└── Resource Monitor
    ├── GPU Memory Monitor
    ├── CPU/RAM Monitor
    └── Threshold Alerter
```

## Dependency on Persistence Layer

```python
# training/manager.py
from persistence import ModelStore, DatasetStore

class TrainingManager:
    """
    Manages training jobs and model lifecycle.
    Delegates storage to Persistence Layer.
    """

    def __init__(
        self,
        model_store: ModelStore,
        dataset_store: DatasetStore
    ):
        self._models = model_store
        self._datasets = dataset_store
        self._current_job: TrainingJob | None = None

    async def start_training(self, config: TrainingConfig) -> str:
        """Start a training job"""
        # Validate dataset exists via Persistence Layer
        dataset = await self._datasets.get(config.dataset_name)
        if not dataset:
            raise ValueError(f"Dataset not found: {config.dataset_name}")

        # Create training job
        job_id = str(uuid.uuid4())
        self._current_job = TrainingJob(job_id, config)

        # Start training in background
        asyncio.create_task(self._run_training(self._current_job))

        return job_id

    async def _run_training(self, job: TrainingJob) -> None:
        """Execute training and save model via Persistence Layer"""
        try:
            # Run YOLO training
            model = YOLO(job.config.base_model)
            results = model.train(
                data=str(await self._datasets.get_data_yaml_path(job.config.dataset_name)),
                epochs=job.config.epochs,
                imgsz=job.config.image_size,
                # ... other params
            )

            # Save model via Persistence Layer
            await self._models.save(
                name=job.config.model_name,
                weights_path=results.save_dir / "weights" / "best.pt",
                config=job.config,
                metrics=self._extract_metrics(results)
            )

            job.status = "completed"

        except Exception as e:
            job.status = "error"
            job.error = str(e)

    async def list_models(self) -> list[ModelInfo]:
        """List all trained models via Persistence Layer"""
        return await self._models.list()

    async def delete_model(self, name: str) -> bool:
        """Delete a model via Persistence Layer"""
        return await self._models.delete(name)

    async def set_active_model(self, name: str) -> None:
        """Set the active model via Persistence Layer"""
        await self._models.set_active(name)
```

## Training Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Config    │───▶│  Validate   │───▶│   Train     │───▶│   Save      │
│   Input     │    │  Dataset    │    │  (async)    │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
                                     ┌─────────────┐
                                     │  Progress   │
                                     │  Events     │
                                     └─────────────┘
```

## State Machine

```
┌─────────┐   start()   ┌───────────┐  complete()  ┌───────────┐
│  IDLE   │────────────▶│ TRAINING  │─────────────▶│ COMPLETED │
└─────────┘             └───────────┘              └───────────┘
     ▲                       │                          │
     │        cancel()       │                          │
     │◀──────────────────────┘                          │
     │                                                  │
     └──────────────────────────────────────────────────┘
                        (reset)
```

## Training Configuration

```python
@dataclass
class TrainingConfig:
    # Dataset
    dataset_name: str

    # Model
    base_model: str = "yolo11n.pt"  # or yolo11s, yolo11m

    # Training params
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    patience: int = 20
    freeze_layers: int = 10  # freeze backbone for small datasets

    # Learning rate
    lr0: float = 0.001
    lrf: float = 0.01

    # Output
    model_name: str = ""  # user-defined name

    # Resource limits
    max_gpu_memory_percent: float = 0.8
```

## Model Registry Structure

```
models/
└── my_dataset/
    ├── registry.json           # Model index
    └── models/
        ├── detector_v1/
        │   ├── weights/
        │   │   └── best.pt
        │   ├── config.yaml     # Training config used
        │   └── metrics.json    # Final metrics
        └── detector_v2/
            └── ...
```

**registry.json**
```json
{
  "active": "detector_v1",
  "models": {
    "detector_v1": {
      "created": "2024-01-15T14:00:00Z",
      "base_model": "yolo11n.pt",
      "epochs_completed": 87,
      "best_map50": 0.82,
      "dataset_snapshot": "200 images, 3 classes"
    }
  }
}
```

## Resource Monitor

```python
class ResourceMonitor:
    def __init__(self, gpu_threshold=0.9, ram_threshold=0.85):
        self.gpu_threshold = gpu_threshold
        self.ram_threshold = ram_threshold
        self.callbacks = []

    def check(self) -> ResourceStatus:
        gpu_usage = get_gpu_memory_usage()
        ram_usage = get_ram_usage()

        status = ResourceStatus(
            gpu_percent=gpu_usage,
            ram_percent=ram_usage,
            constrained=(gpu_usage > self.gpu_threshold or
                        ram_usage > self.ram_threshold)
        )

        if status.constrained:
            self.notify_callbacks(status)

        return status
```
