# Inference Subsystem (Produces Output Feeds)

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Load and run trained models or YOLO-World for detection. Subscribes to raw feeds from the [Feeds Subsystem](feeds.md), runs detection, and **produces inference output feeds** that are registered back with Feeds. These output feeds can be consumed by WebSocket (live preview), CaptureController (frame capture), logging, etc.

## Components

```
Inference Subsystem
├── Inference Manager
│   ├── Feed Subscription (subscribes to raw feeds)
│   ├── Output Feed Producer (registers with Feeds Subsystem)
│   └── Session Lifecycle (start/stop inference sessions)
├── Model Loader
│   ├── YOLO-World Loader
│   ├── Fine-tuned Model Loader
│   └── Model Cache (keep loaded models in memory)
├── Detection Runner
│   ├── Frame Preprocessor
│   ├── Inference Executor
│   └── Result Postprocessor (NMS, filtering)
└── Model Switcher
    ├── Hot-swap Handler
    └── Memory Manager (unload old model)
```

## Key Concept: Inference Output Feeds

When inference starts on a raw feed, the Inference Subsystem:
1. Subscribes to the raw feed (e.g., camera)
2. Runs detection on each frame
3. Creates an **InferenceFrame** containing both the image and detections
4. Registers this as a derived feed with the Feeds Subsystem
5. Any subscriber (WebSocket, CaptureController) can subscribe to this derived feed

## Data Flow

```
┌─────────────────┐    subscribe    ┌─────────────────┐
│   Raw Feed      │───────────────▶│   Inference     │
│   (camera)      │                │   Subsystem     │
└─────────────────┘                └─────────────────┘
                                          │
                                          │ run detection
                                          ▼
                                   ┌─────────────────┐
                                   │ InferenceFrame  │
                                   │ (image + dets)  │
                                   └─────────────────┘
                                          │
                                          │ register as derived feed
                                          ▼
                                   ┌─────────────────┐
                                   │ Feeds Subsystem │
                                   │ (derived feed)  │
                                   └─────────────────┘
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                   ┌──────────┐    ┌───────────┐    ┌──────────┐
                   │WebSocket │    │ Capture   │    │ Logging  │
                   │(preview) │    │Controller │    │ [Ideal]  │
                   └──────────┘    └───────────┘    └──────────┘
```

## Inference Manager

```python
class InferenceManager:
    """
    Manages inference sessions. Subscribes to raw feeds,
    runs detection, and produces output feeds.
    """

    def __init__(self, feed_manager: FeedManager):
        self._feed_manager = feed_manager
        self._sessions: dict[str, InferenceSession] = {}

    async def start_inference(
        self,
        source_feed_id: str,
        model_id: str,
        prompts: dict[int, list[str]] | None = None
    ) -> str:
        """
        Start inference on a feed. Returns the output feed ID.
        """
        # Generate output feed ID
        output_feed_id = f"inference_{source_feed_id}_{uuid.uuid4().hex[:8]}"

        # Load model
        model = self._load_model(model_id, prompts)

        # Register derived feed with Feeds Subsystem
        self._feed_manager.register_derived_feed(
            feed_id=output_feed_id,
            source_feed_id=source_feed_id,
            name=f"Inference on {source_feed_id}"
        )

        # Create session and subscribe to raw feed
        session = InferenceSession(
            output_feed_id=output_feed_id,
            source_feed_id=source_feed_id,
            model=model,
            feed_manager=self._feed_manager
        )

        self._feed_manager.subscribe(
            source_feed_id,
            session.on_frame
        )

        self._sessions[output_feed_id] = session
        return output_feed_id

    async def stop_inference(self, output_feed_id: str) -> None:
        """Stop an inference session"""
        session = self._sessions.get(output_feed_id)
        if session:
            self._feed_manager.unsubscribe(
                session.source_feed_id,
                session.on_frame
            )
            self._feed_manager.unregister_derived_feed(output_feed_id)
            del self._sessions[output_feed_id]


class InferenceSession:
    """A single inference session processing frames from a source feed"""

    def __init__(
        self,
        output_feed_id: str,
        source_feed_id: str,
        model: LoadedModel,
        feed_manager: FeedManager
    ):
        self.output_feed_id = output_feed_id
        self.source_feed_id = source_feed_id
        self._model = model
        self._feed_manager = feed_manager
        self._sequence = 0

    def on_frame(self, frame: Frame) -> None:
        """Process a raw frame and push inference result to output feed"""
        # Run detection
        start_time = time.time()
        results = self._model.model(frame.data)
        inference_time = (time.time() - start_time) * 1000

        # Convert to Detection objects
        detections = self._parse_results(results)

        # Create inference frame
        self._sequence += 1
        inference_frame = InferenceFrame(
            data=frame.data,
            timestamp=frame.timestamp,
            sequence=self._sequence,
            feed_id=self.output_feed_id,
            source_feed_id=self.source_feed_id,
            width=frame.width,
            height=frame.height,
            detections=detections,
            inference_time_ms=inference_time
        )

        # Push to derived feed (notifies all subscribers)
        self._feed_manager.push_derived_frame(
            self.output_feed_id,
            inference_frame
        )
```

## Model Types

```python
class ModelType(Enum):
    YOLO_WORLD = "yolo_world"      # Zero-shot with text prompts
    FINE_TUNED = "fine_tuned"      # Custom trained model

@dataclass
class LoadedModel:
    model_type: ModelType
    model: YOLO
    classes: dict          # class_id → name
    prompts: dict = None   # For YOLO-World only
```
