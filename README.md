# YOLO Dataset Creator

A simple, browser-based tool for creating object detection datasets in YOLO format. Upload images or capture from webcam, draw bounding boxes, assign labels, and export datasets ready for training with [Ultralytics YOLO](https://docs.ultralytics.com/).

> **Note:** This project was vibecoded with [Claude](https://claude.ai) and is intended for **local/personal use only**. It has not been audited or tested for security and should not be deployed on public networks or in production environments.

## Features

- **Image Import**: Upload images via file picker or drag-and-drop
- **Webcam Capture**: Capture images directly from your webcam
- **Bounding Box Editor**: Draw, move, and resize annotation boxes
- **Label Management**: Create and manage class labels with custom colors
- **Dataset Splits**: Organize images into train/val/test sets
- **Keyboard Shortcuts**: Fast annotation workflow with hotkeys
- **YOLO Format Export**: Saves directly in YOLO-compatible format with `data.yaml`

## Installation

### Option 1: Docker (Recommended)

```bash
# Using docker-compose
docker-compose up -d

# Or build and run manually
docker build -t yolo-dataset-creator .
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/datasets:/app/datasets \
  --name yolo-dataset-creator \
  yolo-dataset-creator
```

### Option 2: Python

Requirements: Python 3.8+

```bash
cd yolo_dataset_creator
pip install -r requirements.txt
python server.py
```

Open http://localhost:5000 in your browser.

## Usage

### 1. Create a Dataset
Click "New Dataset" and enter a name. This creates the YOLO directory structure:
```
datasets/my_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 2. Add Labels
Click "+ Add" in the Labels panel to create class labels (e.g., "person", "car", "dog").

### 3. Import Images
- **Upload**: Click "Upload" or drag images onto the canvas
- **Webcam**: Click "Webcam" to capture images directly

### 4. Annotate
1. Select an image from the list
2. Press `N` or click the box tool to enter draw mode
3. Click and drag to draw a bounding box
4. Select a label (click in Labels panel or press `1-9`)
5. Press `Ctrl+S` to save

### 5. Edit Annotations
- **Select**: Click on a box to select it
- **Move**: Drag a selected box to reposition
- **Resize**: Drag corner or edge handles
- **Change Label**: With box selected, click a label or press `1-9`
- **Delete**: Press `Delete` or click × in the Annotations panel

### 6. Organize Splits
Use the Split dropdown to assign images to train/val/test sets.

### 7. Export Dataset
Click "Export ZIP" to download the complete dataset as a zip file. The export modal shows:
- Dataset size and file count
- Image distribution across splits
- Warning for large datasets (>100MB)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-9` | Select label / Change selected box label |
| `N` | Draw mode |
| `V` | Select mode |
| `Delete` | Delete selected box |
| `Ctrl+S` | Save annotations |
| `←` `→` | Previous/Next image |
| `+` `-` | Zoom in/out |
| `Esc` | Cancel/Deselect |

## YOLO Format

Annotations are saved in YOLO format:
- One `.txt` file per image with matching filename
- Each line: `class_id x_center y_center width height`
- Coordinates are normalized (0-1) relative to image dimensions

Example label file:
```
0 0.525 0.376 0.284 0.418
1 0.735 0.298 0.193 0.337
```

The `data.yaml` file is auto-generated:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 2
names:
  0: person
  1: car
```

## Creating a Good Dataset

A high-quality dataset is essential for training accurate models. Key principles:

| Principle | Description |
|-----------|-------------|
| **Diversity** | Include varied lighting, angles, backgrounds, and object sizes |
| **Balance** | Aim for similar numbers of examples per class |
| **Accuracy** | Draw tight bounding boxes that fully contain objects |
| **Quantity** | More data generally improves performance (aim for 100+ images per class) |
| **Real-world match** | Training data should reflect your deployment environment |
| **Negative examples** | Include images without objects to reduce false positives |

### Tips
- Annotate objects even if partially occluded
- Be consistent with labeling decisions across images
- Use the train/val/test split (e.g., 70%/20%/10%)
- Review annotations for errors before training

### Further Reading
- [Ultralytics: Tips for Best Training Results](https://docs.ultralytics.com/guides/model-training-tips/)
- [Roboflow: How to Create a Good Dataset](https://blog.roboflow.com/tips-for-how-to-label-images/)
- [Google ML: Data Preparation and Feature Engineering](https://developers.google.com/machine-learning/data-prep)
- [CVAT Documentation: Annotation Best Practices](https://docs.cvat.ai/docs/manual/advanced/annotation-with-rectangles/)

## Training with YOLO

After creating your dataset, train with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='datasets/my_dataset/data.yaml', epochs=100)
```

## Export & Large Datasets

The export feature creates a zip file containing the complete dataset structure.

**Technical notes:**
- Images are stored uncompressed in the zip (ZIP_STORED) since JPG/PNG are already compressed
- The export modal shows dataset size before download to avoid surprises
- Datasets >100MB display a warning
- Very large datasets (1GB+) may take time to prepare and download

**Browser limits:** Most modern browsers handle multi-GB downloads, but performance varies. For very large datasets, consider:
- Using the dataset directly from the `datasets/` folder
- Compressing with external tools for better ratios
- Splitting into multiple smaller datasets

## Project Structure

```
yolo_dataset_creator/
├── server.py           # Flask backend
├── static/
│   ├── index.html      # Main UI
│   ├── style.css       # Styles
│   └── app.js          # Frontend logic
├── datasets/           # Created datasets (gitignored)
├── requirements.txt
├── CLAUDE.md           # Development documentation
└── README.md
```

## License

MIT
