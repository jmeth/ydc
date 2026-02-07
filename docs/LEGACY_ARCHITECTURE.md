# YOLO Dataset Creator

A simple web-based tool for creating YOLO-format datasets for object detection training.

## Project Overview

This tool allows users to:
- Upload images or capture them directly from a webcam
- Draw bounding boxes around objects
- Assign class labels to each box
- Organize images into train/val/test splits
- Export datasets in YOLO format compatible with Ultralytics

## Architecture

### Tech Stack
- **Frontend**: Plain HTML/CSS/JavaScript (no build step)
- **Backend**: Python with Flask
- **Storage**: Local filesystem in YOLO directory structure

### Directory Structure
```
yolo_dataset_creator/
├── server.py              # Flask backend server
├── static/
│   ├── index.html         # Main application UI
│   ├── style.css          # Styles
│   └── app.js             # Frontend application logic
├── datasets/              # Dataset storage (gitignored)
│   └── <dataset_name>/
│       ├── data.yaml      # YOLO dataset configuration
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
└── CLAUDE.md
```

### YOLO Format Reference

**Label files**: One `.txt` per image with same filename, containing:
```
class_id x_center y_center width height
```
- Coordinates are normalized (0-1) relative to image dimensions
- Class IDs are zero-indexed integers

**data.yaml**:
```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
nc: <number_of_classes>
names:
  0: class_name_1
  1: class_name_2
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets` | List all datasets |
| POST | `/api/datasets` | Create new dataset |
| POST | `/api/datasets/import` | Import dataset from zip file |
| GET | `/api/datasets/<name>` | Get dataset info (classes, stats) |
| PUT | `/api/datasets/<name>` | Update dataset (classes, etc.) |
| GET | `/api/datasets/<name>/images` | List all images with metadata |
| POST | `/api/datasets/<name>/images` | Upload image (multipart form) |
| DELETE | `/api/datasets/<name>/images/<id>` | Delete image and its label |
| GET | `/api/datasets/<name>/labels/<image_id>` | Get annotations for image |
| PUT | `/api/datasets/<name>/labels/<image_id>` | Save annotations for image |
| PUT | `/api/datasets/<name>/split/<image_id>` | Move image to different split |

## Frontend Components

### Canvas Editor (`app.js`)
- Image display with zoom/pan support
- Bounding box drawing (click-drag)
- Box selection with visual feedback
- Box moving (drag selected box)
- Box resizing (drag corner or edge handles)
- Label changing (click label or press 1-9 with box selected)
- Coordinate normalization for YOLO format

### Label Manager
- Create/edit/delete class labels
- Color assignment for visualization
- Keyboard shortcuts: 1-9 for quick label selection

### Image Browser
- Thumbnail grid or list view
- Annotation status indicators (annotated/empty)
- Click to load image into editor
- Keyboard navigation: arrow keys, Page Up/Down

### Capture Panel
- File upload (drag-drop or file picker)
- Webcam capture with preview
- Batch upload support

### Split Selector
- Radio buttons: train/val/test
- Moves image between split directories

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1-9 | Select label class / Change selected box label |
| Delete/Backspace | Delete selected box |
| Shift+Delete | Delete current image |
| Escape | Deselect / Cancel drawing |
| Arrow Left/Right | Previous/Next image |
| Ctrl+S | Save annotations |
| N | New bounding box mode |
| V | Select mode |
| W | Open webcam capture |

## Mouse Interactions

| Action | Description |
|--------|-------------|
| Click box | Select box |
| Drag box | Move box |
| Drag corner handle | Resize box (proportional) |
| Drag edge handle | Resize box (single axis) |
| Click label (with box selected) | Change box label |

## Development

### Running the Server
```bash
cd yolo_dataset_creator
pip install flask
python server.py
```
Server runs at `http://localhost:5000`

### Dependencies
- Python 3.8+
- Flask

## Implementation Notes

- Images are stored as-is (no resizing) to preserve quality
- Labels use UUID-based filenames to avoid conflicts
- Webcam capture saves as JPEG with timestamp filename
- All file operations go through the backend API
- Frontend uses vanilla JS with no external dependencies
