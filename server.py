#!/usr/bin/env python3
"""
YOLO Dataset Creator - Backend Server

A simple Flask server for creating YOLO-format datasets.
"""

import os
import io
import json
import uuid
import yaml
import zipfile
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file, Response

app = Flask(__name__, static_folder='static')

# Configuration
DATASETS_DIR = Path(__file__).parent / 'datasets'
DATASETS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_dataset_path(name):
    """Get the path to a dataset directory."""
    return DATASETS_DIR / name


def load_dataset_config(dataset_path):
    """Load dataset configuration from data.yaml."""
    config_path = dataset_path / 'data.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def save_dataset_config(dataset_path, config):
    """Save dataset configuration to data.yaml."""
    config_path = dataset_path / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_dataset_structure(dataset_path):
    """Create the YOLO dataset directory structure."""
    for split in ['train', 'val', 'test']:
        (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)


def get_image_info(dataset_path, split, filename):
    """Get information about an image including its annotations."""
    image_path = dataset_path / 'images' / split / filename
    label_path = dataset_path / 'labels' / split / (Path(filename).stem + '.txt')

    # Count annotations
    annotation_count = 0
    if label_path.exists():
        with open(label_path, 'r') as f:
            annotation_count = len([line for line in f.readlines() if line.strip()])

    return {
        'filename': filename,
        'split': split,
        'path': f'/api/datasets/{dataset_path.name}/images/{split}/{filename}',
        'annotationCount': annotation_count,
        'hasAnnotations': annotation_count > 0
    }


def get_all_images(dataset_path):
    """Get all images in a dataset across all splits."""
    images = []
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / 'images' / split
        if split_dir.exists():
            for f in split_dir.iterdir():
                if f.is_file() and allowed_file(f.name):
                    images.append(get_image_info(dataset_path, split, f.name))
    return images


# ============ Static File Routes ============

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


# ============ Dataset Routes ============

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all datasets."""
    datasets = []
    for d in DATASETS_DIR.iterdir():
        if d.is_dir() and (d / 'data.yaml').exists():
            config = load_dataset_config(d)
            image_count = len(get_all_images(d))
            datasets.append({
                'name': d.name,
                'classes': config.get('names', {}),
                'imageCount': image_count
            })
    return jsonify(datasets)


@app.route('/api/datasets', methods=['POST'])
def create_dataset():
    """Create a new dataset."""
    data = request.json
    name = data.get('name', '').strip()

    if not name:
        return jsonify({'error': 'Dataset name is required'}), 400

    # Sanitize name
    name = ''.join(c for c in name if c.isalnum() or c in '-_')

    dataset_path = get_dataset_path(name)
    if dataset_path.exists():
        return jsonify({'error': 'Dataset already exists'}), 400

    # Create structure
    create_dataset_structure(dataset_path)

    # Create initial config
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 0,
        'names': {}
    }
    save_dataset_config(dataset_path, config)

    return jsonify({
        'name': name,
        'classes': {},
        'imageCount': 0
    }), 201


@app.route('/api/datasets/<name>', methods=['GET'])
def get_dataset(name):
    """Get dataset information."""
    dataset_path = get_dataset_path(name)
    if not dataset_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    config = load_dataset_config(dataset_path)
    images = get_all_images(dataset_path)

    # Count images per split
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for img in images:
        split_counts[img['split']] += 1

    return jsonify({
        'name': name,
        'classes': config.get('names', {}),
        'imageCount': len(images),
        'splitCounts': split_counts
    })


@app.route('/api/datasets/<name>', methods=['PUT'])
def update_dataset(name):
    """Update dataset configuration (classes)."""
    dataset_path = get_dataset_path(name)
    if not dataset_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    data = request.json
    config = load_dataset_config(dataset_path)

    if 'classes' in data:
        # Convert to proper format {0: 'name', 1: 'name', ...}
        classes = data['classes']
        if isinstance(classes, list):
            classes = {i: c for i, c in enumerate(classes)}
        config['names'] = classes
        config['nc'] = len(classes)

    save_dataset_config(dataset_path, config)

    return jsonify({
        'name': name,
        'classes': config['names'],
        'nc': config['nc']
    })


@app.route('/api/datasets/<name>', methods=['DELETE'])
def delete_dataset(name):
    """Delete a dataset."""
    dataset_path = get_dataset_path(name)
    if not dataset_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    import shutil
    shutil.rmtree(dataset_path)

    return jsonify({'success': True})


# ============ Image Routes ============

@app.route('/api/datasets/<name>/images', methods=['GET'])
def list_images(name):
    """List all images in a dataset."""
    dataset_path = get_dataset_path(name)
    if not dataset_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    images = get_all_images(dataset_path)
    return jsonify(images)


@app.route('/api/datasets/<name>/images', methods=['POST'])
def upload_image(name):
    """Upload an image to the dataset."""
    dataset_path = get_dataset_path(name)
    if not dataset_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    split = request.form.get('split', 'train')
    if split not in ['train', 'val', 'test']:
        split = 'train'

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # Generate unique filename
    ext = file.filename.rsplit('.', 1)[1].lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    filename = f'{timestamp}_{unique_id}.{ext}'

    # Save file
    save_path = dataset_path / 'images' / split / filename
    file.save(save_path)

    return jsonify(get_image_info(dataset_path, split, filename)), 201


@app.route('/api/datasets/<name>/images/<split>/<filename>', methods=['GET'])
def get_image(name, split, filename):
    """Serve an image file."""
    dataset_path = get_dataset_path(name)
    image_path = dataset_path / 'images' / split / filename

    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    return send_file(image_path)


@app.route('/api/datasets/<name>/images/<split>/<filename>', methods=['DELETE'])
def delete_image(name, split, filename):
    """Delete an image and its label file."""
    dataset_path = get_dataset_path(name)
    image_path = dataset_path / 'images' / split / filename
    label_path = dataset_path / 'labels' / split / (Path(filename).stem + '.txt')

    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    image_path.unlink()
    if label_path.exists():
        label_path.unlink()

    return jsonify({'success': True})


# ============ Label Routes ============

@app.route('/api/datasets/<name>/labels/<split>/<filename>', methods=['GET'])
def get_labels(name, split, filename):
    """Get annotations for an image."""
    dataset_path = get_dataset_path(name)
    label_path = dataset_path / 'labels' / split / (Path(filename).stem + '.txt')

    annotations = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        'classId': int(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })

    return jsonify(annotations)


@app.route('/api/datasets/<name>/labels/<split>/<filename>', methods=['PUT'])
def save_labels(name, split, filename):
    """Save annotations for an image."""
    dataset_path = get_dataset_path(name)

    # Verify image exists
    image_path = dataset_path / 'images' / split / filename
    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    label_path = dataset_path / 'labels' / split / (Path(filename).stem + '.txt')

    annotations = request.json

    if not annotations:
        # Remove label file if no annotations
        if label_path.exists():
            label_path.unlink()
        return jsonify({'success': True, 'count': 0})

    # Write YOLO format
    with open(label_path, 'w') as f:
        for ann in annotations:
            class_id = int(ann['classId'])
            x = float(ann['x'])
            y = float(ann['y'])
            w = float(ann['width'])
            h = float(ann['height'])
            f.write(f'{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

    return jsonify({'success': True, 'count': len(annotations)})


# ============ Split Routes ============

@app.route('/api/datasets/<name>/split/<split>/<filename>', methods=['PUT'])
def change_split(name, split, filename):
    """Move an image to a different split."""
    dataset_path = get_dataset_path(name)

    data = request.json
    new_split = data.get('split')

    if new_split not in ['train', 'val', 'test']:
        return jsonify({'error': 'Invalid split'}), 400

    if new_split == split:
        return jsonify({'success': True, 'message': 'Already in this split'})

    # Move image
    old_image = dataset_path / 'images' / split / filename
    new_image = dataset_path / 'images' / new_split / filename

    if not old_image.exists():
        return jsonify({'error': 'Image not found'}), 404

    old_image.rename(new_image)

    # Move label if exists
    label_name = Path(filename).stem + '.txt'
    old_label = dataset_path / 'labels' / split / label_name
    new_label = dataset_path / 'labels' / new_split / label_name

    if old_label.exists():
        old_label.rename(new_label)

    return jsonify({
        'success': True,
        'image': get_image_info(dataset_path, new_split, filename)
    })


# ============ Export Routes ============

@app.route('/api/datasets/<name>/stats', methods=['GET'])
def get_dataset_stats(name):
    """Get dataset statistics including total size."""
    dataset_path = get_dataset_path(name)
    if not dataset_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    total_size = 0
    file_count = 0

    # Calculate size of all files
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            file_path = Path(root) / f
            total_size += file_path.stat().st_size
            file_count += 1

    # Get image counts per split
    split_counts = {}
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / 'images' / split
        if split_dir.exists():
            split_counts[split] = len([f for f in split_dir.iterdir() if f.is_file()])
        else:
            split_counts[split] = 0

    return jsonify({
        'name': name,
        'totalSize': total_size,
        'totalSizeFormatted': format_size(total_size),
        'fileCount': file_count,
        'splitCounts': split_counts
    })


def format_size(size_bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f'{size_bytes:.1f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.1f} TB'


@app.route('/api/datasets/<name>/export', methods=['GET'])
def export_dataset(name):
    """Export dataset as a streaming zip file."""
    dataset_path = get_dataset_path(name)
    if not dataset_path.exists():
        return jsonify({'error': 'Dataset not found'}), 404

    def generate_zip():
        """Generator that yields zip file chunks."""
        # Create zip in memory with streaming
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
            # Walk through all files in the dataset
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = Path(root) / file
                    # Calculate relative path within zip
                    arc_name = str(file_path.relative_to(dataset_path.parent))

                    # Add file to zip
                    zf.write(file_path, arc_name)

                    # Yield data periodically to enable streaming
                    if buffer.tell() > 1024 * 1024:  # Every ~1MB
                        buffer.seek(0)
                        yield buffer.read()
                        buffer.seek(0)
                        buffer.truncate()

        # Yield remaining data
        buffer.seek(0)
        yield buffer.read()

    # For smaller datasets, create zip in memory for simplicity
    # For larger datasets, the streaming approach above helps
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                arc_name = str(file_path.relative_to(dataset_path.parent))
                zf.write(file_path, arc_name)

    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype='application/zip',
        headers={
            'Content-Disposition': f'attachment; filename={name}.zip',
            'Content-Length': len(buffer.getvalue())
        }
    )


if __name__ == '__main__':
    import os
    debug_mode = os.environ.get('FLASK_ENV') != 'production'

    print('Starting YOLO Dataset Creator server...')
    print(f'Datasets directory: {DATASETS_DIR.absolute()}')
    print('Open http://localhost:5001 in your browser')

    # Bind to 0.0.0.0 for Docker compatibility
    app.run(host='0.0.0.0', port=5001, debug=debug_mode)
