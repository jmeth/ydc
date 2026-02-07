# Maintainability Recommendations

## server.py

### 1. Remove dead code
The `generate_zip()` generator (lines 448-473) is defined but never used. Delete it.

### 2. Extract common patterns
The "check dataset exists or 404" pattern repeats in most routes. Create a decorator:

```python
def require_dataset(f):
    @functools.wraps(f)
    def wrapper(name, *args, **kwargs):
        dataset_path = get_dataset_path(name)
        if not dataset_path.exists():
            return jsonify({'error': 'Dataset not found'}), 404
        return f(name, dataset_path, *args, **kwargs)
    return wrapper
```

### 3. Move the `shutil` import to the top
Line 216 has it inside `delete_dataset` - should be with other imports.

### 4. Use Flask Blueprints
Organize routes into blueprints (datasets, images, labels, export). Makes it easier to test and split the file later.

### 5. Add type hints
```python
def get_dataset_path(name: str) -> Path:
def load_dataset_config(dataset_path: Path) -> dict | None:
```

### 6. Add input validation
Currently dataset names are only sanitized on creation, but other inputs (splits, filenames) could use validation.

---

## app.js

### 1. Split into modules
The file has clear sections that could be separate files:
- `state.js` - state management
- `api.js` - API calls (already isolated as an object)
- `canvas.js` - canvas rendering and mouse handlers
- `ui.js` - DOM rendering functions
- `modals.js` - modal management

### 2. Replace innerHTML template strings with DOM methods
The current approach (lines 1065-1086, 1105-1113, etc.) is XSS-vulnerable if filenames contain HTML. Use `textContent` for user data:

```javascript
const div = document.createElement('div');
div.textContent = img.filename;  // safe
```

### 3. Extract coordinate conversion utilities
Functions like `getCanvasCoords`, `getNormalizedCoords`, `getPixelCoords`, and the resize handle logic (lines 758-826) could be a separate `coords.js` module.

### 4. Debounce frequent renders
`render()` is called often (every mouse move during drag). The full re-render of image list and label list on every call is wasteful:

```javascript
function render() {
    renderCanvas();  // always needed
    renderAnnotationList();  // depends on selection
}
// Call renderImageList/renderLabelList only when their data changes
```

### 5. Use event delegation
Instead of attaching handlers to each `.image-item` after render (lines 1089-1096), use a single delegated handler on the parent:

```javascript
elements.imageList.addEventListener('click', (e) => {
    const item = e.target.closest('.image-item');
    if (!item) return;
    // handle click
});
```

### 6. Consider JSDoc for IDE support
If not adopting TypeScript:

```javascript
/** @typedef {{classId: number, x: number, y: number, width: number, height: number}} Annotation */
```

---

---

## Testing

### Unit Tests (Python - pytest)

**Setup:**
```bash
pip install pytest pytest-cov
```

**Structure:**
```
tests/
├── conftest.py          # Fixtures (test client, temp datasets)
├── test_api_datasets.py # Dataset CRUD operations
├── test_api_images.py   # Image upload/delete/split
├── test_api_labels.py   # Annotation save/load
└── test_helpers.py      # Utility functions
```

**Key fixtures for `conftest.py`:**
```python
import pytest
import tempfile
from pathlib import Path
from server import app, DATASETS_DIR

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def temp_datasets(tmp_path, monkeypatch):
    """Use a temporary directory for datasets during tests."""
    monkeypatch.setattr('server.DATASETS_DIR', tmp_path)
    return tmp_path
```

**What to test:**
- Dataset creation with valid/invalid names
- Dataset deletion removes all files
- Image upload stores file correctly
- Image upload rejects invalid file types
- Label save/load roundtrip preserves data
- Split change moves both image and label files
- Export produces valid zip with correct structure

### Unit Tests (JavaScript - Vitest or Jest)

**Setup:**
```bash
npm init -y
npm install -D vitest jsdom
```

**Structure:**
```
static/
├── js/
│   ├── api.js           # Extract from app.js
│   ├── coords.js        # Coordinate utilities
│   └── state.js         # State management
└── tests/
    ├── coords.test.js
    └── state.test.js
```

**What to test:**
- `getNormalizedCoords` / `getPixelCoords` roundtrip
- `getResizeHandle` hit detection
- `isInsideBox` boundary conditions
- State transitions (selecting, drawing, dragging)
- Class ID re-indexing after deletion

**Example test:**
```javascript
import { describe, it, expect } from 'vitest';
import { getNormalizedCoords, getPixelCoords } from '../js/coords.js';

describe('coordinate conversion', () => {
    it('roundtrips correctly', () => {
        const imageSize = { width: 1000, height: 800 };
        const box = { x: 100, y: 200, width: 300, height: 400 };

        const normalized = getNormalizedCoords(box, imageSize);
        const back = getPixelCoords(normalized, imageSize);

        expect(back.x).toBeCloseTo(box.x);
        expect(back.y).toBeCloseTo(box.y);
    });
});
```

### End-to-End Tests (Playwright)

**Setup:**
```bash
npm install -D @playwright/test
npx playwright install
```

**Structure:**
```
e2e/
├── playwright.config.js
├── fixtures.js          # Start/stop server, create test data
└── tests/
    ├── dataset.spec.js  # Create, select, delete datasets
    ├── upload.spec.js   # File upload, webcam capture
    ├── annotate.spec.js # Draw boxes, resize, move, delete
    └── export.spec.js   # Download and verify zip
```

**Key test scenarios:**

1. **Dataset workflow:**
   - Create new dataset
   - Verify it appears in dropdown
   - Delete dataset
   - Verify it's removed

2. **Annotation workflow:**
   - Upload image
   - Draw bounding box
   - Verify box appears in annotation list
   - Change box class via keyboard (1-9)
   - Save and reload page
   - Verify annotations persist

3. **Image navigation:**
   - Upload multiple images
   - Navigate with arrow keys
   - Verify correct image loads
   - Verify unsaved changes prompt

4. **Canvas interactions:**
   - Draw box with mouse drag
   - Select box by clicking
   - Move box by dragging
   - Resize box via corner handles
   - Delete box with Delete key

**Example Playwright test:**
```javascript
import { test, expect } from '@playwright/test';

test('create dataset and upload image', async ({ page }) => {
    await page.goto('http://localhost:5001');

    // Create dataset
    await page.click('#new-dataset-btn');
    await page.fill('#dataset-name', 'test-dataset');
    await page.click('#create-dataset');

    // Verify selected
    await expect(page.locator('#dataset-select')).toHaveValue('test-dataset');

    // Upload image
    const fileInput = page.locator('#file-input');
    await fileInput.setInputFiles('e2e/fixtures/sample.jpg');

    // Verify image appears
    await expect(page.locator('.image-item')).toHaveCount(1);
});
```

**Playwright config (`playwright.config.js`):**
```javascript
export default {
    testDir: './e2e/tests',
    webServer: {
        command: 'python server.py',
        port: 5001,
        reuseExistingServer: !process.env.CI,
    },
    use: {
        baseURL: 'http://localhost:5001',
    },
};
```

### CI Integration

Add to `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Python deps
        run: pip install flask pytest pytest-cov

      - name: Run Python tests
        run: pytest tests/ --cov=server

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install Node deps
        run: npm ci

      - name: Run JS unit tests
        run: npm test

      - name: Install Playwright
        run: npx playwright install --with-deps

      - name: Run E2E tests
        run: npx playwright test
```

---

## Release Pipeline

### Overview

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Lint   │───▶│  Scan   │───▶│  Test   │───▶│  Build  │───▶│ Release │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### Linting

**Python (Ruff):**
```bash
pip install ruff
```

```yaml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["E501"]  # line length handled separately
```

**JavaScript (ESLint):**
```bash
npm install -D eslint
```

```javascript
// eslint.config.js
export default [
    {
        files: ["static/**/*.js"],
        rules: {
            "no-unused-vars": "error",
            "no-undef": "error",
            "eqeqeq": "warn",
        },
        languageOptions: {
            globals: {
                document: "readonly",
                window: "readonly",
                navigator: "readonly",
                fetch: "readonly",
                Image: "readonly",
                File: "readonly",
                FormData: "readonly",
                Blob: "readonly",
                confirm: "readonly",
            }
        }
    }
];
```

**CSS (Stylelint):**
```bash
npm install -D stylelint stylelint-config-standard
```

```json
// .stylelintrc.json
{
    "extends": "stylelint-config-standard",
    "rules": {
        "selector-class-pattern": null
    }
}
```

### Security Scanning

**Dependency scanning:**
```bash
# Python
pip install pip-audit
pip-audit

# JavaScript
npm audit
```

**Static analysis (Bandit for Python):**
```bash
pip install bandit
bandit -r server.py
```

**Container scanning (Trivy):**
```bash
trivy image yolo-dataset-creator:latest
```

**Secret scanning (Gitleaks):**
```bash
gitleaks detect --source .
```

### Container Build

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY server.py .
COPY static/ static/

# Create datasets directory
RUN mkdir -p datasets

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

ENV FLASK_ENV=production

CMD ["python", "server.py"]
```

**requirements.txt:**
```
flask>=3.0.0
pyyaml>=6.0
```

**.dockerignore:**
```
datasets/
*.pyc
__pycache__/
.git/
.github/
node_modules/
e2e/
tests/
*.md
.env
```

### GitHub Actions Workflow

**`.github/workflows/release.yml`:**
```yaml
name: Release

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install linters
        run: |
          pip install ruff bandit
          npm ci

      - name: Lint Python
        run: ruff check server.py

      - name: Lint JavaScript
        run: npx eslint static/

      - name: Lint CSS
        run: npx stylelint "static/**/*.css"

  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Dependency scan (Python)
        run: |
          pip install pip-audit
          pip install -r requirements.txt
          pip-audit

      - name: Dependency scan (Node)
        run: npm audit --audit-level=high

      - name: Security scan (Bandit)
        run: |
          pip install bandit
          bandit -r server.py -ll

      - name: Secret scan
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  test:
    needs: [lint, scan]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: |
          pip install flask pyyaml pytest pytest-cov
          npm ci

      - name: Run Python tests
        run: pytest tests/ --cov=server --cov-report=xml

      - name: Run JS tests
        run: npm test

      - name: Install Playwright
        run: npx playwright install --with-deps chromium

      - name: Run E2E tests
        run: npx playwright test

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml

  build:
    needs: [test]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    outputs:
      digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan container
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}
          format: 'sarif'
          output: 'trivy-results.sarif'
        if: github.event_name != 'pull_request'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
        if: github.event_name != 'pull_request'

  release:
    needs: [build]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        run: |
          # Get commits since last tag
          PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")
          if [ -n "$PREV_TAG" ]; then
            CHANGES=$(git log --pretty=format:"- %s" $PREV_TAG..HEAD)
          else
            CHANGES=$(git log --pretty=format:"- %s")
          fi
          echo "changes<<EOF" >> $GITHUB_OUTPUT
          echo "$CHANGES" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          body: |
            ## Changes
            ${{ steps.changelog.outputs.changes }}

            ## Container Image
            ```bash
            docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }}
            ```

            ## Run
            ```bash
            docker run -p 5001:5001 -v $(pwd)/datasets:/app/datasets ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }}
            ```
          generate_release_notes: true
```

### Release Process

1. **Development:** Work on feature branch, PR to main
2. **PR checks:** Lint → Scan → Test (no container push)
3. **Merge to main:** Full pipeline, pushes `main` and `sha-xxxxx` tags to registry
4. **Create release:** Tag with `v1.2.3`, triggers:
   - Full pipeline
   - Pushes `v1.2.3`, `1.2`, `latest` tags
   - Creates GitHub Release with changelog and container pull instructions

**Tagging a release:**
```bash
git tag v1.0.0
git push origin v1.0.0
```

### Local Development Commands

Add to `package.json`:
```json
{
  "scripts": {
    "lint": "eslint static/ && stylelint 'static/**/*.css'",
    "lint:py": "ruff check server.py && bandit -r server.py -ll",
    "test": "vitest run",
    "test:e2e": "playwright test",
    "docker:build": "docker build -t yolo-dataset-creator .",
    "docker:run": "docker run -p 5001:5001 -v $(pwd)/datasets:/app/datasets yolo-dataset-creator"
  }
}
```

Add to `Makefile`:
```makefile
.PHONY: lint test build run

lint:
	ruff check server.py
	bandit -r server.py -ll
	npm run lint

test:
	pytest tests/
	npm test

e2e:
	npx playwright test

build:
	docker build -t yolo-dataset-creator .

run:
	docker run -p 5001:5001 -v $$(pwd)/datasets:/app/datasets yolo-dataset-creator
```

---

## Quick Wins (minimal effort, high impact)

| Priority | Change | Category |
|----------|--------|----------|
| High | Remove dead `generate_zip()` code | Code quality |
| High | Move `shutil` import to top | Code quality |
| High | Use `textContent` instead of innerHTML for user data | Security |
| High | Add Ruff for Python linting | Pipeline |
| High | Add `requirements.txt` | Pipeline |
| Medium | Extract `api.js` as a separate module | Code quality |
| Medium | Add the `require_dataset` decorator | Code quality |
| Medium | Add pytest for backend API tests | Testing |
| Medium | Add Playwright for critical path E2E | Testing |
| Medium | Add Dockerfile | Pipeline |
| Medium | Add basic GitHub Actions workflow | Pipeline |
| Lower | Full JS modularization | Code quality |
| Lower | JS unit tests (requires modularization first) | Testing |
| Lower | Container scanning with Trivy | Security |
| Lower | Full release automation | Pipeline |
