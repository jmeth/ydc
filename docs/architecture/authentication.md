# Authentication Subsystem [Ideal State Only]

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Manage user authentication, session handling, and role-based access control (RBAC). Not implemented in MVP - system operates without authentication.

## Components

```
Auth Subsystem [Ideal]
├── Auth Manager
│   ├── User Store (CRUD operations)
│   ├── Session Manager
│   ├── Token Generator (JWT)
│   └── Password Hasher (bcrypt)
├── RBAC Engine
│   ├── Role Definitions
│   ├── Permission Checks
│   └── Resource Guards
├── Middleware
│   ├── Authentication Middleware
│   ├── Authorization Middleware
│   └── Rate Limiter
└── Audit Logger
    ├── Action Logging
    └── Access Logging
```

## Data Models

```python
from enum import Enum
from dataclasses import dataclass
from typing import Set

class Role(Enum):
    ADMIN = "admin"       # Full access to everything
    OPERATOR = "operator" # Can scan, annotate, train, run inference
    VIEWER = "viewer"     # Read-only access

class Permission(Enum):
    # Scan
    SCAN_VIEW = "scan:view"
    SCAN_START = "scan:start"
    SCAN_STOP = "scan:stop"
    SCAN_CONFIGURE = "scan:configure"

    # Dataset
    DATASET_VIEW = "dataset:view"
    DATASET_CREATE = "dataset:create"
    DATASET_EDIT = "dataset:edit"
    DATASET_DELETE = "dataset:delete"
    DATASET_EXPORT = "dataset:export"
    DATASET_IMPORT = "dataset:import"

    # Training
    TRAINING_VIEW = "training:view"
    TRAINING_START = "training:start"
    TRAINING_STOP = "training:stop"

    # Models
    MODEL_VIEW = "model:view"
    MODEL_DELETE = "model:delete"
    MODEL_ACTIVATE = "model:activate"

    # Inference
    INFERENCE_VIEW = "inference:view"
    INFERENCE_START = "inference:start"
    INFERENCE_STOP = "inference:stop"

    # System
    SETTINGS_VIEW = "settings:view"
    SETTINGS_EDIT = "settings:edit"
    USERS_MANAGE = "users:manage"

# Role → Permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.OPERATOR: {
        # Scan
        Permission.SCAN_VIEW, Permission.SCAN_START,
        Permission.SCAN_STOP, Permission.SCAN_CONFIGURE,
        # Dataset
        Permission.DATASET_VIEW, Permission.DATASET_CREATE,
        Permission.DATASET_EDIT, Permission.DATASET_DELETE,
        Permission.DATASET_EXPORT, Permission.DATASET_IMPORT,
        # Training
        Permission.TRAINING_VIEW, Permission.TRAINING_START,
        Permission.TRAINING_STOP,
        # Models
        Permission.MODEL_VIEW, Permission.MODEL_DELETE,
        Permission.MODEL_ACTIVATE,
        # Inference
        Permission.INFERENCE_VIEW, Permission.INFERENCE_START,
        Permission.INFERENCE_STOP,
        # System
        Permission.SETTINGS_VIEW,
    },
    Role.VIEWER: {
        Permission.SCAN_VIEW,
        Permission.DATASET_VIEW,
        Permission.TRAINING_VIEW,
        Permission.MODEL_VIEW,
        Permission.INFERENCE_VIEW,
    }
}

@dataclass
class User:
    id: str
    username: str
    password_hash: str
    role: Role
    created_at: float
    last_login: float = None
    active: bool = True
```

## Authentication Flow

```
┌──────────┐     POST /api/auth/login      ┌──────────────┐
│  Client  │──────────────────────────────▶│ Auth Manager │
│          │  {username, password}         │              │
└──────────┘                               └──────────────┘
                                                  │
                                                  ▼ validate
                                           ┌──────────────┐
                                           │  User Store  │
                                           └──────────────┘
                                                  │
     ┌──────────────────────────────────────────────┘
     │ JWT token
     ▼
┌──────────┐     Authorization: Bearer <token>     ┌──────────────┐
│  Client  │──────────────────────────────────────▶│  Middleware  │
│          │                                       │              │
└──────────┘                                       └──────────────┘
                                                          │
                                                          ▼ check permissions
                                                   ┌──────────────┐
                                                   │ RBAC Engine  │
                                                   └──────────────┘
```

## Auth Middleware

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Dependency to extract and validate current user from JWT token"""
    token = credentials.credentials

    user = auth_manager.validate_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    return user

def require_permission(permission: Permission):
    """Dependency factory to require a specific permission"""
    async def check_permission(user: User = Depends(get_current_user)) -> User:
        if not rbac.has_permission(user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        return user
    return check_permission

# Usage example
@router.post("/api/training/start")
async def start_training(
    config: TrainingConfig,
    user: User = Depends(require_permission(Permission.TRAINING_START))
):
    # Only users with TRAINING_START permission can access
    ...
```

## API Endpoints

```
# Authentication
POST   /api/auth/login          Login, returns JWT token
POST   /api/auth/logout         Invalidate session
POST   /api/auth/refresh        Refresh token
GET    /api/auth/me             Get current user info

# User Management (Admin only)
GET    /api/users               List all users
POST   /api/users               Create user
GET    /api/users/:id           Get user details
PUT    /api/users/:id           Update user
DELETE /api/users/:id           Delete user
PUT    /api/users/:id/role      Change user role
```

## Storage

For ideal state with database:
```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_login REAL,
    active INTEGER DEFAULT 1
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE audit_log (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    action TEXT NOT NULL,
    resource TEXT,
    details TEXT,
    timestamp REAL NOT NULL,
    ip_address TEXT
);
```
