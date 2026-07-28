# ATLAS API Documentation

## Overview

ATLAS exposes a FastAPI-based HTTP API. This document describes the Phase 1 API surface, which provides health monitoring and kernel introspection.

> **Note:** Business logic APIs (tasks, memory, tools) will be added in Phases 3-9. Phase 1 only exposes kernel-level endpoints.

---

## Base URL

```
http://localhost:8000/api/v1
```

---

## Endpoints

### Health

#### `GET /health`

Returns the overall health status of ATLAS.

**Response 200:**

```json
{
    "status": "healthy",
    "version": "0.1.0",
    "uptime_seconds": 3600.5,
    "timestamp": "2026-07-28T12:00:00Z"
}
```

**Response 503 (unhealthy):**

```json
{
    "status": "unhealthy",
    "version": "0.1.0",
    "uptime_seconds": 10.2,
    "timestamp": "2026-07-28T12:00:00Z",
    "details": {
        "database": "unreachable"
    }
}
```

#### `GET /health/ready`

Returns readiness status (is ATLAS ready to accept traffic?).

**Response 200:**

```json
{
    "ready": true,
    "modules": {
        "total": 5,
        "active": 5,
        "failed": 0
    }
}
```

#### `GET /health/live`

Returns liveness status (is ATLAS process alive?).

**Response 200:**

```json
{
    "alive": true,
    "kernel": "running"
}
```

### System

#### `GET /system/modules`

Returns all registered modules and their status.

**Response 200:**

```json
{
    "modules": [
        {
            "name": "config",
            "version": "1.0.0",
            "status": "active",
            "dependencies": []
        },
        {
            "name": "logging",
            "version": "1.0.0",
            "status": "active",
            "dependencies": ["config"]
        }
    ]
}
```

#### `GET /system/events`

Returns event bus statistics.

**Response 200:**

```json
{
    "total_events_emitted": 1024,
    "registered_event_types": 12,
    "registered_handlers": 48,
    "events_by_type": {
        "TaskCreated": 150,
        "TaskCompleted": 140
    }
}
```

#### `GET /system/config`

Returns the current configuration (sanitized — secrets masked).

**Response 200:**

```json
{
    "app": {
        "name": "atlas",
        "version": "0.1.0",
        "environment": "development"
    },
    "logging": {
        "level": "DEBUG",
        "sinks": ["console", "file"]
    },
    "database": {
        "host": "localhost",
        "port": 5432,
        "database": "atlas",
        "username": "***"
    }
}
```

---

## Future Endpoints (Phase 3+)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tasks` | Create a task |
| GET | `/tasks/{id}` | Get task status |
| GET | `/tasks` | List tasks |
| POST | `/agents` | Create/spawn an agent |
| DELETE | `/agents/{id}` | Terminate an agent |
| GET | `/memory/search` | Search memory |
| GET | `/tools` | List available tools |
| POST | `/tools/{name}/execute` | Execute a tool |

---

## Error Format

All errors follow a consistent format:

```json
{
    "error": {
        "code": "MODULE_NOT_FOUND",
        "message": "Module 'xyz' is not registered",
        "details": {
            "module_name": "xyz",
            "registered_modules": ["a", "b", "c"]
        }
    }
}
```

---

## Authentication

Phase 1 has no authentication. Authentication will be added in Phase 9 (API Gateway).

---

## WebSocket

WebSocket support for streaming responses will be added in Phase 9.
