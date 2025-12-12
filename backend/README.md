# AddaxAI Backend

FastAPI backend for the AddaxAI camera trap wildlife analysis platform.

## Setup

### 1. Create virtual environment

```bash
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` and update the paths:

```bash
cp .env.example .env
```

**Required configuration:**
- `USER_DATA_DIR`: Where all project data will be stored (must be absolute path)
- `DATABASE_URL`: SQLite database location
- `MODELS_DIR`, `MODEL_MANIFESTS_DIR`, `MODEL_WEIGHTS_DIR`, `MODEL_ENVIRONMENTS_DIR`: Model storage paths

**Important:** Following DEVELOPERS.md principles, the application will **crash** if required environment variables are not set. This is intentional - we want to fail fast during development.

### 4. Run development server

```bash
./venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Project Structure

```
backend/
├── app/
│   ├── api/          # API route handlers
│   ├── core/         # Core configuration
│   ├── db/           # Database setup
│   ├── models/       # SQLAlchemy models
│   ├── ml/           # ML pipeline (future)
│   ├── scripts/      # Utility scripts (future)
│   └── main.py       # FastAPI application
├── alembic/          # Database migrations
├── tests/            # Tests (future)
├── requirements.txt  # Dependencies
└── pyproject.toml    # Project configuration
```

## Database Migrations

Generate a new migration after model changes:

```bash
./venv/bin/alembic revision --autogenerate -m "description of changes"
```

Apply migrations:

```bash
./venv/bin/alembic upgrade head
```

## Development

### Code Quality

Format code with ruff:

```bash
./venv/bin/ruff format .
```

Lint code:

```bash
./venv/bin/ruff check .
```

Type check with mypy:

```bash
./venv/bin/mypy app/
```

### Testing

Run tests:

```bash
./venv/bin/pytest
```

## Developer Conventions

See [DEVELOPERS.md](../DEVELOPERS.md) in the repository root for project-wide conventions:

1. **Crash early and loudly** - No silent failures
2. **Explicit configuration** - No defaults for critical settings
3. **Type hints everywhere**
4. **Simple solutions** - Avoid cleverness
5. **No backward compatibility** - Free to refactor
