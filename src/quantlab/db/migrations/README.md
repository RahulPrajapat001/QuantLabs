# Alembic migrations

Schema history for QuantLab. The database URL is read from
`QUANTLAB_DATABASE_URL` via Pydantic Settings in `env.py` — never hardcoded here
or in `alembic.ini`.

```bash
# apply everything
alembic upgrade head

# create a new revision from model changes
alembic revision --autogenerate -m "describe change"
```

Migrations must run clean from an empty database on both SQLite (dev) and
PostgreSQL (prod).
