BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS prediction_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id TEXT,
    user_id TEXT,
    image_filename TEXT,
    label TEXT,
    confidence DOUBLE PRECISION,
    all_predictions JSONB,
    processing_time DOUBLE PRECISION,
    model_version TEXT,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version TEXT NOT NULL,
    metric_date DATE NOT NULL,
    total_predictions INTEGER NOT NULL DEFAULT 0,
    avg_confidence DOUBLE PRECISION,
    avg_processing_time DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_version, metric_date)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
