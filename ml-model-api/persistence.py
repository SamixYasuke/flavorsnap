import json
from datetime import datetime, date
from typing import Any, Dict, Optional

from db_config import get_connection


def _ensure_uuid_func(cur):
    cur.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp') THEN
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            END IF;
        END$$;
        """
    )


def log_prediction_history(payload: Dict[str, Any], duration: float, status: str, request_meta: Optional[Dict[str, Any]] = None):
    conn = get_connection()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                _ensure_uuid_func(cur)
                cur.execute(
                    """
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
                    """
                )
                label = payload.get("label")
                confidence = payload.get("confidence")
                all_preds = payload.get("all_predictions") or payload.get("predictions")
                model_version = payload.get("model_version") or payload.get("model") or None
                image_filename = payload.get("filename") or payload.get("image") or None
                request_id = (request_meta or {}).get("request_id")
                user_id = (request_meta or {}).get("user_id")
                error_message = (request_meta or {}).get("error_message")

                cur.execute(
                    """
                    INSERT INTO prediction_history
                    (request_id, user_id, image_filename, label, confidence, all_predictions, processing_time, model_version, success, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    """,
                    (
                        request_id,
                        user_id,
                        image_filename,
                        label,
                        confidence,
                        json.dumps(all_preds) if all_preds is not None else json.dumps([]),
                        duration,
                        model_version,
                        status == "success",
                        error_message,
                    ),
                )

                # Upsert model performance daily aggregates
                cur.execute(
                    """
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
                    """
                )
                metric_date = date.today()
                cur.execute(
                    """
                    INSERT INTO model_performance_metrics (model_version, metric_date, total_predictions, avg_confidence, avg_processing_time)
                    VALUES (%s, %s, 1, %s, %s)
                    ON CONFLICT (model_version, metric_date)
                    DO UPDATE SET
                        total_predictions = model_performance_metrics.total_predictions + 1,
                        avg_confidence = CASE
                            WHEN EXCLUDED.avg_confidence IS NULL THEN model_performance_metrics.avg_confidence
                            WHEN model_performance_metrics.avg_confidence IS NULL THEN EXCLUDED.avg_confidence
                            ELSE (model_performance_metrics.avg_confidence * model_performance_metrics.total_predictions + EXCLUDED.avg_confidence) / (model_performance_metrics.total_predictions + 1)
                        END,
                        avg_processing_time = CASE
                            WHEN EXCLUDED.avg_processing_time IS NULL THEN model_performance_metrics.avg_processing_time
                            WHEN model_performance_metrics.avg_processing_time IS NULL THEN EXCLUDED.avg_processing_time
                            ELSE (model_performance_metrics.avg_processing_time * model_performance_metrics.total_predictions + EXCLUDED.avg_processing_time) / (model_performance_metrics.total_predictions + 1)
                        END;
                    """,
                    (
                        model_version or "unknown",
                        metric_date,
                        confidence if isinstance(confidence, (int, float)) else None,
                        duration if isinstance(duration, (int, float)) else None,
                    ),
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def purge_old_history(days: int) -> int:
    conn = get_connection()
    if not conn:
        return 0
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM prediction_history
                    WHERE created_at < NOW() - INTERVAL '%s days'
                    """,
                    (days,),
                )
                deleted = cur.rowcount
                return deleted or 0
    finally:
        try:
            conn.close()
        except Exception:
            pass
