BEGIN;

CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at
ON prediction_history (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_history_label
ON prediction_history (label);

CREATE INDEX IF NOT EXISTS idx_prediction_history_user_id
ON prediction_history (user_id);

CREATE INDEX IF NOT EXISTS idx_prediction_history_model_version
ON prediction_history (model_version);

COMMIT;
