#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL is required}"
: "${BACKUP_FILE:?BACKUP_FILE is required}"

pg_restore -c -d "$DATABASE_URL" "$BACKUP_FILE"
echo "restored"
