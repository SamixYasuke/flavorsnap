#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL is required}"

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="${BACKUP_DIR:-./backups}"
mkdir -p "$backup_dir"
outfile="$backup_dir/flavorsnap_backup_${timestamp}.sql"

pg_dump "$DATABASE_URL" -Fc -f "$outfile"
echo "$outfile"
