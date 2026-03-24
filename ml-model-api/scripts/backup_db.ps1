Param(
  [Parameter(Mandatory=$true)][string]$DatabaseUrl,
  [string]$BackupDir = ".\backups"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
$outfile = Join-Path $BackupDir "flavorsnap_backup_$timestamp.sql"

pg_dump $DatabaseUrl -Fc -f $outfile
Write-Output $outfile
