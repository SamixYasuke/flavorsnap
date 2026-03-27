# FlavorSnap Backup and Recovery System

This directory contains comprehensive backup and recovery scripts for the FlavorSnap project, ensuring data safety and disaster recovery capabilities.

## 📁 Files Overview

### Scripts
- **`backup.sh`** - Linux/macOS backup script
- **`backup.bat`** - Windows backup script  
- **`recovery.sh`** - Linux/macOS recovery script
- **`recovery.bat`** - Windows recovery script
- **`test_backup_recovery.sh`** - Comprehensive test suite (Linux/macOS)

### Documentation
- **`README.md`** - This documentation file

## 🚀 Quick Start

### For Linux/macOS Users

```bash
# Make scripts executable
chmod +x scripts/backup.sh scripts/recovery.sh scripts/test_backup_recovery.sh

# Create a backup
./scripts/backup.sh

# List available backups
./scripts/recovery.sh --list

# Recover from backup (interactive)
./scripts/recovery.sh --interactive

# Run tests
./scripts/test_backup_recovery.sh
```

### For Windows Users

```cmd
# Create a backup
scripts\backup.bat

# List available backups
scripts\recovery.bat --list

# Recover from backup
scripts\recovery.bat backup.7z

# Force overwrite existing files during recovery
scripts\recovery.bat --force backup.7z
```

## 📋 Features

### Backup System ✨
- **Automated Backup**: Comprehensive backup of all critical project files
- **Compression**: Uses 7-Zip for efficient compression
- **Metadata Tracking**: JSON metadata with backup information
- **Checksum Verification**: SHA256 checksums for integrity verification
- **Automatic Cleanup**: Configurable retention policy for old backups
- **Logging**: Detailed logs for troubleshooting and audit trails

### Recovery System 🔄
- **Selective Recovery**: Choose what to recover
- **Interactive Mode**: User-friendly interactive recovery
- **Force Override**: Option to overwrite existing files
- **Pre-Recovery Backup**: Automatic backup before recovery operations
- **Validation**: Post-recovery validation and verification
- **Rollback Support**: Pre-recovery backups for rollback capability

### Testing Framework 🧪
- **Comprehensive Tests**: Full test coverage for backup/recovery functions
- **Automated Validation**: Automated integrity and functionality checks
- **Error Handling**: Tests for error conditions and edge cases
- **Performance Testing**: Backup/recovery performance validation

## 📦 What Gets Backed Up

### 🤖 Model Files
- `model.pth` - Main trained model
- `food_classes.txt` - Food classification categories
- Model-related Python files from `ml-model-api/`

### 📊 Dataset
- Complete `dataset/` directory structure
- Training and test data
- All image files and annotations

### 📤 User Uploads
- Complete `uploads/` directory
- All user-submitted images
- Organized by prediction classes

### ⚙️ Configuration
- `package.json` files (root and frontend)
- `requirements.txt` for Python dependencies
- Environment configuration files
- Build configuration files (Cargo.toml, tsconfig.json, etc.)

### 🔧 Scripts & Tools
- All scripts from the `scripts/` directory
- Setup scripts (`setup.*`)
- Training notebooks (`train_model.ipynb`)

### 🗄️ Database Files
- SQLite databases (*.db, *.sqlite, *.sqlite3)
- SQL dump files
- Data directories

## 🔧 Configuration Options

### Backup Script Options

```bash
# Linux/macOS
./backup.sh --keep-count 10    # Keep last 10 backups
./backup.sh --skip-cleanup     # Don't clean up old backups

# Windows
backup.bat --keep-count 10
backup.bat --skip-cleanup
```

### Recovery Script Options

```bash
# Linux/macOS
./recovery.sh --interactive    # Interactive recovery mode
./recovery.sh --force          # Force overwrite existing files
./recovery.sh --list           # List available backups

# Windows
recovery.bat --list
recovery.bat --force backup.7z
```

## 📁 Backup Structure

Each backup creates a compressed archive with the following structure:

```
flavorsnap_backup_YYYYMMDD_HHMMSS.7z
├── models/                    # Model files and weights
├── dataset/                   # Training and test data
├── uploads/                   # User uploaded images
├── config/                    # Configuration files
├── scripts/                   # Scripts and tools
├── database/                  # Database files
├── backup_info.json           # Backup metadata
└── checksums.sha256           # File integrity checksums
```

## 🛡️ Safety Features

### ✅ Integrity Verification
- SHA256 checksums for all files
- Backup metadata validation
- Archive integrity checks

### 🔄 Pre-Recovery Protection
- Automatic pre-recovery backups
- Confirmation prompts for destructive operations
- Rollback capability

### 📝 Comprehensive Logging
- Detailed operation logs
- Error tracking and reporting
- Audit trail for compliance

### 🔒 Error Handling
- Graceful error handling and recovery
- Validation before destructive operations
- Clear error messages and guidance

## 🧪 Testing

### Running Tests

```bash
# Linux/macOS
./scripts/test_backup_recovery.sh

# Windows (manual testing)
# Run backup.bat and recovery.bat with test data
```

### Test Coverage

- ✅ Script existence and permissions
- ✅ Backup creation and compression
- ✅ Backup integrity verification
- ✅ Recovery process functionality
- ✅ Metadata generation
- ✅ Checksum verification
- ✅ File structure validation
- ✅ Cleanup functionality
- ✅ Error handling
- ✅ Permissions validation

## 📊 Monitoring and Maintenance

### Backup Monitoring
- Check backup logs regularly: `backups/backup_*.log`
- Monitor backup file sizes and compression ratios
- Verify backup retention policy compliance

### Recovery Testing
- Test recovery procedures monthly
- Validate backup integrity quarterly
- Document and update recovery procedures

### Storage Management
- Monitor backup storage usage
- Adjust retention policies as needed
- Consider off-site backup for critical data

## 🚨 Emergency Procedures

### Immediate Response
1. **Stop Services**: Stop all FlavorSnap services
2. **Assess Damage**: Identify what needs recovery
3. **Check Backups**: Verify latest backup availability
4. **Plan Recovery**: Determine recovery strategy

### Recovery Execution
1. **Create Pre-Recovery Backup**: Protect current state
2. **Select Backup**: Choose appropriate backup version
3. **Execute Recovery**: Run recovery with appropriate options
4. **Validate**: Verify system functionality
5. **Monitor**: Watch for issues post-recovery

### Post-Recovery
1. **Test All Functions**: Verify complete system functionality
2. **Update Documentation**: Record recovery details
3. **Review Procedures**: Identify improvements
4. **Communicate**: Notify stakeholders of recovery completion

## 🔧 Troubleshooting

### Common Issues

#### Backup Fails
```bash
# Check dependencies
./backup.sh --help

# Verify permissions
ls -la scripts/

# Check disk space
df -h
```

#### Recovery Fails
```bash
# List available backups
./recovery.sh --list

# Verify backup integrity
7z t backup_file.7z

# Check logs
tail -f backups/recovery_*.log
```

#### Permission Issues
```bash
# Fix script permissions
chmod +x scripts/*.sh

# Fix file ownership
sudo chown -R user:group /path/to/project
```

### Getting Help

- Check log files in `backups/` directory
- Review this documentation
- Test with the provided test suite
- Check system dependencies (7-Zip, robocopy, etc.)

## 📈 Best Practices

### Regular Operations
- Schedule automated backups (daily/weekly)
- Test recovery procedures regularly
- Monitor backup success rates
- Maintain adequate storage capacity

### Security Considerations
- Store backups in secure locations
- Encrypt sensitive backup data
- Limit access to backup files
- Regular backup access audits

### Documentation
- Document any custom configurations
- Record recovery procedures and outcomes
- Maintain change logs for backup systems
- Train team members on recovery procedures

## 🔄 Version History

### v1.0.0 (Current)
- Initial release
- Comprehensive backup and recovery system
- Cross-platform support (Linux/macOS/Windows)
- Automated testing suite
- Metadata and checksum support
- Interactive recovery mode
- Pre-recovery backup protection

## 📞 Support

For issues with the backup and recovery system:

1. Check this documentation first
2. Review log files for error details
3. Run the test suite to identify issues
4. Check system dependencies
5. Contact the FlavorSnap development team

---

**Note**: This backup and recovery system is designed to protect your FlavorSnap data. Regular testing and monitoring are essential for ensuring reliable disaster recovery capabilities.
