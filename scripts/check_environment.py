#!/usr/bin/env python3
"""
FlavorSnap Environment Validation Script

This script validates the FlavorSnap installation environment by checking:
- System requirements
- Dependencies and versions
- Configuration files
- Model availability
- Network connectivity
- GPU support (if applicable)

Usage:
    python scripts/check_environment.py [options]

Options:
    --verbose           Verbose output
    --fix               Attempt to fix common issues
    --check-gpu         Check GPU configuration
    --check-network     Check network connectivity
    --check-config      Validate configuration files
    --help              Show this help message
"""

import argparse
import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """Environment validation class for FlavorSnap"""
    
    def __init__(self, args):
        self.args = args
        self.platform = platform.system().lower()
        self.project_root = Path(__file__).parent.parent
        self.issues = []
        self.warnings = []
        self.fixes_applied = []
        
        # Minimum requirements
        self.requirements = {
            'python': {'min_version': (3, 8), 'recommended': (3, 9)},
            'node': {'min_version': (18, 0), 'recommended': (18, 17)},
            'ram': {'min_mb': 4096, 'recommended_mb': 8192},
            'storage': {'min_mb': 10240, 'recommended_mb': 20480}
        }
        
    def run_command(self, command: str, shell: bool = True) -> Tuple[bool, str, str]:
        """Run a command and return success, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=30
            )
            return True, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def check_python(self) -> Dict:
        """Check Python installation and version"""
        result = {'status': 'unknown', 'version': None, 'issues': []}
        
        # Check Python command
        python_cmds = ['python', 'python3', 'py']
        python_cmd = None
        
        for cmd in python_cmds:
            success, stdout, _ = self.run_command(f"{cmd} --version")
            if success:
                python_cmd = cmd
                version_str = stdout.replace('Python ', '').strip()
                result['version'] = version_str
                break
        
        if not python_cmd:
            result['status'] = 'error'
            result['issues'].append('Python not found')
            self.issues.append('Python is not installed or not in PATH')
            return result
        
        # Parse version
        try:
            version_parts = [int(x) for x in result['version'].split('.')[:2]]
            result['version_tuple'] = tuple(version_parts)
            
            # Check minimum version
            if version_parts >= self.requirements['python']['min_version']:
                result['status'] = 'ok'
                
                # Check if recommended version
                if version_parts >= self.requirements['python']['recommended']:
                    result['status'] = 'excellent'
                else:
                    result['issues'].append(f'Python {result["version"]} is below recommended {self.requirements["python"]["recommended"][0]}.{self.requirements["python"]["recommended"][1]}')
                    self.warnings.append(f'Consider upgrading to Python {self.requirements["python"]["recommended"][0]}.{self.requirements["python"]["recommended"][1]}+')
            else:
                result['status'] = 'error'
                result['issues'].append(f'Python {result["version"]} is below minimum {self.requirements["python"]["min_version"][0]}.{self.requirements["python"]["min_version"][1]}')
                self.issues.append(f'Python {self.requirements["python"]["min_version"][0]}.{self.requirements["python"]["min_version"][1]}+ is required')
        
        except ValueError:
            result['status'] = 'error'
            result['issues'].append('Could not parse Python version')
            self.issues.append('Python version parsing failed')
        
        return result
    
    def check_node(self) -> Dict:
        """Check Node.js installation and version"""
        result = {'status': 'unknown', 'version': None, 'npm_version': None, 'issues': []}
        
        # Check Node.js
        success, stdout, _ = self.run_command("node --version")
        if success:
            version_str = stdout.replace('v', '').strip()
            result['version'] = version_str
            
            try:
                version_parts = [int(x) for x in version_str.split('.')[:2]]
                result['version_tuple'] = tuple(version_parts)
                
                if version_parts >= self.requirements['node']['min_version']:
                    result['status'] = 'ok'
                    
                    if version_parts >= self.requirements['node']['recommended']:
                        result['status'] = 'excellent'
                    else:
                        result['issues'].append(f'Node.js {version_str} is below recommended {self.requirements["node"]["recommended"][0]}.{self.requirements["node"]["recommended"][1]}')
                        self.warnings.append(f'Consider upgrading to Node.js {self.requirements["node"]["recommended"][0]}.{self.requirements["node"]["recommended"][1]}+')
                else:
                    result['status'] = 'error'
                    result['issues'].append(f'Node.js {version_str} is below minimum {self.requirements["node"]["min_version"][0]}.{self.requirements["node"]["min_version"][1]}')
                    self.issues.append(f'Node.js {self.requirements["node"]["min_version"][0]}.{self.requirements["node"]["min_version"][1]}+ is required')
            
            except ValueError:
                result['status'] = 'error'
                result['issues'].append('Could not parse Node.js version')
                self.issues.append('Node.js version parsing failed')
        else:
            result['status'] = 'error'
            result['issues'].append('Node.js not found')
            self.issues.append('Node.js is not installed or not in PATH')
        
        # Check npm
        success, stdout, _ = self.run_command("npm --version")
        if success:
            result['npm_version'] = stdout.strip()
        else:
            result['issues'].append('npm not found')
            self.issues.append('npm is not installed or not in PATH')
        
        return result
    
    def check_git(self) -> Dict:
        """Check Git installation"""
        result = {'status': 'unknown', 'version': None, 'issues': []}
        
        success, stdout, _ = self.run_command("git --version")
        if success:
            result['version'] = stdout.strip()
            result['status'] = 'ok'
        else:
            result['status'] = 'error'
            result['issues'].append('Git not found')
            self.issues.append('Git is not installed or not in PATH')
        
        return result
    
    def check_docker(self) -> Dict:
        """Check Docker and Docker Compose"""
        result = {'status': 'unknown', 'docker_version': None, 'compose_version': None, 'issues': []}
        
        # Check Docker
        success, stdout, _ = self.run_command("docker --version")
        if success:
            result['docker_version'] = stdout.strip()
            result['status'] = 'ok'
        else:
            result['issues'].append('Docker not found')
            self.warnings.append('Docker is not installed (optional for manual installation)')
        
        # Check Docker Compose
        compose_cmds = ['docker-compose --version', 'docker compose version']
        for cmd in compose_cmds:
            success, stdout, _ = self.run_command(cmd)
            if success:
                result['compose_version'] = stdout.strip()
                break
        
        if not result['compose_version'] and result['docker_version']:
            result['issues'].append('Docker Compose not found')
            self.warnings.append('Docker Compose is not installed (optional for manual installation)')
        
        if result['docker_version'] and result['compose_version']:
            result['status'] = 'excellent'
        elif result['docker_version']:
            result['status'] = 'ok'
        
        return result
    
    def check_system_resources(self) -> Dict:
        """Check system resources (RAM, Storage)"""
        result = {'status': 'ok', 'ram_mb': 0, 'storage_mb': 0, 'issues': []}
        
        try:
            # Check RAM
            if self.platform == 'windows':
                success, stdout, _ = self.run_command('wmic computersystem get TotalPhysicalMemory')
                if success:
                    lines = stdout.split('\n')
                    for line in lines:
                        if line.strip() and 'TotalPhysicalMemory' not in line:
                            ram_bytes = int(line.strip())
                            result['ram_mb'] = ram_bytes // (1024 * 1024)
                            break
            elif self.platform == 'darwin':
                success, stdout, _ = self.run_command('sysctl hw.memsize')
                if success:
                    ram_bytes = int(stdout.split(':')[1].strip())
                    result['ram_mb'] = ram_bytes // (1024 * 1024)
            else:  # Linux
                success, stdout, _ = self.run_command('free -m')
                if success:
                    lines = stdout.split('\n')
                    for line in lines:
                        if line.startswith('Mem:'):
                            result['ram_mb'] = int(line.split()[1])
                            break
            
            # Check storage
            if self.platform == 'windows':
                success, stdout, _ = self.run_command(f'wmic logicaldisk get size,freespace,caption')
                if success:
                    lines = stdout.split('\n')
                    for line in lines:
                        if ':' in line and line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                free_bytes = int(parts[1])
                                result['storage_mb'] = max(result['storage_mb'], free_bytes // (1024 * 1024))
            else:
                success, stdout, _ = self.run_command(f'df -m {self.project_root}')
                if success:
                    lines = stdout.split('\n')
                    for line in lines:
                        if line.startswith('/') and not line.startswith('Filesystem'):
                            result['storage_mb'] = int(line.split()[3])
                            break
            
            # Validate resources
            if result['ram_mb'] < self.requirements['ram']['min_mb']:
                result['status'] = 'error'
                result['issues'].append(f'Insufficient RAM: {result["ram_mb"]}MB < {self.requirements["ram"]["min_mb"]}MB minimum')
                self.issues.append(f'System needs at least {self.requirements["ram"]["min_mb"]}MB RAM')
            elif result['ram_mb'] < self.requirements['ram']['recommended_mb']:
                result['status'] = 'ok'
                result['issues'].append(f'Low RAM: {result["ram_mb"]}MB < {self.requirements["ram"]["recommended_mb"]}MB recommended')
                self.warnings.append(f'Consider upgrading to {self.requirements["ram"]["recommended_mb"]}MB RAM for better performance')
            
            if result['storage_mb'] < self.requirements['storage']['min_mb']:
                result['status'] = 'error'
                result['issues'].append(f'Insufficient storage: {result["storage_mb"]}MB < {self.requirements["storage"]["min_mb"]}MB minimum')
                self.issues.append(f'System needs at least {self.requirements["storage"]["min_mb"]}MB free storage')
        
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'Could not check system resources: {e}')
            self.warnings.append('Could not verify system resources')
        
        return result
    
    def check_project_structure(self) -> Dict:
        """Check project structure and files"""
        result = {'status': 'ok', 'missing_files': [], 'issues': []}
        
        required_files = [
            'README.md',
            'config.yaml',
            '.env.example',
            'model.pth',
            'food_classes.txt',
            'frontend/package.json',
            'ml-model-api/requirements.txt',
            'frontend/pages/index.tsx'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                result['missing_files'].append(file_path)
                result['status'] = 'warning'
        
        # Check critical files
        critical_files = ['model.pth', 'frontend/package.json', 'ml-model-api/requirements.txt']
        for file_path in critical_files:
            if file_path in result['missing_files']:
                result['status'] = 'error'
                self.issues.append(f'Critical file missing: {file_path}')
        
        if result['missing_files']:
            result['issues'].append(f'Missing files: {", ".join(result["missing_files"])}')
            self.warnings.append(f'Some files are missing: {", ".join(result["missing_files"])}')
        
        return result
    
    def check_python_packages(self) -> Dict:
        """Check Python packages"""
        result = {'status': 'ok', 'packages': {}, 'missing_packages': [], 'issues': []}
        
        required_packages = [
            'torch',
            'torchvision', 
            'flask',
            'pillow',
            'numpy',
            'requests'
        ]
        
        # Check if we're in a virtual environment
        venv_path = self.project_root / 'ml-model-api' / 'venv'
        if venv_path.exists():
            # Activate virtual environment and check packages
            if self.platform == 'windows':
                activate_cmd = str(venv_path / 'Scripts' / 'activate')
            else:
                activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
            
            for package in required_packages:
                success, stdout, _ = self.run_command(f"{activate_cmd} && python -c \"import {package}; print({package}.__version__)\"")
                if success:
                    result['packages'][package] = stdout.strip()
                else:
                    result['missing_packages'].append(package)
        else:
            # Check in current environment
            for package in required_packages:
                success, stdout, _ = self.run_command(f"python -c \"import {package}; print({package}.__version__)\"")
                if success:
                    result['packages'][package] = stdout.strip()
                else:
                    result['missing_packages'].append(package)
        
        if result['missing_packages']:
            result['status'] = 'warning'
            result['issues'].append(f'Missing Python packages: {", ".join(result["missing_packages"])}')
            self.warnings.append(f'Some Python packages are missing: {", ".join(result["missing_packages"])}')
        
        return result
    
    def check_node_packages(self) -> Dict:
        """Check Node.js packages"""
        result = {'status': 'ok', 'node_modules_exists': False, 'issues': []}
        
        frontend_dir = self.project_root / 'frontend'
        node_modules_dir = frontend_dir / 'node_modules'
        
        if node_modules_dir.exists():
            result['node_modules_exists'] = True
            result['status'] = 'excellent'
        else:
            result['status'] = 'warning'
            result['issues'].append('node_modules not found - run npm install')
            self.warnings.append('Frontend dependencies not installed - run npm install in frontend directory')
        
        return result
    
    def check_gpu_support(self) -> Dict:
        """Check GPU support"""
        result = {'status': 'ok', 'cuda_available': False, 'gpu_info': None, 'issues': []}
        
        if not self.args.check_gpu:
            return result
        
        try:
            # Check CUDA availability
            success, stdout, _ = self.run_command('python -c "import torch; print(torch.cuda.is_available())"')
            if success:
                result['cuda_available'] = 'True' in stdout
                
                if result['cuda_available']:
                    result['status'] = 'excellent'
                    # Get GPU info
                    success, gpu_stdout, _ = self.run_command('python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'No GPU\')"')
                    if success:
                        result['gpu_info'] = gpu_stdout
                else:
                    result['issues'].append('CUDA not available - GPU acceleration disabled')
                    self.warnings.append('CUDA not available - consider installing CUDA for GPU acceleration')
            
            # Check NVIDIA driver
            success, stdout, _ = self.run_command('nvidia-smi')
            if success:
                result['nvidia_smi'] = stdout.split('\n')[0]  # First line with driver version
            else:
                result['issues'].append('NVIDIA driver not found')
                self.warnings.append('NVIDIA driver not installed - required for GPU support')
        
        except Exception as e:
            result['status'] = 'warning'
            result['issues'].append(f'Could not check GPU support: {e}')
        
        return result
    
    def check_network_connectivity(self) -> Dict:
        """Check network connectivity"""
        result = {'status': 'ok', 'connectivity': {}, 'issues': []}
        
        if not self.args.check_network:
            return result
        
        urls = [
            ('Google', 'https://www.google.com'),
            ('GitHub', 'https://github.com'),
            ('PyPI', 'https://pypi.org'),
            ('NPM', 'https://www.npmjs.com')
        ]
        
        for name, url in urls:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    result['connectivity'][name] = 'ok'
            except urllib.error.URLError as e:
                result['connectivity'][name] = 'failed'
                result['issues'].append(f'Cannot connect to {name}: {e}')
        
        failed_count = sum(1 for status in result['connectivity'].values() if status == 'failed')
        if failed_count > 0:
            result['status'] = 'warning'
            self.warnings.append(f'Network connectivity issues detected ({failed_count}/{len(urls)} sites failed)')
        
        return result
    
    def check_configuration(self) -> Dict:
        """Check configuration files"""
        result = {'status': 'ok', 'config_files': {}, 'issues': []}
        
        if not self.args.check_config:
            return result
        
        config_files = [
            ('.env', 'Environment configuration'),
            ('config.yaml', 'YAML configuration'),
            ('frontend/.env.local', 'Frontend environment')
        ]
        
        for file_path, description in config_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                result['config_files'][file_path] = 'exists'
            else:
                result['config_files'][file_path] = 'missing'
                if '.env' in file_path:
                    result['status'] = 'warning'
                    result['issues'].append(f'{description} missing')
                    self.warnings.append(f'{description} file not found')
        
        return result
    
    def apply_fixes(self) -> bool:
        """Attempt to fix common issues"""
        if not self.args.fix:
            return False
        
        logger.info("Attempting to fix common issues...")
        
        fixes_applied = []
        
        # Create .env from template
        env_template = self.project_root / '.env.example'
        env_file = self.project_root / '.env'
        
        if env_template.exists() and not env_file.exists():
            shutil.copy2(env_template, env_file)
            fixes_applied.append('Created .env from template')
            self.fixes_applied.append('Created .env file from .env.example')
        
        # Create frontend .env.local
        frontend_env_example = self.project_root / 'frontend' / '.env.example'
        frontend_env_local = self.project_root / 'frontend' / '.env.local'
        
        if frontend_env_example.exists() and not frontend_env_local.exists():
            shutil.copy2(frontend_env_example, frontend_env_local)
            fixes_applied.append('Created frontend .env.local from template')
            self.fixes_applied.append('Created frontend .env.local file from .env.example')
        
        # Install frontend dependencies if missing
        frontend_dir = self.project_root / 'frontend'
        node_modules_dir = frontend_dir / 'node_modules'
        
        if frontend_dir.exists() and not node_modules_dir.exists():
            try:
                os.chdir(frontend_dir)
                success, _, _ = self.run_command('npm install')
                if success:
                    fixes_applied.append('Installed frontend dependencies')
                    self.fixes_applied.append('Ran npm install in frontend directory')
            except Exception as e:
                logger.warning(f"Failed to install frontend dependencies: {e}")
            finally:
                os.chdir(self.project_root)
        
        logger.info(f"Applied {len(fixes_applied)} fixes")
        return len(fixes_applied) > 0
    
    def print_results(self, results: Dict):
        """Print validation results"""
        print("\n" + "="*60)
        print("🔍 FlavorSnap Environment Validation Results")
        print("="*60)
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Platform: {self.platform.title()}")
        print(f"   Issues: {len(self.issues)}")
        print(f"   Warnings: {len(self.warnings)}")
        
        if self.fixes_applied:
            print(f"   Fixes Applied: {len(self.fixes_applied)}")
        
        # Detailed results
        sections = [
            ('Python', results.get('python')),
            ('Node.js', results.get('node')),
            ('Git', results.get('git')),
            ('Docker', results.get('docker')),
            ('System Resources', results.get('system_resources')),
            ('Project Structure', results.get('project_structure')),
            ('Python Packages', results.get('python_packages')),
            ('Node Packages', results.get('node_packages')),
            ('GPU Support', results.get('gpu_support')),
            ('Network Connectivity', results.get('network_connectivity')),
            ('Configuration', results.get('configuration'))
        ]
        
        for section_name, section_result in sections:
            if section_result:
                self._print_section(section_name, section_result)
        
        # Issues and warnings
        if self.issues:
            print(f"\n❌ Issues ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if self.fixes_applied:
            print(f"\n🔧 Fixes Applied ({len(self.fixes_applied)}):")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"   {i}. {fix}")
        
        # Overall status
        print(f"\n🎯 Overall Status: ", end="")
        
        if len(self.issues) == 0:
            if len(self.warnings) == 0:
                print("✅ Excellent - Ready to go!")
            else:
                print("✅ Good - Minor warnings only")
        elif len(self.issues) <= 2:
            print("⚠️  Needs Attention - Fix issues before proceeding")
        else:
            print("❌ Not Ready - Multiple issues need to be resolved")
        
        print("="*60)
    
    def _print_section(self, name: str, result: Dict):
        """Print a section result"""
        status_icons = {
            'excellent': '✅',
            'ok': '✅',
            'warning': '⚠️',
            'error': '❌',
            'unknown': '❓'
        }
        
        status = result.get('status', 'unknown')
        icon = status_icons.get(status, '❓')
        
        print(f"\n{icon} {name}: {status.title()}")
        
        if 'version' in result and result['version']:
            print(f"   Version: {result['version']}")
        
        if 'npm_version' in result and result['npm_version']:
            print(f"   npm: {result['npm_version']}")
        
        if 'ram_mb' in result and result['ram_mb']:
            print(f"   RAM: {result['ram_mb']}MB")
        
        if 'storage_mb' in result and result['storage_mb']:
            print(f"   Storage: {result['storage_mb']}MB free")
        
        if 'missing_files' in result and result['missing_files']:
            print(f"   Missing files: {', '.join(result['missing_files'])}")
        
        if 'missing_packages' in result and result['missing_packages']:
            print(f"   Missing packages: {', '.join(result['missing_packages'])}")
        
        if 'packages' in result and result['packages']:
            print(f"   Packages: {', '.join(result['packages'].keys())}")
        
        if 'issues' in result and result['issues']:
            for issue in result['issues']:
                print(f"   ⚠️  {issue}")
    
    def run(self) -> bool:
        """Run the validation"""
        logger.info("Starting FlavorSnap environment validation...")
        
        # Apply fixes if requested
        if self.args.fix:
            self.apply_fixes()
        
        # Run all checks
        results = {
            'python': self.check_python(),
            'node': self.check_node(),
            'git': self.check_git(),
            'docker': self.check_docker(),
            'system_resources': self.check_system_resources(),
            'project_structure': self.check_project_structure(),
            'python_packages': self.check_python_packages(),
            'node_packages': self.check_node_packages(),
            'gpu_support': self.check_gpu_support(),
            'network_connectivity': self.check_network_connectivity(),
            'configuration': self.check_configuration()
        }
        
        # Print results
        self.print_results(results)
        
        # Return success if no critical issues
        return len(self.issues) == 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='FlavorSnap Environment Validation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_environment.py                    # Basic validation
  python scripts/check_environment.py --verbose         # Verbose output
  python scripts/check_environment.py --fix             # Fix common issues
  python scripts/check_environment.py --check-gpu       # Check GPU support
  python scripts/check_environment.py --check-network  # Check network
  python scripts/check_environment.py --check-config   # Check configuration
  python scripts/check_environment.py --all           # All checks
        """
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to fix common issues'
    )
    
    parser.add_argument(
        '--check-gpu',
        action='store_true',
        help='Check GPU configuration'
    )
    
    parser.add_argument(
        '--check-network',
        action='store_true',
        help='Check network connectivity'
    )
    
    parser.add_argument(
        '--check-config',
        action='store_true',
        help='Validate configuration files'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all checks (GPU, network, config)'
    )
    
    args = parser.parse_args()
    
    # Enable all checks if --all is specified
    if args.all:
        args.check_gpu = True
        args.check_network = True
        args.check_config = True
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validator
    validator = EnvironmentValidator(args)
    success = validator.run()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
