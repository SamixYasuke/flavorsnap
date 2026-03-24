#!/usr/bin/env python3
"""
FlavorSnap Automated Installation Script

This script automates the installation of FlavorSnap with support for:
- Multiple installation methods (Docker, Manual, Auto)
- Platform-specific setup (Windows, macOS, Linux)
- GPU configuration
- Environment validation
- Dependency installation

Usage:
    python scripts/install.py [options]

Options:
    --method METHOD     Installation method: docker, manual, auto (default: auto)
    --environment ENV   Target environment: development, production, test (default: development)
    --with-gpu          Enable GPU support
    --skip-deps         Skip dependency installation
    --verbose           Verbose output
    --help              Show this help message
"""

import argparse
import os
import sys
import subprocess
import platform
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('installation.log')
    ]
)
logger = logging.getLogger(__name__)

class FlavorSnapInstaller:
    """Main installer class for FlavorSnap"""
    
    def __init__(self, args):
        self.args = args
        self.platform = platform.system().lower()
        self.project_root = Path(__file__).parent.parent
        self.config = self.load_config()
        
        # Platform-specific settings
        self.package_managers = {
            'windows': {
                'node': 'choco install nodejs --version=18.17.0',
                'python': 'choco install python --version=3.9.7',
                'git': 'choco install git',
                'docker': 'choco install docker-desktop'
            },
            'darwin': {  # macOS
                'node': 'brew install node@18',
                'python': 'brew install python@3.9',
                'git': 'brew install git',
                'docker': 'brew install --cask docker'
            },
            'linux': {
                'node': 'curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs',
                'python': 'sudo apt install python3.9 python3.9-pip python3.9-venv -y',
                'git': 'sudo apt install git -y',
                'docker': 'curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh'
            }
        }
        
    def load_config(self) -> Dict:
        """Load installation configuration"""
        config_file = self.project_root / 'config.yaml'
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.warning("PyYAML not found, using default configuration")
        
        return {
            'app': {
                'name': 'FlavorSnap',
                'version': '1.0.0'
            },
            'installation': {
                'method': 'auto',
                'environment': 'development',
                'gpu_support': False
            }
        }
    
    def run_command(self, command: str, shell: bool = True, check: bool = True) -> Tuple[bool, str, str]:
        """Run a command and return success, stdout, stderr"""
        try:
            if self.args.verbose:
                logger.info(f"Running: {command}")
            
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                check=check
            )
            
            if self.args.verbose and result.stdout:
                logger.info(f"Output: {result.stdout}")
            
            return True, result.stdout, result.stderr
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {e.stderr}")
            return False, e.stdout, e.stderr
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are installed"""
        dependencies = {
            'node': self.check_command('node --version'),
            'python': self.check_command('python --version') or self.check_command('python3 --version'),
            'git': self.check_command('git --version'),
            'docker': self.check_command('docker --version'),
            'docker-compose': self.check_command('docker-compose --version') or self.check_command('docker compose version')
        }
        
        logger.info("Dependency Check Results:")
        for dep, installed in dependencies.items():
            status = "✅" if installed else "❌"
            logger.info(f"  {status} {dep}")
        
        return dependencies
    
    def check_command(self, command: str) -> bool:
        """Check if a command is available"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def install_dependencies(self) -> bool:
        """Install missing dependencies"""
        if self.args.skip_deps:
            logger.info("Skipping dependency installation")
            return True
        
        dependencies = self.check_dependencies()
        missing = [dep for dep, installed in dependencies.items() if not installed]
        
        if not missing:
            logger.info("All dependencies are already installed")
            return True
        
        logger.info(f"Installing missing dependencies: {missing}")
        
        if self.platform not in self.package_managers:
            logger.error(f"Unsupported platform: {self.platform}")
            return False
        
        # Install package manager if needed
        if self.platform == 'linux':
            self._install_package_manager_linux()
        elif self.platform == 'darwin':
            self._install_package_manager_macos()
        elif self.platform == 'windows':
            self._install_package_manager_windows()
        
        # Install dependencies
        for dep in missing:
            if dep in self.package_managers[self.platform]:
                logger.info(f"Installing {dep}...")
                success, _, _ = self.run_command(
                    self.package_managers[self.platform][dep],
                    check=False
                )
                if not success:
                    logger.warning(f"Failed to install {dep}, please install manually")
        
        return True
    
    def _install_package_manager_linux(self):
        """Install package managers on Linux"""
        # Check for Homebrew on Linux
        if not self.check_command('brew --version'):
            logger.info("Installing Homebrew on Linux...")
            self.run_command('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', check=False)
    
    def _install_package_manager_macos(self):
        """Install package managers on macOS"""
        # Check for Homebrew
        if not self.check_command('brew --version'):
            logger.info("Installing Homebrew on macOS...")
            self.run_command('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', check=False)
    
    def _install_package_manager_windows(self):
        """Install package managers on Windows"""
        # Check for Chocolatey
        if not self.check_command('choco --version'):
            logger.info("Installing Chocolatey on Windows...")
            cmd = 'Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString(\'https://community.chocolatey.org/install.ps1\'))'
            self.run_command(f'powershell -Command "{cmd}"', check=False)
    
    def setup_environment(self) -> bool:
        """Setup environment configuration"""
        logger.info("Setting up environment configuration...")
        
        # Copy environment template
        env_template = self.project_root / '.env.example'
        env_file = self.project_root / '.env'
        
        if env_template.exists() and not env_file.exists():
            shutil.copy2(env_template, env_file)
            logger.info("Created .env file from template")
        
        # Update environment-specific settings
        if env_file.exists():
            self._update_env_file(env_file)
        
        return True
    
    def _update_env_file(self, env_file: Path):
        """Update .env file with installation settings"""
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Update environment
        content = content.replace('NODE_ENV=development', f'NODE_ENV={self.args.environment}')
        
        # Add GPU settings if requested
        if self.args.with_gpu:
            if 'USE_GPU=' not in content:
                content += '\n# GPU Configuration\nUSE_GPU=true\nCUDA_VISIBLE_DEVICES=0\n'
            else:
                content = content.replace('USE_GPU=false', 'USE_GPU=true')
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated .env file for {self.args.environment} environment")
    
    def install_docker(self) -> bool:
        """Install using Docker method"""
        logger.info("Installing with Docker...")
        
        # Check Docker is available
        if not self.check_command('docker --version'):
            logger.error("Docker is not installed. Please install Docker first.")
            return False
        
        # Build and run containers
        docker_script = self.project_root / 'scripts' / 'docker_run.sh'
        
        if docker_script.exists():
            logger.info("Running Docker setup script...")
            cmd = f"./scripts/docker_run.sh -e {self.args.environment} -d"
            success, stdout, stderr = self.run_command(cmd, check=False)
            
            if success:
                logger.info("Docker installation completed successfully")
                return True
            else:
                logger.error("Docker installation failed")
                return False
        else:
            # Manual Docker setup
            logger.info("Running manual Docker setup...")
            return self._manual_docker_setup()
    
    def _manual_docker_setup(self) -> bool:
        """Manual Docker setup when script is not available"""
        try:
            # Build images
            logger.info("Building Docker images...")
            self.run_command("docker-compose build", check=False)
            
            # Start services
            logger.info("Starting Docker services...")
            self.run_command("docker-compose up -d", check=False)
            
            # Wait for services to be ready
            logger.info("Waiting for services to be ready...")
            import time
            time.sleep(30)
            
            # Check services
            success, _, _ = self.run_command("docker-compose ps")
            return success
            
        except Exception as e:
            logger.error(f"Manual Docker setup failed: {e}")
            return False
    
    def install_manual(self) -> bool:
        """Install using manual method"""
        logger.info("Installing manually...")
        
        # Setup frontend
        frontend_success = self._setup_frontend()
        
        # Setup backend
        backend_success = self._setup_backend()
        
        # Setup model
        model_success = self._setup_model()
        
        return frontend_success and backend_success and model_success
    
    def _setup_frontend(self) -> bool:
        """Setup frontend application"""
        logger.info("Setting up frontend...")
        
        frontend_dir = self.project_root / 'frontend'
        if not frontend_dir.exists():
            logger.warning("Frontend directory not found")
            return False
        
        try:
            os.chdir(frontend_dir)
            
            # Install dependencies
            logger.info("Installing frontend dependencies...")
            success, _, _ = self.run_command("npm install")
            if not success:
                return False
            
            # Setup environment
            env_local = frontend_dir / '.env.local'
            env_example = frontend_dir / '.env.example'
            
            if env_example.exists() and not env_local.exists():
                shutil.copy2(env_example, env_local)
                logger.info("Created frontend .env.local file")
            
            return True
            
        except Exception as e:
            logger.error(f"Frontend setup failed: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def _setup_backend(self) -> bool:
        """Setup backend application"""
        logger.info("Setting up backend...")
        
        backend_dir = self.project_root / 'ml-model-api'
        if not backend_dir.exists():
            logger.warning("Backend directory not found")
            return False
        
        try:
            os.chdir(backend_dir)
            
            # Create virtual environment
            logger.info("Creating Python virtual environment...")
            venv_success, _, _ = self.run_command("python -m venv venv", check=False)
            
            if venv_success:
                # Activate virtual environment and install dependencies
                if self.platform == 'windows':
                    activate_cmd = "venv\\Scripts\\activate"
                else:
                    activate_cmd = "source venv/bin/activate"
                
                logger.info("Installing backend dependencies...")
                success, _, _ = self.run_command(f"{activate_cmd} && pip install -r requirements.txt")
                
                if not success:
                    # Fallback: install without requirements.txt
                    logger.info("Installing essential packages...")
                    packages = ["torch", "torchvision", "flask", "flask-cors", "pillow", "numpy", "requests"]
                    for package in packages:
                        self.run_command(f"{activate_cmd} && pip install {package}", check=False)
                
                return True
            else:
                logger.error("Failed to create virtual environment")
                return False
                
        except Exception as e:
            logger.error(f"Backend setup failed: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def _setup_model(self) -> bool:
        """Setup ML model"""
        logger.info("Setting up ML model...")
        
        model_file = self.project_root / 'model.pth'
        if model_file.exists():
            logger.info("Model file already exists")
            return True
        
        # Note: In a real implementation, you might download the model
        # For now, we'll just check if it exists
        logger.warning("Model file not found. Please ensure model.pth is in the project root")
        return False
    
    def install_auto(self) -> bool:
        """Auto-detect best installation method"""
        logger.info("Auto-detecting installation method...")
        
        dependencies = self.check_dependencies()
        
        # Prefer Docker if available
        if dependencies['docker'] and dependencies['docker-compose']:
            logger.info("Docker detected, using Docker installation")
            return self.install_docker()
        
        # Fall back to manual
        logger.info("Docker not available, using manual installation")
        return self.install_manual()
    
    def validate_installation(self) -> bool:
        """Validate the installation"""
        logger.info("Validating installation...")
        
        validation_script = self.project_root / 'scripts' / 'check_environment.py'
        
        if validation_script.exists():
            success, _, _ = self.run_command("python scripts/check_environment.py", check=False)
            return success
        else:
            # Basic validation
            return self._basic_validation()
    
    def _basic_validation(self) -> bool:
        """Basic installation validation"""
        checks = []
        
        # Check frontend
        frontend_dir = self.project_root / 'frontend'
        if frontend_dir.exists():
            node_modules = frontend_dir / 'node_modules'
            checks.append(node_modules.exists())
        
        # Check backend
        backend_dir = self.project_root / 'ml-model-api'
        if backend_dir.exists():
            venv_dir = backend_dir / 'venv'
            checks.append(venv_dir.exists())
        
        # Check configuration
        env_file = self.project_root / '.env'
        checks.append(env_file.exists())
        
        # Check model
        model_file = self.project_root / 'model.pth'
        checks.append(model_file.exists())
        
        passed = sum(checks)
        total = len(checks)
        
        logger.info(f"Validation: {passed}/{total} checks passed")
        
        return passed >= total - 1  # Allow one check to fail
    
    def run(self) -> bool:
        """Run the installation"""
        logger.info(f"Starting FlavorSnap installation...")
        logger.info(f"Platform: {self.platform}")
        logger.info(f"Method: {self.args.method}")
        logger.info(f"Environment: {self.args.environment}")
        logger.info(f"GPU Support: {self.args.with_gpu}")
        
        try:
            # Step 1: Install dependencies
            if not self.install_dependencies():
                logger.error("Dependency installation failed")
                return False
            
            # Step 2: Setup environment
            if not self.setup_environment():
                logger.error("Environment setup failed")
                return False
            
            # Step 3: Install based on method
            installation_success = False
            
            if self.args.method == 'docker':
                installation_success = self.install_docker()
            elif self.args.method == 'manual':
                installation_success = self.install_manual()
            elif self.args.method == 'auto':
                installation_success = self.install_auto()
            else:
                logger.error(f"Unknown installation method: {self.args.method}")
                return False
            
            if not installation_success:
                logger.error("Installation failed")
                return False
            
            # Step 4: Validate installation
            if not self.validate_installation():
                logger.warning("Installation validation failed, but installation may still work")
            
            logger.info("🎉 FlavorSnap installation completed successfully!")
            self._print_next_steps()
            
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def _print_next_steps(self):
        """Print next steps for the user"""
        logger.info("\n" + "="*50)
        logger.info("🚀 Next Steps:")
        logger.info("="*50)
        
        if self.args.method == 'docker':
            logger.info("1. Services are already running with Docker")
            logger.info("2. Frontend: http://localhost:3000")
            logger.info("3. Backend API: http://localhost:5000")
            logger.info("4. Check status: docker-compose ps")
        else:
            logger.info("1. Start the backend:")
            logger.info("   cd ml-model-api")
            if self.platform == 'windows':
                logger.info("   venv\\Scripts\\activate")
            else:
                logger.info("   source venv/bin/activate")
            logger.info("   python app.py")
            logger.info("")
            logger.info("2. Start the frontend (in new terminal):")
            logger.info("   cd frontend")
            logger.info("   npm run dev")
            logger.info("")
            logger.info("3. Access the application:")
            logger.info("   Frontend: http://localhost:3000")
            logger.info("   Backend: http://localhost:5000")
        
        logger.info("")
        logger.info("📚 Documentation: docs/installation.md")
        logger.info("🐛 Issues: https://github.com/olaleyeolajide81-sketch/flavorsnap/issues")
        logger.info("💬 Community: https://t.me/+Tf3Ll4oRiGk5ZTM0")
        logger.info("="*50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='FlavorSnap Automated Installation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/install.py                           # Auto installation
  python scripts/install.py --method docker           # Docker installation
  python scripts/install.py --method manual --with-gpu # Manual with GPU
  python scripts/install.py --environment production  # Production setup
  python scripts/install.py --skip-deps --verbose    # Skip deps, verbose output
        """
    )
    
    parser.add_argument(
        '--method',
        choices=['docker', 'manual', 'auto'],
        default='auto',
        help='Installation method (default: auto)'
    )
    
    parser.add_argument(
        '--environment',
        choices=['development', 'production', 'test'],
        default='development',
        help='Target environment (default: development)'
    )
    
    parser.add_argument(
        '--with-gpu',
        action='store_true',
        help='Enable GPU support'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run installer
    installer = FlavorSnapInstaller(args)
    success = installer.run()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
