#!/usr/bin/env python3
"""
FlavorSnap Project Structure Analyzer

This script analyzes the FlavorSnap project structure and generates comprehensive
reports about file organization, dependencies, and architectural patterns.

Usage:
    python scripts/analyze_structure.py [options]

Options:
    --format FORMAT    Output format: json, yaml, tree (default: tree)
    --output FILE      Output file (default: stdout)
    --depth LEVEL      Maximum directory depth to analyze (default: 5)
    --include-stats    Include file statistics
    --check-deps       Analyze dependencies
    --validate         Validate project structure
    --help             Show this help message
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import subprocess
import re

class ProjectStructureAnalyzer:
    """Analyzes and reports on project structure."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.structure = {}
        self.stats = defaultdict(int)
        self.dependencies = {}
        self.issues = []
        
        # File patterns for different categories
        self.patterns = {
            'frontend': {
                'extensions': ['.tsx', '.ts', '.jsx', '.js'],
                'directories': ['frontend', 'pages', 'components', 'hooks', 'lib'],
                'files': ['package.json', 'next.config.ts', 'tsconfig.json', 'tailwind.config.ts']
            },
            'backend': {
                'extensions': ['.py'],
                'directories': ['ml-model-api', 'api', 'services'],
                'files': ['app.py', 'requirements.txt', 'model_loader.py']
            },
            'contracts': {
                'extensions': ['.rs'],
                'directories': ['contracts', 'flavorsnap-food-registry'],
                'files': ['Cargo.toml']
            },
            'config': {
                'extensions': ['.yaml', '.yml', '.env', '.json'],
                'directories': ['config'],
                'files': ['config.yaml', '.env.example', 'docker-compose.yml']
            },
            'docs': {
                'extensions': ['.md', '.txt'],
                'directories': ['docs'],
                'files': ['README.md', 'CONTRIBUTING.md', 'LICENSE']
            },
            'tests': {
                'extensions': ['.test.ts', '.test.js', '.test.py', '.spec.ts', '.spec.js'],
                'directories': ['__tests__', 'tests', 'test'],
                'files': ['jest.config.js', 'pytest.ini']
            },
            'docker': {
                'files': ['Dockerfile', 'docker-compose.yml', '.dockerignore']
            },
            'git': {
                'files': ['.gitignore', '.gitattributes']
            }
        }
        
    def analyze(self) -> Dict[str, Any]:
        """Perform complete project structure analysis."""
        print(f"🔍 Analyzing project structure at: {self.project_root}")
        
        # Build directory tree
        self.structure['tree'] = self._build_tree(self.project_root)
        
        # Analyze components
        self.structure['components'] = self._analyze_components()
        
        # Collect statistics
        self.structure['statistics'] = self._collect_statistics()
        
        # Analyze dependencies if requested
        if hasattr(self, 'check_deps') and self.check_deps:
            self.structure['dependencies'] = self._analyze_dependencies()
        
        # Validate structure if requested
        if hasattr(self, 'validate') and self.validate:
            self.structure['validation'] = self._validate_structure()
        
        # Generate recommendations
        self.structure['recommendations'] = self._generate_recommendations()
        
        return self.structure
    
    def _build_tree(self, path: Path, max_depth: int = 5, current_depth: int = 0) -> Dict[str, Any]:
        """Build directory tree structure."""
        if current_depth >= max_depth:
            return {}
        
        tree = {
            'name': path.name,
            'path': str(path.relative_to(self.project_root)),
            'type': 'directory' if path.is_dir() else 'file',
            'size': 0,
            'children': []
        }
        
        if path.is_file():
            try:
                tree['size'] = path.stat().st_size
                tree['extension'] = path.suffix.lower()
            except OSError:
                tree['size'] = 0
        
        if path.is_dir():
            try:
                # Sort entries: directories first, then files
                entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                
                for entry in entries:
                    # Skip hidden files and common ignore patterns
                    if entry.name.startswith('.') and entry.name not in ['.gitignore', '.dockerignore']:
                        continue
                    if entry.name in ['node_modules', '__pycache__', '.next', 'dist', 'build']:
                        continue
                    
                    child = self._build_tree(entry, max_depth, current_depth + 1)
                    if child:
                        tree['children'].append(child)
                        tree['size'] += child.get('size', 0)
                        
            except PermissionError:
                tree['error'] = 'Permission denied'
        
        return tree
    
    def _analyze_components(self) -> Dict[str, Any]:
        """Analyze project components by category."""
        components = {}
        
        for category, config in self.patterns.items():
            components[category] = {
                'files': [],
                'directories': [],
                'statistics': {
                    'file_count': 0,
                    'total_size': 0,
                    'extension_counts': Counter()
                }
            }
        
        def categorize_path(path: Path):
            """Categorize a file or directory."""
            relative_path = path.relative_to(self.project_root)
            
            for category, config in self.patterns.items():
                # Check by extension
                if path.is_file() and path.suffix.lower() in config.get('extensions', []):
                    return category
                
                # Check by directory
                for dir_pattern in config.get('directories', []):
                    if dir_pattern in str(relative_path).lower():
                        return category
                
                # Check by filename
                if path.name in config.get('files', []):
                    return category
            
            return 'other'
        
        # Walk through project files
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.next', 'dist', 'build']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = root_path / file
                category = categorize_path(file_path)
                
                if category in components:
                    try:
                        file_info = {
                            'name': file,
                            'path': str(file_path.relative_to(self.project_root)),
                            'size': file_path.stat().st_size,
                            'extension': file_path.suffix.lower()
                        }
                        components[category]['files'].append(file_info)
                        components[category]['statistics']['file_count'] += 1
                        components[category]['statistics']['total_size'] += file_info['size']
                        components[category]['statistics']['extension_counts'][file_info['extension']] += 1
                    except OSError:
                        continue
        
        # Analyze directories
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            for dir_name in dirs:
                if dir_name.startswith('.') or dir_name in ['node_modules', '__pycache__', '.next', 'dist', 'build']:
                    continue
                    
                dir_path = root_path / dir_name
                category = categorize_path(dir_path)
                
                if category in components:
                    components[category]['directories'].append({
                        'name': dir_name,
                        'path': str(dir_path.relative_to(self.project_root))
                    })
        
        return components
    
    def _collect_statistics(self) -> Dict[str, Any]:
        """Collect project-wide statistics."""
        stats = {
            'total_files': 0,
            'total_directories': 0,
            'total_size': 0,
            'file_extensions': Counter(),
            'largest_files': [],
            'programming_languages': Counter(),
            'configuration_files': 0,
            'documentation_files': 0,
            'test_files': 0
        }
        
        files_info = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.next', 'dist', 'build']]
            
            stats['total_directories'] += len(dirs)
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    extension = file_path.suffix.lower()
                    
                    stats['total_files'] += 1
                    stats['total_size'] += file_size
                    stats['file_extensions'][extension] += 1
                    
                    # Track largest files
                    files_info.append({
                        'name': file,
                        'path': str(file_path.relative_to(self.project_root)),
                        'size': file_size
                    })
                    
                    # Categorize files
                    if extension in ['.md', '.txt', '.rst']:
                        stats['documentation_files'] += 1
                    elif 'test' in file.lower() or extension in ['.test.ts', '.test.js', '.test.py']:
                        stats['test_files'] += 1
                    elif extension in ['.yaml', '.yml', '.env', '.json', '.toml', '.ini']:
                        stats['configuration_files'] += 1
                    
                    # Programming languages
                    if extension in ['.py', '.js', '.ts', '.tsx', '.jsx', '.rs']:
                        lang = extension[1:]  # Remove dot
                        stats['programming_languages'][lang] += 1
                        
                except OSError:
                    continue
        
        # Sort and keep top 10 largest files
        stats['largest_files'] = sorted(files_info, key=lambda x: x['size'], reverse=True)[:10]
        
        return stats
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        dependencies = {
            'frontend': {},
            'backend': {},
            'contracts': {}
        }
        
        # Analyze frontend dependencies
        package_json = self.project_root / 'frontend' / 'package.json'
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                    dependencies['frontend'] = {
                        'dependencies': package_data.get('dependencies', {}),
                        'devDependencies': package_data.get('devDependencies', {}),
                        'scripts': package_data.get('scripts', {})
                    }
            except (json.JSONDecodeError, IOError):
                pass
        
        # Analyze backend dependencies
        requirements_txt = self.project_root / 'ml-model-api' / 'requirements.txt'
        if requirements_txt.exists():
            try:
                with open(requirements_txt, 'r') as f:
                    requirements = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            requirements.append(line)
                    dependencies['backend'] = {'requirements': requirements}
            except IOError:
                pass
        
        # Analyze Rust dependencies
        cargo_toml = self.project_root / 'Cargo.toml'
        if cargo_toml.exists():
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    tomllib = None
            
            if tomllib:
                try:
                    with open(cargo_toml, 'rb') as f:
                        cargo_data = tomllib.load(f)
                        dependencies['contracts'] = {
                            'workspace_members': cargo_data.get('workspace', {}).get('members', []),
                            'dependencies': cargo_data.get('dependencies', {})
                        }
                except Exception:
                    pass
        
        return dependencies
    
    def _validate_structure(self) -> Dict[str, Any]:
        """Validate project structure against expected patterns."""
        validation = {
            'errors': [],
            'warnings': [],
            'missing_files': [],
            'unexpected_files': [],
            'structure_score': 0
        }
        
        # Expected critical files
        expected_files = [
            'README.md',
            'model.pth',
            'food_classes.txt',
            'config.yaml',
            '.gitignore',
            '.env.example'
        ]
        
        # Expected directories
        expected_dirs = [
            'frontend',
            'ml-model-api',
            'docs',
            'scripts',
            'contracts',
            'dataset',
            'models',
            'uploads'
        ]
        
        # Check for missing files
        for file_path in expected_files:
            if not (self.project_root / file_path).exists():
                validation['missing_files'].append(file_path)
                validation['errors'].append(f"Missing critical file: {file_path}")
        
        # Check for missing directories
        for dir_path in expected_dirs:
            if not (self.project_root / dir_path).exists():
                validation['missing_files'].append(f"{dir_path}/")
                validation['warnings'].append(f"Missing expected directory: {dir_path}")
        
        # Check frontend structure
        frontend_dir = self.project_root / 'frontend'
        if frontend_dir.exists():
            frontend_files = ['package.json', 'next.config.ts', 'tsconfig.json']
            for file_path in frontend_files:
                if not (frontend_dir / file_path).exists():
                    validation['warnings'].append(f"Missing frontend file: {file_path}")
        
        # Check backend structure
        backend_dir = self.project_root / 'ml-model-api'
        if backend_dir.exists():
            backend_files = ['app.py', 'requirements.txt']
            for file_path in backend_files:
                if not (backend_dir / file_path).exists():
                    validation['warnings'].append(f"Missing backend file: {file_path}")
        
        # Calculate structure score
        total_expected = len(expected_files) + len(expected_dirs)
        total_found = total_expected - len(validation['missing_files'])
        validation['structure_score'] = max(0, (total_found / total_expected) * 100)
        
        return validation
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Check for common issues
        if not (self.project_root / '.github' / 'workflows').exists():
            recommendations.append("Consider adding GitHub Actions workflows for CI/CD")
        
        if not (self.project_root / 'CONTRIBUTING.md').exists():
            recommendations.append("Add CONTRIBUTING.md to guide contributors")
        
        if not (self.project_root / 'LICENSE').exists():
            recommendations.append("Add a LICENSE file to specify usage terms")
        
        # Check for security files
        if not (self.project_root / 'SECURITY.md').exists():
            recommendations.append("Add SECURITY.md for security policy and reporting")
        
        # Check for documentation completeness
        docs_dir = self.project_root / 'docs'
        if docs_dir.exists():
            expected_docs = ['installation.md', 'configuration.md', 'troubleshooting.md']
            for doc in expected_docs:
                if not (docs_dir / doc).exists():
                    recommendations.append(f"Consider adding {doc} to documentation")
        
        # Check for testing
        test_dirs = ['tests', '__tests__', 'test']
        has_tests = any((self.project_root / d).exists() for d in test_dirs)
        if not has_tests:
            recommendations.append("Add test directories and implement test suites")
        
        return recommendations
    
    def format_output(self, data: Dict[str, Any], format_type: str = 'tree') -> str:
        """Format analysis output."""
        if format_type == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format_type == 'yaml':
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        else:
            return self._format_tree(data)
    
    def _format_tree(self, data: Dict[str, Any]) -> str:
        """Format output as readable tree."""
        output = []
        output.append("🏗️ FlavorSnap Project Structure Analysis")
        output.append("=" * 50)
        output.append("")
        
        # Statistics
        if 'statistics' in data:
            stats = data['statistics']
            output.append("📊 Project Statistics")
            output.append("-" * 20)
            output.append(f"Total Files: {stats['total_files']}")
            output.append(f"Total Directories: {stats['total_directories']}")
            output.append(f"Total Size: {self._format_size(stats['total_size'])}")
            output.append(f"Documentation Files: {stats['documentation_files']}")
            output.append(f"Test Files: {stats['test_files']}")
            output.append(f"Configuration Files: {stats['configuration_files']}")
            output.append("")
            
            # Programming languages
            if stats['programming_languages']:
                output.append("💻 Programming Languages")
                output.append("-" * 25)
                for lang, count in stats['programming_languages'].most_common():
                    output.append(f"{lang}: {count} files")
                output.append("")
        
        # Components
        if 'components' in data:
            output.append("🧩 Component Analysis")
            output.append("-" * 20)
            for category, info in data['components'].items():
                if info['statistics']['file_count'] > 0:
                    output.append(f"{category.title()}:")
                    output.append(f"  Files: {info['statistics']['file_count']}")
                    output.append(f"  Size: {self._format_size(info['statistics']['total_size'])}")
                    output.append("")
        
        # Validation
        if 'validation' in data:
            validation = data['validation']
            output.append("✅ Structure Validation")
            output.append("-" * 22)
            output.append(f"Structure Score: {validation['structure_score']:.1f}%")
            
            if validation['errors']:
                output.append("\n❌ Errors:")
                for error in validation['errors']:
                    output.append(f"  - {error}")
            
            if validation['warnings']:
                output.append("\n⚠️ Warnings:")
                for warning in validation['warnings']:
                    output.append(f"  - {warning}")
            output.append("")
        
        # Recommendations
        if 'recommendations' in data and data['recommendations']:
            output.append("💡 Recommendations")
            output.append("-" * 18)
            for rec in data['recommendations']:
                output.append(f"• {rec}")
            output.append("")
        
        # Directory tree
        if 'tree' in data:
            output.append("🌳 Directory Tree")
            output.append("-" * 16)
            output.append(self._format_tree_recursive(data['tree']))
        
        return "\n".join(output)
    
    def _format_tree_recursive(self, node: Dict[str, Any], prefix: str = "", is_last: bool = True) -> str:
        """Recursively format tree structure."""
        if not node:
            return ""
        
        output = []
        
        # Current node
        connector = "└── " if is_last else "├── "
        size_info = f" ({self._format_size(node['size'])})" if node.get('size', 0) > 0 else ""
        output.append(f"{prefix}{connector}{node['name']}{size_info}")
        
        # Children
        if node.get('children'):
            children = node['children']
            for i, child in enumerate(children):
                child_prefix = prefix + ("    " if is_last else "│   ")
                child_is_last = (i == len(children) - 1)
                output.append(self._format_tree_recursive(child, child_prefix, child_is_last))
        
        return "\n".join(output)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze FlavorSnap project structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_structure.py
  python scripts/analyze_structure.py --format json --output structure.json
  python scripts/analyze_structure.py --depth 3 --include-stats --check-deps
  python scripts/analyze_structure.py --validate --format yaml
        """
    )
    
    parser.add_argument(
        '--format',
        choices=['tree', 'json', 'yaml'],
        default='tree',
        help='Output format (default: tree)'
    )
    
    parser.add_argument(
        '--output',
        help='Output file (default: stdout)'
    )
    
    parser.add_argument(
        '--depth',
        type=int,
        default=5,
        help='Maximum directory depth (default: 5)'
    )
    
    parser.add_argument(
        '--include-stats',
        action='store_true',
        help='Include detailed statistics'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Analyze project dependencies'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate project structure'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ProjectStructureAnalyzer()
    analyzer.check_deps = args.check_deps
    analyzer.validate = args.validate
    
    # Perform analysis
    try:
        result = analyzer.analyze()
        
        # Format output
        formatted_output = analyzer.format_output(result, args.format)
        
        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
            print(f"✅ Analysis saved to: {args.output}")
        else:
            print(formatted_output)
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
