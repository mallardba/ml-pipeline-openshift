# Development Requirements Documentation

This file explains each dependency in `requirements-dev.txt` and why it's needed for development of the MLOps pipeline project.

## Production Dependencies
### -r requirements.txt
**What it is**: Includes all production dependencies
**Why we need it**: Developers need both production code AND development tools to work effectively

## Testing Framework

### pytest>=7.4.0
**What it is**: Testing framework for Python
**Why we need it**:
- Write and run unit tests
- Test discovery and execution
- Fixture management
- Plugin ecosystem
**Used for**: Unit testing, integration testing

### pytest-cov>=4.1.0
**What it is**: Coverage plugin for pytest
**Why we need it**:
- Measure code coverage
- Identify untested code
- Generate coverage reports
- Quality assurance
**Used for**: Code coverage analysis

### pytest-mock>=3.11.0
**What it is**: Mocking utilities for pytest
**Why we need it**:
- Mock external dependencies
- Isolate units under test
- Control test environment
- Faster test execution
**Used for**: Mocking external services, isolated testing

### pytest-asyncio>=0.21.0
**What it is**: Async testing support for pytest
**Why we need it**:
- Test async/await code
- FastAPI endpoint testing
- Async model inference testing
- Concurrent operation testing
**Used for**: Testing async code, FastAPI endpoints

### httpx>=0.24.0
**What it is**: Modern HTTP client for testing
**Why we need it**:
- Test FastAPI endpoints
- Mock HTTP requests
- Async HTTP testing
- API integration testing
**Used for**: API testing, HTTP client testing

## Code Quality & Linting

### ruff>=0.0.280
**What it is**: Fast Python linter written in Rust
**Why we need it**:
- Replaces flake8, isort, pylint
- 10-100x faster than traditional tools
- Catches bugs and style issues
- Auto-fixes many problems
**Used for**: Code linting, import sorting, style checking

### black>=23.7.0
**What it is**: Uncompromising Python code formatter
**Why we need it**:
- Consistent code formatting
- Industry standard formatter
- Eliminates formatting debates
- Automatic code style
**Used for**: Code formatting, style consistency

### mypy>=1.5.0
**What it is**: Static type checker for Python
**Why we need it**:
- Catch type-related bugs
- Improve code documentation
- Better IDE support
- Refactoring safety
**Used for**: Type checking, static analysis

### pre-commit>=3.3.0
**What it is**: Git hooks framework
**Why we need it**:
- Run checks before commits
- Ensure code quality
- Prevent bad code from entering repo
- Team consistency
**Used for**: Git hooks, automated quality checks

## Documentation

### mkdocs>=1.5.0
**What it is**: Static site generator for documentation
**Why we need it**:
- Generate project documentation
- Markdown-based docs
- Easy to maintain
- Professional documentation
**Used for**: Project documentation, API docs

### mkdocs-material>=9.2.0
**What it is**: Material Design theme for MkDocs
**Why we need it**:
- Beautiful documentation theme
- Responsive design
- Search functionality
- Modern UI/UX
**Used for**: Documentation styling, user experience

### mkdocstrings[python]>=0.22.0
**What it is**: Automatic API documentation from docstrings
**Why we need it**:
- Auto-generate API docs
- Keep docs in sync with code
- Reduce documentation maintenance
- Professional API reference
**Used for**: API documentation, docstring rendering

## Development Tools

### jupyter>=1.0.0
**What it is**: Jupyter Notebook interface
**Why we need it**:
- Interactive data exploration
- Prototype ML models
- Shareable notebooks
- Data analysis
**Used for**: Data exploration, model prototyping

### jupyterlab>=4.0.0
**What it is**: Next-generation Jupyter interface
**Why we need it**:
- Modern notebook interface
- File browser integration
- Terminal access
- Extension ecosystem
**Used for**: Enhanced notebook development

### ipykernel>=6.25.0
**What it is**: IPython kernel for Jupyter
**Why we need it**:
- Python kernel for notebooks
- Rich output display
- Magic commands
- Interactive computing
**Used for**: Notebook kernel, interactive Python

### notebook>=7.0.0
**What it is**: Classic Jupyter Notebook
**Why we need it**:
- Traditional notebook interface
- Compatibility with older notebooks
- Alternative to JupyterLab
- Legacy support
**Used for**: Classic notebook interface

## Debugging & Profiling

### pdbpp>=0.10.0
**What it is**: Enhanced Python debugger
**Why we need it**:
- Better debugging experience
- Syntax highlighting
- Improved pdb interface
- Development debugging
**Used for**: Code debugging, development

### memory-profiler>=0.61.0
**What it is**: Memory usage profiler
**Why we need it**:
- Identify memory leaks
- Optimize memory usage
- Profile ML model memory
- Performance optimization
**Used for**: Memory profiling, optimization

### line-profiler>=4.1.0
**What it is**: Line-by-line profiler
**Why we need it**:
- Identify performance bottlenecks
- Optimize critical code paths
- Profile ML training loops
- Performance analysis
**Used for**: Performance profiling, optimization

## Kubernetes/OpenShift Development

### kubernetes>=27.2.0
**What it is**: Kubernetes Python client
**Why we need it**:
- Interact with Kubernetes API
- Deploy and manage resources
- Monitor cluster state
- Automation scripts
**Used for**: K8s automation, cluster management

### openshift>=0.13.0
**What it is**: OpenShift Python client
**Why we need it**:
- OpenShift-specific operations
- Route management
- OpenShift AI integration
- Red Hat ecosystem
**Used for**: OpenShift automation, RHOAI integration

## Additional ML Development Tools

### wandb>=0.15.0
**What it is**: Weights & Biases experiment tracking
**Why we need it**:
- Experiment tracking and visualization
- Model performance monitoring
- Hyperparameter optimization
- Team collaboration
**Used for**: Experiment tracking, model monitoring

### tensorboard>=2.13.0
**What it is**: TensorBoard visualization toolkit
**Why we need it**:
- Training visualization
- Model graph visualization
- Metrics plotting
- Debugging training
**Used for**: Training visualization, model debugging

### optuna>=3.3.0
**What it is**: Hyperparameter optimization framework
**Why we need it**:
- Automated hyperparameter tuning
- Bayesian optimization
- Multi-objective optimization
- Efficient search strategies
**Used for**: Hyperparameter tuning, optimization

## Data Visualization

### matplotlib>=3.7.0
**What it is**: Python plotting library
**Why we need it**:
- Create plots and visualizations
- Data exploration
- Model performance plots
- Publication-quality figures
**Used for**: Data visualization, plotting

### seaborn>=3.12.0
**What it is**: Statistical data visualization
**Why we need it**:
- Statistical plots
- Beautiful default styles
- Data distribution analysis
- Correlation matrices
**Used for**: Statistical visualization, data analysis

### plotly>=5.15.0
**What it is**: Interactive plotting library
**Why we need it**:
- Interactive visualizations
- Web-based plots
- Dashboard creation
- Real-time updates
**Used for**: Interactive plots, dashboards

## Type Stubs

### types-requests>=2.31.0
**What it is**: Type stubs for requests library
**Why we need it**:
- Better IDE support
- Type checking for requests
- Improved development experience
- Static analysis
**Used for**: Type checking, IDE support

### types-PyYAML>=6.0.0
**What it is**: Type stubs for PyYAML
**Why we need it**:
- Type checking for YAML operations
- Better IDE support
- Static analysis
- Development experience
**Used for**: Type checking, IDE support

### types-click>=8.0.0
**What it is**: Type stubs for Click
**Why we need it**:
- Type checking for CLI code
- Better IDE support
- Static analysis
- Development experience
**Used for**: Type checking, IDE support
