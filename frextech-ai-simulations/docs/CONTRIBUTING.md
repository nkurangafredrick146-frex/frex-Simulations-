
Contributing to FrexTech AI Simulations

Thank you for your interest in contributing to FrexTech AI Simulations! This document provides guidelines and instructions for contributing to the project. Whether you're fixing bugs, adding features, improving documentation, or suggesting ideas, we appreciate your help.

Table of Contents

1. Code of Conduct
2. Getting Started
3. Development Workflow
4. Code Style Guidelines
5. Testing Guidelines
6. Documentation Standards
7. Pull Request Process
8. Project Structure
9. Issue Guidelines
10. Community & Support

Code of Conduct

Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

Our Standards

Examples of behavior that contributes to a positive environment:

· Using welcoming and inclusive language
· Being respectful of differing viewpoints and experiences
· Gracefully accepting constructive criticism
· Focusing on what is best for the community
· Showing empathy towards other community members

Examples of unacceptable behavior:

· The use of sexualized language or imagery and unwelcome sexual attention or advances
· Trolling, insulting or derogatory comments, and personal or political attacks
· Public or private harassment
· Publishing others' private information without explicit permission
· Other conduct which could reasonably be considered inappropriate in a professional setting

Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@frextech-sim.com. All complaints will be reviewed and investigated promptly and fairly.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned with this Code of Conduct.

Getting Started

Prerequisites

· Python: 3.9 or higher
· Git: 2.20 or higher
· Docker: 20.10 or higher (optional, for containerized development)
· CUDA: 11.7 or higher (for GPU development)
· Node.js: 18 or higher (for frontend development)

Initial Setup

1. Fork the Repository
   ```bash
   # Click the 'Fork' button on GitHub
   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/frextech-ai-simulations.git
   cd frextech-ai-simulations
   ```
2. Set Up Development Environment
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```
3. Configure Environment Variables
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your configuration
   # Add API keys, database URLs, etc.
   ```
4. Set Up Git Hooks
   ```bash
   # Install pre-commit hooks
   pre-commit install
   
   # Install commit-msg hook
   pre-commit install --hook-type commit-msg
   ```

Development Modes

Local Development (Recommended)

```bash
# Install all dependencies
pip install -e ".[dev,api,training,gpu]"

# Run development server
python -m src.api.server

# Run training locally (CPU mode)
python scripts/training/train_world_model.py --config configs/model/base.yaml
```

Docker Development

```bash
# Build development container
docker build -f docker/Dockerfile.dev -t frextech-dev .

# Run with GPU support
docker run --gpus all -it -p 8000:8000 -v $(pwd):/app frextech-dev

# Or use docker-compose
docker-compose -f docker-compose.dev.yml up
```

Remote Development (VS Code)

1. Open the repository in VS Code
2. Install the Remote Development extension pack
3. Connect to a remote container or SSH host
4. Use the provided .devcontainer configuration

Development Workflow

Branch Strategy

We use a modified Git Flow workflow:

· main: Production-ready code
· develop: Integration branch for features
· feature/*: New features
· bugfix/*: Bug fixes
· release/*: Release preparation
· hotfix/*: Emergency fixes

Creating a New Feature

1. Create a Feature Branch
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/amazing-feature
   ```
2. Make Your Changes
   ```bash
   # Make atomic commits with descriptive messages
   git add .
   git commit -m "feat: add amazing feature
   
   - Implement core functionality
   - Add unit tests
   - Update documentation"
   ```
3. Keep Your Branch Updated
   ```bash
   # Rebase on develop regularly
   git fetch origin
   git rebase origin/develop
   ```
4. Push Your Changes
   ```bash
   git push -u origin feature/amazing-feature
   ```

Commit Message Format

We follow the Conventional Commits specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:

· feat: New feature
· fix: Bug fix
· docs: Documentation changes
· style: Code style changes (formatting, etc.)
· refactor: Code refactoring
· perf: Performance improvements
· test: Adding or updating tests
· build: Build system or dependency changes
· ci: CI configuration changes
· chore: Other changes that don't modify src or test files

Examples:

```
feat(world-model): add diffusion transformer blocks

- Implement multi-head cross attention
- Add gradient checkpointing
- Update training configuration

Closes #123
```

```
fix(api): handle malformed request bodies

- Add validation for JSON payloads
- Return proper HTTP 400 errors
- Add test cases for edge cases

Fixes #456
```

Pre-commit Hooks

We use pre-commit hooks to maintain code quality automatically:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: check-case-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
  
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.5.3
    hooks:
      - id: commitizen
        stages: [commit-msg]
```

Code Style Guidelines

Python Code Style

We follow PEP 8 with the following modifications:

1. Line Length: 100 characters
2. Imports Order:
   ```python
   # Standard library imports
   import os
   import sys
   from typing import Dict, List, Optional
   
   # Third-party imports
   import numpy as np
   import torch
   import torch.nn as nn
   
   # Local imports
   from src.core.world_model import WorldModel
   from .utils import helper_function
   ```
3. Type Hints: Mandatory for all functions and methods
   ```python
   def process_data(
       data: np.ndarray,
       config: Dict[str, Any],
       batch_size: int = 32
   ) -> Tuple[torch.Tensor, torch.Tensor]:
       """Process input data with given configuration."""
       # Implementation
   ```
4. Documentation Strings: Use Google style
   ```python
   def generate_world(
       prompt: str,
       quality: str = "standard"
   ) -> World:
       """Generate a 3D world from a text prompt.
       
       Args:
           prompt: Text description of the world
           quality: Quality level ("draft", "standard", "premium")
           
       Returns:
           World object containing the generated scene
           
       Raises:
           ValueError: If quality is not one of the allowed values
           RuntimeError: If generation fails
       """
   ```

Naming Conventions

· Variables: snake_case
· Functions/Methods: snake_case()
· Classes: PascalCase
· Constants: UPPER_SNAKE_CASE
· Private: _private_method or __really_private
· Type Variables: T, U, VT, KT

File Organization

1. Module Structure:
   ```
   module_name/
   ├── __init__.py          # Public API exports
   ├── core.py              # Main functionality
   ├── utils.py             # Helper functions
   ├── exceptions.py        # Custom exceptions
   └── tests/               # Test files
   ```
2. Class Structure:
   ```python
   class WorldModel(nn.Module):
       """High-level description of the class."""
       
       # Class constants
       DEFAULT_CONFIG = {...}
       
       def __init__(self, config: Dict[str, Any]):
           """Initialize the model.
           
           Args:
               config: Model configuration dictionary
           """
           super().__init__()
           self.config = config
           self._setup_layers()
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass.
           
           Args:
               x: Input tensor of shape (batch, channels, height, width)
               
           Returns:
               Output tensor
           """
           return self._process(x)
       
       def _setup_layers(self) -> None:
           """Setup neural network layers."""
           # Private method implementation
   ```

Error Handling

1. Use Specific Exceptions:
   ```python
   # Good
   raise ValueError("Quality must be 'draft', 'standard', or 'premium'")
   
   # Bad
   raise Exception("Invalid quality")
   ```
2. Custom Exceptions:
   ```python
   class GenerationError(RuntimeError):
       """Raised when world generation fails."""
       
       def __init__(self, message: str, error_code: str):
           super().__init__(message)
           self.error_code = error_code
   ```
3. Error Recovery:
   ```python
   try:
       result = generate_world(prompt)
   except ValueError as e:
       logger.warning(f"Invalid input: {e}")
       return None
   except GenerationError as e:
       logger.error(f"Generation failed: {e.error_code}")
       raise
   except Exception as e:
       logger.exception("Unexpected error")
       raise RuntimeError("Internal error") from e
   ```

Testing Guidelines

Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_world_model.py
│   └── test_encoders.py
├── integration/            # Integration tests
│   ├── test_api_endpoints.py
│   └── test_full_pipeline.py
├── performance/            # Performance tests
│   ├── benchmark_inference.py
│   └── load_testing.py
└── fixtures/               # Test data
    ├── sample_images/
    └── test_worlds/
```

Writing Tests

```python
import pytest
import torch
from src.core.world_model import WorldModel

class TestWorldModel:
    """Test suite for WorldModel."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        config = {
            "latent_dim": 256,
            "num_layers": 4
        }
        return WorldModel(config)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 256, 256)
    
    def test_forward_pass(self, model, sample_input):
        """Test forward pass produces correct output shape."""
        output = model(sample_input)
        
        assert output.shape == (2, 256, 32, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_batch_independence(self, model, sample_input):
        """Test that batches are processed independently."""
        batch1 = sample_input[0:1]
        batch2 = sample_input[1:2]
        
        output1 = model(batch1)
        output2 = model(batch2)
        output_full = model(sample_input)
        
        # Check batch independence
        torch.testing.assert_close(output_full[0:1], output1)
        torch.testing.assert_close(output_full[1:2], output2)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_various_batch_sizes(self, model, batch_size):
        """Test model works with various batch sizes."""
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        output = model(input_tensor)
        
        assert output.shape[0] == batch_size
    
    @pytest.mark.slow
    def test_training_step(self, model):
        """Test complete training step."""
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.MSELoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        input_tensor = torch.randn(4, 3, 256, 256)
        target = torch.randn(4, 256, 32, 32)
        
        output = model(input_tensor)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
    
    @pytest.mark.gpu
    def test_gpu_support(self, model):
        """Test model works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        model = model.cuda()
        input_tensor = torch.randn(2, 3, 256, 256).cuda()
        
        output = model(input_tensor)
        assert output.is_cuda
```

Test Categories

Mark Description
@pytest.mark.unit Unit tests (default)
@pytest.mark.integration Integration tests
@pytest.mark.slow Tests that take >1 second
@pytest.mark.gpu Requires GPU
@pytest.mark.heavy Requires significant resources
@pytest.mark.flaky Tests that sometimes fail

Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m integration          # Run integration tests
pytest tests/unit/            # Run unit tests only

# With coverage report
pytest --cov=src --cov-report=html

# Parallel execution
pytest -n auto

# Debugging
pytest -v --tb=short          # Verbose with short traceback
pytest --pdb                  # Enter debugger on failure
```

Test Data Management

```python
# tests/fixtures/__init__.py
import pytest
import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent

@pytest.fixture
def sample_world_config():
    """Load sample world configuration."""
    config_path = FIXTURE_DIR / "configs" / "sample_world.json"
    with open(config_path) as f:
        return json.load(f)

@pytest.fixture
def sample_image():
    """Load sample test image."""
    image_path = FIXTURE_DIR / "sample_images" / "test.jpg"
    return load_image(image_path)
```

Documentation Standards

Code Documentation

1. Module Docstrings:
   ```python
   """World Model module.
   
   This module contains the core world generation model based on
   diffusion transformers and Gaussian splatting.
   
   Classes:
       WorldModel: Main world generation model
       DiffusionBlock: Diffusion transformer block
       GaussianDecoder: Gaussian splatting decoder
   
   Functions:
       generate_world: High-level generation function
       load_pretrained: Load pretrained model weights
   """
   ```
2. Class Docstrings:
   ```python
   class WorldModel(nn.Module):
       """Neural network for 3D world generation.
       
       This model takes multimodal inputs (text, images, videos) and
       generates 3D world representations using diffusion transformers
       and Gaussian splatting.
       
       Attributes:
           config: Model configuration dictionary
           text_encoder: CLIP text encoder
           diffusion_model: Latent diffusion transformer
           gaussian_decoder: Gaussian splatting decoder
       """
   ```
3. Function/Method Docstrings:
   ```python
   def generate_world(prompt: str, **kwargs) -> World:
       """Generate a 3D world from text description.
       
       This is the main entry point for world generation. It handles
       the complete pipeline from text encoding to 3D representation.
       
       Args:
           prompt: Text description of the desired world
           **kwargs: Additional generation parameters:
               quality: Generation quality ("draft", "standard", "premium")
               seed: Random seed for reproducibility
               format: Output format ("gaussian", "nerf", "mesh")
               
       Returns:
           World object containing the generated 3D scene
           
       Raises:
           ValueError: If prompt is empty or contains invalid content
           GenerationError: If generation fails due to model errors
           
       Example:
           >>> world = generate_world("A mountain lake at sunset")
           >>> world.export("lake.glb")
       """
   ```

API Documentation

1. OpenAPI/Swagger: All API endpoints must have OpenAPI annotations
   ```python
   @router.post("/generate/text", response_model=GenerationResponse)
   async def generate_from_text(
       request: GenerationRequest,
       api_key: APIKey = Depends(get_api_key)
   ) -> GenerationResponse:
       """Generate 3D world from text prompt.
       
       This endpoint accepts a text description and generates a
       corresponding 3D world using our AI model.
       
       Parameters:
           - request: Generation request containing prompt and parameters
           - api_key: Valid API key for authentication
           
       Returns:
           Generation response with job ID and status
           
       Errors:
           400: Invalid request parameters
           401: Invalid or missing API key
           429: Rate limit exceeded
           500: Internal server error
       """
   ```
2. Example Code: Include examples in documentation
   ```python
   """
   Examples:
       Basic usage:
       ```python
       from frextech import FrexTechClient
       
       client = FrexTechClient(api_key="your_key")
       job = client.generate.from_text("A forest")
       world = job.wait_for_completion()
       ```
       
       Advanced usage:
       ```python
       job = client.generate.from_text(
           prompt="Futuristic city",
           quality="premium",
           seed=42,
           metadata={"style": "cyberpunk"}
       )
       ```
   """
   ```

README Files

Each major directory should have a README.md:

```markdown
# Module Name

## Purpose
Brief description of what this module does and why it exists.

## Key Features
- Feature 1
- Feature 2
- Feature 3

## Usage
```python
# Example code
from module import main_function

result = main_function(input)
```

API Reference

Classes

· ClassName: Description

Functions

· function_name(): Description

Development

Instructions for developers working on this module.

See Also

· Related Module
· API Documentation

```

## Pull Request Process

### Before Submitting

1. **Ensure Code Quality**:
   ```bash
   # Run all pre-commit checks
   pre-commit run --all-files
   
   # Run tests
   pytest
   
   # Check type hints
   mypy src/
   
   # Verify documentation builds
   mkdocs build
```

1. Update Documentation:
   · Update docstrings for new/changed code
   · Update API documentation if needed
   · Add/update examples
   · Update README files if applicable
2. Create Meaningful Commits:
   ```bash
   # Interactive rebase to squash/fixup commits
   git rebase -i origin/develop
   
   # Write good commit messages
   git commit -m "feat(module): add new feature
   
   - Implement core functionality
   - Add comprehensive tests
   - Update documentation
   
   Closes #123"
   ```

PR Checklist

· Code follows the style guidelines
· Tests added/updated and passing
· Documentation updated
· Type hints added for new code
· No new warnings or errors
· Commit messages follow convention
· Branch is up-to-date with develop
· PR description explains changes clearly

PR Template

```markdown
## Description
Brief description of the changes in this PR.

## Related Issues
Closes #123
Fixes #456

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe the tests you ran to verify your changes.

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Screenshots/Logs
If applicable, add screenshots or logs to help explain your changes.

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

Code Review Process

1. Automated Checks:
   · CI pipeline runs tests and checks
   · Code coverage requirements must be met
   · Security scans pass
2. Reviewer Guidelines:
   · Check for correctness and edge cases
   · Ensure code is maintainable and readable
   · Verify documentation is complete
   · Check for performance implications
   · Ensure security best practices
3. Review Comments:
   ```markdown
   ## General Comments
   Overall, this looks good! Just a few suggestions:
   
   ## Specific Suggestions
   ### File: src/core/world_model.py
   - Line 45: Consider using `torch.no_grad()` here for memory efficiency
   - Line 89: Could we add a type hint for the return value?
   
   ### File: tests/test_world_model.py
   - Line 123: Add a test for the edge case when batch_size=0
   
   ## Questions
   - Why was this approach chosen over alternative X?
   - Are there any performance implications for large inputs?
   ```

After Approval

1. Squash and Merge:
   ```bash
   git checkout develop
   git pull origin develop
   git merge --squash feature/amazing-feature
   git commit -m "feat: add amazing feature (#789)"
   git push origin develop
   ```
2. Delete Branch:
   ```bash
   git branch -d feature/amazing-feature
   git push origin --delete feature/amazing-feature
   ```
3. Create Release Notes:
   Update CHANGELOG.md with the changes

Project Structure

Key Directories

```
frextech-ai-simulations/
├── src/                    # Source code
│   ├── core/              # Core AI models
│   ├── api/               # FastAPI server
│   ├── interactive/       # Interactive editing
│   ├── render/            # Rendering engines
│   └── utils/             # Utilities
├── tests/                 # Test files
├── docs/                  # Documentation
├── examples/              # Example code
├── configs/               # Configuration files
├── scripts/               # Utility scripts
└── docker/                # Docker configurations
```

Adding New Modules

1. Create Module Structure:
   ```bash
   mkdir -p src/new_module/{submodule1,submodule2}
   touch src/new_module/__init__.py
   touch src/new_module/submodule1/__init__.py
   touch src/new_module/submodule1/core.py
   ```
2. Update Package Configuration:
   ```toml
   # pyproject.toml
   [project]
   # ...
   
   [project.optional-dependencies]
   new-module = [
       "new_dependency>=1.0.0",
   ]
   ```
3. Add to Documentation:
   ```bash
   # Add module documentation
   mkdir -p docs/modules/new_module
   touch docs/modules/new_module/index.md
   ```

Dependencies Management

```toml
# pyproject.toml example
[project]
name = "frextech-ai-simulations"
version = "0.1.0"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "fastapi>=0.100.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
api = [
    "uvicorn[standard]>=0.23.0",
    "python-multipart>=0.0.6",
]
gpu = [
    "torch>=2.0.0; platform_system != 'Darwin'",
    "nvidia-cuda-runtime-cu11>=11.7.0; platform_system != 'Darwin'",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Issue Guidelines

Creating Issues

Use the provided templates for bug reports and feature requests:

Bug Report Template:

```markdown
## Description
Clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Screenshots/Logs
If applicable, add screenshots or logs.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.0]
- Package Version: [e.g., 0.1.0]
- GPU: [e.g., NVIDIA RTX 4090]

## Additional Context
Add any other context about the problem.
```

Feature Request Template:

```markdown
## Problem Statement
Clear and concise description of the problem.

## Proposed Solution
Description of the solution you'd like.

## Alternative Solutions
Description of alternative solutions you've considered.

## Additional Context
Add any other context or screenshots about the feature request.
```

Issue Labels

Label Purpose
bug Something isn't working
enhancement New feature or request
documentation Documentation improvements
good first issue Good for newcomers
help wanted Extra attention needed
priority: high High priority issue
priority: low Low priority issue
question Further information is requested
wontfix This will not be worked on

Issue Triage Process

1. Initial Triage:
   · Check if issue is valid and reproducible
   · Add appropriate labels
   · Set priority level
   · Assign to relevant team/milestone
2. Resolution:
   · Assign to developer
   · Link to PR when fix is in progress
   · Close when resolved
3. Communication:
   · Acknowledge receipt within 24 hours
   · Provide updates on progress
   · Request more information if needed

Community & Support

Getting Help

1. Documentation: Check docs.frextech-sim.com
2. GitHub Issues: For bugs and feature requests
3. Discord Community: discord.gg/frextech
4. Email: support@frextech-sim.com

Community Guidelines

1. Be Respectful: Treat all community members with respect
2. Be Constructive: Provide constructive feedback and suggestions
3. Share Knowledge: Help others learn and grow
4. Follow Rules: Adhere to community guidelines and codes of conduct

Recognition

We recognize contributors in several ways:

1. Contributor Hall of Fame: Listed in CONTRIBUTORS.md
2. Special Thanks: Acknowledged in release notes
3. Swag: Contributors of significant features may receive swag
4. Beta Access: Early access to new features

Contributor Levels

Level Requirements Benefits
New Contributor First contribution Welcome message, contributor badge
Active Contributor 5+ merged PRs Beta feature access, voting rights
Core Contributor 20+ merged PRs, maintain a module Write access, mentorship role
Maintainer Significant contributions, community trust Admin access, release management

License

By contributing to FrexTech AI Simulations, you agree that your contributions will be licensed under the project's MIT License.

Contact

· Project Lead: Alex Chen (alex@frextech-sim.com)
· Technical Lead: Maria Rodriguez (maria@frextech-sim.com)
· Community Manager: David Kim (david@frextech-sim.com)

---

This document is adapted from multiple open source contribution guides.
Last Updated: January 1, 2024
Version: 2.0
