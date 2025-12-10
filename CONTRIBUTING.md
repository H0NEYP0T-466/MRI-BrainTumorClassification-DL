# 🤝 Contributing to MRI Brain Tumor Classification

Thank you for your interest in contributing to this project! We welcome contributions from the community and are grateful for your support.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## 📜 Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## 🎯 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- Use the bug report template
- Describe the exact steps to reproduce the problem
- Provide specific examples
- Describe the behavior you observed and what you expected
- Include screenshots if applicable
- Note your environment (OS, Python version, Node.js version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use the feature request template
- Provide a clear description of the problem and solution
- Explain why this enhancement would be useful
- List any alternatives you've considered

### Pull Requests

- Fill in the pull request template
- Follow the code style guidelines
- Include appropriate test coverage
- Update documentation as needed
- Ensure all tests pass

## 🚀 Getting Started

### Prerequisites

- Node.js 16+ (for frontend)
- Python 3.8+ (for backend)
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MRI-BrainTumorClassification-DL.git
   cd MRI-BrainTumorClassification-DL
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL.git
   ```

### Set Up Development Environment

#### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Frontend Setup

```bash
# From root directory
npm install
```

## 🔄 Development Workflow

### Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/description` - for new features
- `fix/description` - for bug fixes
- `docs/description` - for documentation changes
- `refactor/description` - for code refactoring
- `test/description` - for test additions/changes

### Make Changes

1. Make your changes in your branch
2. Test your changes thoroughly
3. Ensure code follows style guidelines
4. Update documentation if needed

### Commit Messages

Write clear, descriptive commit messages following this format:

```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system or dependency changes
- `ci`: CI/CD changes
- `chore`: Other changes that don't modify src or test files

Example:
```
feat: add CLAHE preprocessing step

Implement Contrast Limited Adaptive Histogram Equalization
to improve contrast in MRI images before model inference.

Closes #123
```

## 🎨 Code Style Guidelines

### Python (Backend)

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for functions and classes
- Use meaningful variable and function names

Format your code using:
```bash
black app/
```

### TypeScript/JavaScript (Frontend)

- Follow the existing ESLint configuration
- Use TypeScript for all new code
- Use functional components with hooks
- Maximum line length: 100 characters
- Use meaningful variable and function names

Lint your code:
```bash
npm run lint
```

### CSS

- Use CSS modules for component styles
- Follow BEM naming convention when applicable
- Use CSS variables from `src/styles/variables.css`
- Keep styles modular and reusable

## 🧪 Testing Guidelines

### Backend Tests

```bash
cd backend
pytest
```

- Write unit tests for new functions
- Include integration tests for API endpoints
- Aim for meaningful test coverage
- Test edge cases and error conditions

### Frontend Tests

```bash
npm run test  # if configured
```

- Test component rendering
- Test user interactions
- Test API integration points

### Manual Testing

Before submitting a PR:

1. Test the backend API:
   ```bash
   cd backend
   uvicorn app.main:app --reload
   # Test endpoints at http://localhost:8000/docs
   ```

2. Test the frontend:
   ```bash
   npm run dev
   # Test UI at http://localhost:5173
   ```

3. Test the complete workflow:
   - Upload an image
   - Verify preprocessing steps display correctly
   - Check prediction results
   - Verify error handling

## 📚 Documentation

### Code Documentation

- Add docstrings to Python functions and classes
- Add JSDoc comments to TypeScript functions
- Document complex algorithms and logic
- Include examples where helpful

### User Documentation

- Update README.md if adding new features
- Update API documentation for new endpoints
- Add comments for configuration changes
- Document any new environment variables

### Examples

Python docstring example:
```python
def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess MRI image for model inference.
    
    Args:
        image_path: Path to the input image file
        target_size: Desired output size (width, height)
    
    Returns:
        Preprocessed image as numpy array
        
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image cannot be processed
    """
    # Implementation
```

TypeScript JSDoc example:
```typescript
/**
 * Uploads an image file to the prediction API
 * @param file - The image file to upload
 * @returns Promise with prediction results
 * @throws {Error} If upload fails or file is invalid
 */
async function uploadImage(file: File): Promise<PredictionResult> {
  // Implementation
}
```

## 📤 Submitting Changes

### Before Submitting

1. Pull the latest changes from upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Run all tests and linting:
   ```bash
   # Backend
   cd backend
   pytest
   
   # Frontend
   cd ..
   npm run lint
   npm run build
   ```

3. Ensure your commits are clean and descriptive

### Create a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to the GitHub repository and create a Pull Request

3. Fill out the PR template completely:
   - Link related issues
   - Describe your changes
   - List breaking changes (if any)
   - Add screenshots for UI changes
   - Check all applicable boxes in the checklist

4. Wait for review and address feedback

### Review Process

- Maintainers will review your PR
- Automated tests will run
- You may be asked to make changes
- Once approved, your PR will be merged

### After Merge

1. Delete your feature branch:
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. Update your local main branch:
   ```bash
   git checkout main
   git pull upstream main
   ```

## 🐛 Reporting Security Issues

Please do not report security vulnerabilities through public GitHub issues.

See [SECURITY.md](SECURITY.md) for information on how to report security issues responsibly.

## 💬 Questions?

- Check existing issues and discussions
- Create a new issue with the question label
- Reach out to maintainers

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

<p align="center">Happy Contributing! 🎉</p>
