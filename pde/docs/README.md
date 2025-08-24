# Documentation

This directory contains the MkDocs documentation for the PDE SDK.

## Quick Start

### Building Documentation

```bash
# Build the documentation
make docs

# Or directly with uv
uv run mkdocs build
```

### Serving Documentation Locally

```bash
# Start development server
make docs-serve

# Or directly with uv
uv run mkdocs serve
```

The documentation will be available at `http://localhost:8000`.

### Deploying to GitHub Pages

```bash
# Deploy to GitHub Pages
make docs-deploy

# Or directly with uv
uv run mkdocs gh-deploy --force
```

## Documentation Structure

```
docs/
├── index.md                 # Main landing page
├── overview.md              # SDK overview and architecture
├── equations-setup.md       # Equations and boundary conditions
├── domains-grids.md         # Spatial discretization
├── solvers-timestepping.md  # Time-stepping solvers
├── solvers-iterative.md     # Iterative solvers
├── visualization.md         # Plotting and visualization
├── best-practices.md        # Best practices and troubleshooting
├── Roadmap.md               # Development roadmap
└── README.md               # This file
```

## Writing Documentation

### MkDocs Material Theme Features

The documentation uses the Material theme with several extensions:

- **Admonitions**: Use `!!! note`, `!!! warning`, etc.
- **Code blocks**: Syntax highlighting with copy buttons
- **Math**: MathJax support for equations
- **Tabs**: Content tabs for different examples
- **Icons**: Material Design icons and emojis
- **Cards**: Grid layouts for navigation

### Example Admonitions

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

### Example Cards

```markdown
<div class="grid cards" markdown>
- :material-rocket: **Get Started**
- :material-book: **User Guide**
</div>
```

### Math Equations

```markdown
$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
$$
```

## Configuration

The documentation is configured in `mkdocs.yml` with:

- Material theme with dark/light mode toggle
- Search functionality
- Code copying and annotation features
- Custom navigation structure
- Social links and analytics (placeholder)

## Contributing

When adding new documentation:

1. Follow the existing structure
2. Use the Material theme features
3. Test locally with `mkdocs serve`
4. Ensure all links work correctly
5. Update the navigation in `mkdocs.yml` if needed

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Theme](https://squidfunk.github.io/mkdocs-material/)
- [Material Icons](https://fonts.google.com/icons)
- [Python Markdown Extensions](https://facelessuser.github.io/pymdown-extensions/)
