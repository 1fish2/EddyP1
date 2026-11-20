# EddyP1 Development Guidelines

## Project Context
- **Purpose**: The core of this project is floating point math code to compute Atmospheric CO2 Perimeter Flux (ACPF), supplemented with Marimo notebooks to visualize and explore the data, the ACPF computation, and sensor reliability, and to generate graphs for research reports.
- **No external clients**: This is an internal research and computation project. It won't become a library, but it will eventually get used as an element of a data processing pipeline. Prioritize rapid evolution and simplicity over backward-compatibility or public API boilerplate.
- **Keep `__init__.py` simple**: Do not maintain explicit symbol re-exports in package `__init__.py` files; import directly from submodules.

## Tooling & Environment
- **Environment**: The virtual environment is managed by `uv` in `.venv/` based on `pyproject.toml`.
- **Tech stack**: Leverage Numpy, Scipy, scipy.spatial.ConvexHull, Polars, Marimo, and Pytest to do what they do best. If there are more fitting libraries or additional fitting libraries for the purpose, point that out.
- **Testing & Execution**: Maintain `uv run pytest` as a way to run the tests.

## Design Philosophy
- **Test Driven Development**: Maintain thorough automated tests for the floating point math computations, and write the tests first. Also write tests for the visualization code.
- **Lean APIs**: Avoid speculative "just in case they're useful" method aliases, convenience wrappers, or redundant accessors. Keep classes and functions focused on known requirements.
- **Minimize Code Duplication**: Share core logic and avoid duplicated geometric or mathematical transformations across notebooks and modules. Refactor as needed to reduce code duplication, or indicate when to run a refactoring step separate from a feature-change step.
