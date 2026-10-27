# EddyP1 Development Guidelines

## Project Context
- **No external clients**: This is an internal research and computation project for Atmospheric CO₂ Perimeter Flux (ACPF). Prioritize rapid evolution and simplicity over rigid backward-compatibility or public API boilerplate.
- **Keep `__init__.py` simple**: Do not maintain explicit symbol re-exports in package `__init__.py` files; import directly from submodules (e.g. `perimeter_flux.geometry`).

## Tooling & Environment
- **Environment**: Virtual environment is managed by `uv` in `.venv/` based on `pyproject.toml`.
- **Testing & Execution**: Run commands using `uv run pytest` or `.venv/bin/pytest`.

## Design Philosophy
- **Lean APIs**: Avoid speculative "just in case they're useful" method aliases, convenience wrappers, or redundant accessors. Keep classes and functions focused on current requirements.
- **Minimal Code Duplication**: Share core logic and avoid duplicated geometric or mathematical transformations across notebooks and modules.
