# EddyP1 Development Guidelines

## Project Context

- **Purpose**: The core of this project is floating point math code to compute Atmospheric CO2 Perimeter Flux (ACPF), supplemented with Marimo notebooks to visualize and explore the data, the ACPF computation, and sensor reliability, and to generate graphs for research reports.
- **No external clients**: This is an internal research and computation project. It won't become a library, but it will eventually get used as an element of a data processing pipeline. Prioritize rapid evolution and simplicity over backward-compatibility or public API boilerplate.
- **Keep `__init__.py` simple**: Do not maintain explicit symbol re-exports in package `__init__.py` files; import directly from submodules.

## Tooling & Environment

- **Environment**: The virtual environment is managed by `uv` in `.venv/` based on `pyproject.toml`.
- **Tech stack**: Leverage Numpy, SciPy, scipy.spatial.ConvexHull, Polars, Marimo, and Pytest to do what they do best. If there are more fitting libraries or additional fitting libraries for the purpose, point that out.
- **Testing & Execution**: Maintain `uv run pytest` for tests, `uv run ruff check .` for linting, and `uv run ty check` for static type checking. Run all three (`uv run ruff check . && uv run ty check && uv run pytest`) after code changes.
- **Notebooks**: Place Marimo notebooks in `notebooks/` (covered by Ruff `extend-per-file-ignores` in `pyproject.toml`).
- **Tools**: Support PyCharm, Antigravity, and Antigravity IDE.
- **LaTeX in Markdown**: Write LaTeX in the portable `$...$` / multiline `$$...$$` subset so it renders cleanly across PyCharm, Antigravity IDE, and GitHub (do not use GitHub-only `` $`...`$ `` or ```` ```math ```` blocks, which PyCharm's native preview does not render):
  - **Subscripts (`_`)**: Surround `_` with ASCII spaces inside math (e.g., `$\mathbf{p} _ i$`, `$H _ {\text{MBL}}$`) so GitHub's CommonMark pass never treats `}_` / `_{` as `<em>` italics across expressions.
  - **Display blocks & multi-row environments**: Place `$$` on its own lines with blank lines before and after; put `\\` at the very end of each physical row line (never mid-line and never with trailing `[8pt]`), and never leave a standalone `=` or `-` alone on a line inside `$$`.
  - **Parentheses, punctuation & bold/italics**: PyCharm rejects `($` at the start of a parenthetical and GitHub rejects zero-width spaces (`U+200B`) or `/` / `-` directly before `$` (e.g., `wind/$H$` or `small-$H$`)—instead move parentheses inside math for pure math (`$(h _ {\text{ellip}} = 0)$`), start mixed parentheticals with a word (`(where $x$ = East)`), and separate `/` or `-` with a space (`wind / $H _ {\text{MBL}}$`, `small $H _ {\text{MBL}}$`). Do not wrap `$...$` inside `*...*` italics (GitHub disables math inside `<em>`), do not start a bold span with `$` (move it outside: `$N = 2$ **baseline**`), and avoid ending a parenthetical with `...)$)` (GitHub's closing-`$` edge case).
  - **Backslash-punctuation & macros**: GitHub's CommonMark pass strips `\` before ASCII punctuation even inside math, so use alphabetic macros instead: `\thinspace` (not `\,`), `\negthinspace` (not `\!`), `\lbrace`/`\rbrace` (not `\{`/`\}`), `\lvert`/`\rvert` and `\lVert`/`\rVert` (not `|`/`\|`, which also avoids table column breaks), and `\mathrm` (not `\operatorname`). Place `%` outside math delimiters (e.g., `$< 0.02$%`) so `\%` is never stripped into a LaTeX comment.

## Design Philosophy

- **Test Driven Development**: Maintain thorough automated tests for the floating point math computations, and write the tests first. Also write tests for the visualization code.
- **Lean APIs**: Avoid speculative "just in case they're useful" method aliases, convenience wrappers, or redundant accessors. Keep classes and functions focused on known requirements.
- **Minimize Code Duplication**: Share core logic and avoid duplicated geometric or mathematical transformations across notebooks and modules. Refactor as needed to reduce code duplication, or indicate when to run a refactoring step separate from a feature-change step.
