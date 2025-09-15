from pathlib import Path


def get_project_root(start: Path | None = None) -> Path:
    """Walk upwards until a project marker is found."""
    start = start or Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists() or (parent / "setup.py").exists():
            return parent
    raise RuntimeError("Project root not found")
