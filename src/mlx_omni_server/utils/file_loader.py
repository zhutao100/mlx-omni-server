from importlib import resources


def read_package_text(*path_parts: str, package: str | None = None, encoding: str = "utf-8") -> str:
    """Read a text resource from the installed package."""
    if not path_parts:
        raise ValueError("path_parts must not be empty")

    target_package = package or __package__.split(".")[0]
    try:
        resource = resources.files(target_package)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise FileNotFoundError(f"Package '{target_package}' is not available") from exc

    for part in path_parts:
        resource = resource / part

    try:
        return resource.read_text(encoding=encoding)
    except FileNotFoundError:
        raise
    except OSError as exc:
        resource_name = "/".join(path_parts)
        raise OSError(f"Unable to read resource '{resource_name}'") from exc
