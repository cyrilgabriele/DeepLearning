"""Top-level package marker for project-specific modules."""

# Re-export data utilities for convenient imports inside tests and scripts
from . import data, models  # noqa: F401
