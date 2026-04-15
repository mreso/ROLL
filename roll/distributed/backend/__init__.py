from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roll.distributed.backend.interface import Backend

_backend: Backend | None = None


def get_backend() -> Backend:
    global _backend
    if _backend is None:
        backend_name = os.environ.get("ROLL_BACKEND", "ray")
        if backend_name == "ray":
            from roll.distributed.backend.ray_backend import RayBackend

            _backend = RayBackend()
        elif backend_name == "monarch":
            from roll.distributed.backend.monarch_backend import MonarchBackend

            _backend = MonarchBackend()
        else:
            raise ValueError(
                f"Unknown backend: {backend_name!r}. "
                "Supported: 'ray', 'monarch'."
            )
    return _backend


def reset_backend() -> None:
    """Reset the cached backend instance. Useful for testing."""
    global _backend
    _backend = None


# Convenience re-exports
from roll.distributed.backend.types import (  # noqa: E402, F401
    ActorGroup,
    ActorHandle,
    BackendConfig,
    NodeInfo,
    PlacementSpec,
    RemoteRef,
)
