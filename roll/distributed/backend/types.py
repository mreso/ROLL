from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RemoteRef:
    """Opaque handle to a value on a remote worker."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any):
        self._inner = inner

    def __await__(self):
        return self._inner.__await__()

    def __repr__(self) -> str:
        return f"RemoteRef({self._inner!r})"


class ActorHandle:
    """Opaque handle to a single remote actor."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, name)

    def __repr__(self) -> str:
        return f"ActorHandle({self._inner!r})"


class ActorGroup:
    """Ordered collection of actor handles."""

    __slots__ = ("handles",)

    def __init__(self, handles: List[ActorHandle]):
        self.handles = handles

    def __len__(self) -> int:
        return len(self.handles)

    def __getitem__(self, idx: int) -> ActorHandle:
        return self.handles[idx]

    def __iter__(self) -> Iterator[ActorHandle]:
        return iter(self.handles)

    def __repr__(self) -> str:
        return f"ActorGroup(len={len(self.handles)})"


@dataclass
class PlacementSpec:
    """Where to place an actor. Backend translates to its native equivalent."""

    node_rank: Optional[int] = None
    node_id: Optional[str] = None
    gpu_rank: Optional[int] = None
    placement_group: Any = None


@dataclass
class NodeInfo:
    """Discovered node information."""

    node_id: str
    ip: str
    resources: Dict[str, float]
    alive: bool


@dataclass
class BackendConfig:
    """Runtime initialization config."""

    address: Optional[str] = None
    namespace: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    log_to_driver: bool = True
