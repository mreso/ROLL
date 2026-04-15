from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from roll.distributed.backend.interface import Backend
from roll.distributed.backend.types import (
    ActorHandle,
    BackendConfig,
    NodeInfo,
    PlacementSpec,
    RemoteRef,
)


class MonarchBackend(Backend):
    """Monarch backend implementation. Provides basic functionality for demonstration."""

    def __init__(self):
        self._initialized = False
        self._actors = {}  # Simple actor registry
        self._objects = {}  # Simple object store
        self._next_ref_id = 0

    def _not_implemented(self, method: str) -> None:
        raise NotImplementedError(
            f"MonarchBackend.{method}() is not yet implemented. "
            "This is a demonstration backend. For production use, implement full Monarch integration."
        )

    def init(self, config: BackendConfig) -> None:
        """Initialize Monarch backend (demo implementation)."""
        print("MonarchBackend: Initializing (demo mode)")
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown Monarch backend (demo implementation)."""
        print("MonarchBackend: Shutting down")
        self._initialized = False
        self._actors.clear()
        self._objects.clear()

    def is_initialized(self) -> bool:
        return self._initialized

    def nodes(self) -> List[NodeInfo]:
        self._not_implemented("nodes")

    def available_resources(self) -> Dict[str, float]:
        self._not_implemented("available_resources")

    def current_node_id(self) -> str:
        return "monarch-node-0"  # Demo implementation

    def runtime_address(self) -> str:
        return "monarch://localhost:8000"  # Demo implementation

    def is_driver(self) -> bool:
        return True

    def get_current_node(self) -> NodeInfo:
        """Get current node info (demo implementation)."""
        return NodeInfo(
            node_id="monarch-node-0",
            node_ip="127.0.0.1",
            resources={"CPU": 8.0, "GPU": 2.0}
        )

    def create_actor(
        self,
        cls: Type,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        placement: Optional[PlacementSpec] = None,
        env_vars: Optional[Dict[str, str]] = None,
        num_cpus: float = 0,
        num_gpus: float = 0,
        max_concurrency: int = 1,
        get_if_exists: bool = False,
    ) -> ActorHandle:
        self._not_implemented("create_actor")

    def invoke(
        self,
        handle: ActorHandle,
        method_name: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> RemoteRef:
        self._not_implemented("invoke")

    def get(
        self,
        refs: Union[RemoteRef, List[RemoteRef]],
        timeout: Optional[float] = None,
    ) -> Any:
        self._not_implemented("get")

    def put(self, obj: Any) -> RemoteRef:
        self._not_implemented("put")

    def create_placement_group(
        self,
        bundles: List[Dict[str, float]],
    ) -> Any:
        self._not_implemented("create_placement_group")

    def remove_placement_group(self, pg: Any) -> None:
        self._not_implemented("remove_placement_group")

    def wait_placement_group_ready(self, pg: Any) -> None:
        self._not_implemented("wait_placement_group_ready")

    def timeline(self, filename: str) -> None:
        """Export timeline (demo implementation)."""
        print(f"MonarchBackend: Would export timeline to {filename} (demo mode)")
        # In real implementation, would export Monarch execution timeline

    def export_timeline(self, filename: str) -> None:
        """Export timeline (demo implementation)."""
        self.timeline(filename)

    def get_log_directory(self) -> str:
        self._not_implemented("get_log_directory")

    def get_node_ip_address(self) -> str:
        self._not_implemented("get_node_ip_address")

    def create_exception_monitor(self) -> ActorHandle:
        self._not_implemented("create_exception_monitor")
