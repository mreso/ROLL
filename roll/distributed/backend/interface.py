from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from roll.distributed.backend.types import (
    ActorHandle,
    BackendConfig,
    NodeInfo,
    PlacementSpec,
    RemoteRef,
)


class Backend(ABC):
    """Protocol that all distributed backends must implement."""

    # --- Lifecycle ---

    @abstractmethod
    def init(self, config: BackendConfig) -> None:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

    @abstractmethod
    def is_initialized(self) -> bool:
        ...

    # --- Cluster Introspection ---

    @abstractmethod
    def nodes(self) -> List[NodeInfo]:
        ...

    @abstractmethod
    def available_resources(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def current_node_id(self) -> str:
        ...

    @abstractmethod
    def runtime_address(self) -> str:
        ...

    @abstractmethod
    def is_driver(self) -> bool:
        ...

    # --- Actor Lifecycle ---

    @abstractmethod
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
        ...

    # --- Remote Invocation ---

    @abstractmethod
    def invoke(
        self,
        handle: ActorHandle,
        method_name: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> RemoteRef:
        ...

    # --- Blocking Materialization ---

    @abstractmethod
    def get(
        self,
        refs: Union[RemoteRef, List[RemoteRef]],
        timeout: Optional[float] = None,
    ) -> Any:
        ...

    # --- Object Store ---

    @abstractmethod
    def put(self, obj: Any) -> RemoteRef:
        ...

    # --- Placement ---

    @abstractmethod
    def create_placement_group(
        self,
        bundles: List[Dict[str, float]],
    ) -> Any:
        ...

    @abstractmethod
    def remove_placement_group(self, pg: Any) -> None:
        ...

    @abstractmethod
    def wait_placement_group_ready(self, pg: Any) -> None:
        ...

    # --- Profiling ---

    @abstractmethod
    def timeline(self, filename: str) -> None:
        ...

    # --- Log Monitoring ---

    @abstractmethod
    def get_log_directory(self) -> str:
        """Get the log directory for this backend."""
        ...

    @abstractmethod
    def get_node_ip_address(self) -> str:
        """Get the IP address of the current node."""
        ...

    @abstractmethod
    def create_exception_monitor(self) -> ActorHandle:
        """Create an exception monitor actor."""
        ...
