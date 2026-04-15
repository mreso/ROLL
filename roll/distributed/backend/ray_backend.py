from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type, Union

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

from roll.distributed.backend.interface import Backend
from roll.distributed.backend.types import (
    ActorHandle,
    BackendConfig,
    NodeInfo,
    PlacementSpec,
    RemoteRef,
)


class RayBackend(Backend):
    """Backend implementation wrapping Ray."""

    # --- Lifecycle ---

    def init(self, config: BackendConfig) -> None:
        if ray.is_initialized():
            return
        runtime_env = {}
        if config.env_vars:
            runtime_env["env_vars"] = config.env_vars
        ray.init(
            address=config.address,
            namespace=config.namespace,
            ignore_reinit_error=True,
            log_to_driver=config.log_to_driver,
            runtime_env=runtime_env or None,
        )

    def shutdown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def is_initialized(self) -> bool:
        return ray.is_initialized()

    # --- Cluster Introspection ---

    def nodes(self) -> List[NodeInfo]:
        raw_nodes = ray.nodes()
        result = []
        for node in raw_nodes:
            result.append(
                NodeInfo(
                    node_id=node.get("NodeID", ""),
                    ip=node.get("NodeManagerAddress", ""),
                    resources=node.get("Resources", {}),
                    alive=node.get("Alive", False),
                )
            )
        return result

    def available_resources(self) -> Dict[str, float]:
        return dict(ray.available_resources())

    def current_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def runtime_address(self) -> str:
        return ray.get_runtime_context().gcs_address

    def is_driver(self) -> bool:
        if not ray.is_initialized():
            return True
        try:
            from ray._private.worker import WORKER_MODE
            return ray.get_runtime_context().worker.mode != WORKER_MODE
        except Exception:
            return True

    def get_actor(self, name: str, namespace: Optional[str] = None) -> ActorHandle:
        return ActorHandle(ray.get_actor(name, namespace=namespace))

    def get_current_node(self) -> NodeInfo:
        """Get current node information."""
        return NodeInfo(
            node_id=self.current_node_id(),
            ip=self.get_node_ip_address(),
            resources=self.available_resources(),
            alive=True
        )

    # --- Actor Lifecycle ---

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
        kwargs = kwargs or {}

        # Wrap as remote class if not already
        if not hasattr(cls, "__ray_actor_class__"):
            remote_cls = ray.remote(cls)
        else:
            remote_cls = cls

        # Build options
        options: Dict[str, Any] = {}
        if name is not None:
            options["name"] = name
        if namespace is not None:
            options["namespace"] = namespace
        if num_cpus > 0:
            options["num_cpus"] = num_cpus
        if num_gpus > 0:
            options["num_gpus"] = num_gpus
        if max_concurrency > 1:
            options["max_concurrency"] = max_concurrency
        if get_if_exists:
            options["get_if_exists"] = True

        # Placement
        if placement is not None:
            if placement.placement_group is not None:
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=placement.placement_group,
                )
            elif placement.node_id is not None:
                options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=placement.node_id,
                    soft=False,
                )

        # Environment variables
        if env_vars:
            options["runtime_env"] = RuntimeEnv(env_vars=env_vars)

        actor_handle = remote_cls.options(**options).remote(*args, **kwargs)
        return ActorHandle(inner=actor_handle)

    # --- Remote Invocation ---

    def invoke(
        self,
        handle: ActorHandle,
        method_name: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> RemoteRef:
        kwargs = kwargs or {}
        ray_handle = handle._inner
        remote_method = getattr(ray_handle, method_name)
        obj_ref = remote_method.remote(*args, **kwargs)
        return RemoteRef(inner=obj_ref)

    # --- Blocking Materialization ---

    def get(
        self,
        refs: Union[RemoteRef, List[RemoteRef]],
        timeout: Optional[float] = None,
    ) -> Any:
        if isinstance(refs, RemoteRef):
            return ray.get(refs._inner, timeout=timeout)
        elif isinstance(refs, list):
            raw_refs = [r._inner if isinstance(r, RemoteRef) else r for r in refs]
            return ray.get(raw_refs, timeout=timeout)
        else:
            # Pass through raw ray.ObjectRef for backwards compat during migration
            return ray.get(refs, timeout=timeout)

    # --- Object Store ---

    def put(self, obj: Any) -> RemoteRef:
        return RemoteRef(inner=ray.put(obj))

    # --- Placement ---

    def create_placement_group(
        self,
        bundles: List[Dict[str, float]],
    ) -> Any:
        pg = ray.util.placement_group(bundles)
        return pg

    def remove_placement_group(self, pg: Any) -> None:
        ray.util.remove_placement_group(pg)

    def wait_placement_group_ready(self, pg: Any) -> None:
        ray.get(pg.ready())

    # --- Profiling ---

    def timeline(self, filename: str) -> None:
        try:
            ray.timeline(filename=filename)
        except Exception:
            pass

    # --- Log Monitoring ---

    def get_log_directory(self) -> str:
        """Get Ray's log directory."""
        log_dir = os.path.dirname(os.path.dirname(ray.nodes()[0]["ObjectStoreSocketName"]))
        return os.path.join(log_dir, "logs")

    def get_node_ip_address(self) -> str:
        """Get the IP address of the current node."""
        return ray.util.get_node_ip_address()

    def create_exception_monitor(self) -> ActorHandle:
        """Create Ray-based exception monitor actor."""
        from roll.distributed.scheduler.log_monitor import RayExceptionMonitor
        actor = ray.remote(RayExceptionMonitor).options(
            name="ExceptionMonitor",
            get_if_exists=False
        ).remote()
        return ActorHandle(inner=actor)
