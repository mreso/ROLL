"""Monarch backend for the ROLL distributed RL framework.

This module provides a full Backend implementation backed by Meta's torchmonarch
package.  All torchmonarch imports are lazy so the module can be imported and
inspected even on machines where torchmonarch is not installed; actual Monarch
functionality is only required when methods that talk to the runtime are called.

Design overview
---------------
* **Worker proxy**: ROLL workers are plain Python classes.  Monarch requires
  actors to subclass ``monarch.actor.Actor`` with ``@endpoint`` methods.  A
  ``_MonarchWorkerProxy`` adapter class is created lazily on first use; it wraps
  an arbitrary worker class and delegates calls via a generic ``_invoke``
  endpoint.
* **Single-actor meshes**: Each ROLL actor maps to a 1-proc ProcMesh with a
  single proxy actor spawned on it.  ``ActorHandle._inner`` stores a
  ``_MonarchActorRef`` that holds the actor mesh, proc mesh, and actor name.
* **Name registry**: Named actors are tracked in ``_actor_registry`` for
  ``get_actor()`` lookups.
* **Object store shim**: A local ``dict`` with incrementing IDs stands in for a
  distributed object store.  ``RemoteRef._inner`` wraps either an
  ``_ObjectStoreRef`` (for ``put()`` values) or a ``_MonarchFutureRef`` (for
  pending ``invoke()`` results).
* **Resource queries**: ``torch.cuda.device_count()``, ``os.cpu_count()``, and
  ``socket`` are used for cluster introspection since Monarch does not expose a
  runtime resource query API.
* **Placement groups**: Stored as configuration dicts and consulted when actors
  reference them.  Monarch binds placement to process creation, so readiness is
  immediate.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

from roll.distributed.backend.interface import Backend
from roll.distributed.backend.types import (
    ActorHandle,
    BackendConfig,
    NodeInfo,
    PlacementSpec,
    RemoteRef,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight inner wrappers stored inside ActorHandle / RemoteRef
# ---------------------------------------------------------------------------


class _MonarchActorRef:
    """Holds the Monarch actor mesh and proc mesh for a single logical actor."""

    __slots__ = ("actor_mesh", "proc_mesh", "name", "_is_local")

    def __init__(self, actor_mesh: Any, proc_mesh: Any, name: Optional[str]):
        self.actor_mesh = actor_mesh
        self.proc_mesh = proc_mesh
        self.name = name
        # Local actors (proc_mesh=None at creation time) store a raw worker
        # instance in actor_mesh and use the synchronous fallback in invoke().
        # Remote Monarch actors always go through the _invoke endpoint.
        self._is_local = proc_mesh is None

    def __repr__(self) -> str:
        return f"_MonarchActorRef(name={self.name!r})"

    def __getstate__(self):
        # proc_mesh is only needed by the driver for spawning; skip it.
        return {"actor_mesh": self.actor_mesh, "name": self.name, "_is_local": self._is_local}

    def __setstate__(self, state):
        object.__setattr__(self, "actor_mesh", state["actor_mesh"])
        object.__setattr__(self, "proc_mesh", None)
        object.__setattr__(self, "name", state["name"])
        object.__setattr__(self, "_is_local", state.get("_is_local", False))

    def __getattr__(self, name):
        """Return a proxy that mimics Ray's remote method pattern.

        Supports both:
          handle.method.remote(*args, **kwargs)  — Ray-style
          handle.method(*args, **kwargs)         — direct call style
        """
        actor_ref = self

        class _MethodProxy:
            """Proxy returned by attribute access on a Monarch actor handle."""

            def remote(self_proxy, *args, **kwargs):
                from roll.distributed.backend import get_backend
                from roll.distributed.backend.types import ActorHandle
                return get_backend().invoke(ActorHandle(inner=actor_ref), name, args, kwargs)

            def __call__(self_proxy, *args, **kwargs):
                return self_proxy.remote(*args, **kwargs)

        return _MethodProxy()


class _MonarchFutureRef:
    """Holds a Monarch ``ValueMesh`` future for deferred result retrieval."""

    __slots__ = ("future",)

    def __init__(self, future: Any):
        self.future = future

    def __repr__(self) -> str:
        return f"_MonarchFutureRef({self.future!r})"

    def __await__(self):
        """Support ``await RemoteRef(inner=_MonarchFutureRef(...))``."""

        async def _resolve():
            value_mesh = self.future.get()
            if hasattr(value_mesh, "item"):
                try:
                    return value_mesh.item()
                except Exception:
                    pass
            if hasattr(value_mesh, "values"):
                try:
                    values = list(value_mesh.values())
                    return values[0] if len(values) == 1 else values
                except Exception:
                    pass
            return value_mesh

        return _resolve().__await__()


class _ObjectStoreRef:
    """Holds a key into the local object store."""

    __slots__ = ("ref_id",)

    def __init__(self, ref_id: int):
        self.ref_id = ref_id

    def __repr__(self) -> str:
        return f"_ObjectStoreRef(id={self.ref_id})"

    def __await__(self):
        """Support ``await RemoteRef(inner=_ObjectStoreRef(...))``."""
        from roll.distributed.backend import get_backend

        async def _resolve():
            return get_backend()._object_store.get(self.ref_id)

        return _resolve().__await__()


class _PlacementGroupRef:
    """Holds configuration for a Monarch placement group."""

    __slots__ = ("pg_id", "bundles")

    def __init__(self, pg_id: int, bundles: List[Dict[str, float]]):
        self.pg_id = pg_id
        self.bundles = bundles

    def __repr__(self) -> str:
        return f"_PlacementGroupRef(id={self.pg_id}, bundles={len(self.bundles)})"


# ---------------------------------------------------------------------------
# Lazy worker proxy factory
# ---------------------------------------------------------------------------

_worker_proxy_cls: Optional[Type] = None
_proxy_lock = threading.Lock()


def _get_worker_proxy_cls() -> Type:
    """Return the ``_MonarchWorkerProxy`` class, creating it on first call.

    The class must be built lazily because it inherits from
    ``monarch.actor.Actor`` which may not be importable at module load time.
    """
    global _worker_proxy_cls
    if _worker_proxy_cls is not None:
        return _worker_proxy_cls

    with _proxy_lock:
        # Double-checked locking.
        if _worker_proxy_cls is not None:
            return _worker_proxy_cls

        from monarch.actor import Actor, endpoint  # type: ignore[import-untyped]

        class _MonarchWorkerProxy(Actor):
            """Generic proxy that wraps any plain-Python worker class.

            On construction the real worker is instantiated.  The single
            ``_invoke`` endpoint dispatches arbitrary method calls into the
            underlying worker, handling both synchronous and coroutine
            return values.
            """

            def __init__(self, worker_cls: Type, worker_args: tuple, worker_kwargs: dict, env_vars: dict = None):
                # Inject per-actor environment variables into this worker
                # process BEFORE creating the worker.  Worker.__init__ reads
                # RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK from
                # os.environ immediately, so they must be set first.
                import os as _os

                if env_vars:
                    for key, value in env_vars.items():
                        _os.environ[key] = str(value)

                # Ensure the backend singleton is initialized in this worker
                # process.  In Ray every worker has the Ray client; in Monarch
                # we replicate that by auto-initializing the backend here.
                from roll.distributed.backend import get_backend
                from roll.distributed.backend.types import BackendConfig

                _backend = get_backend()
                if not _backend.is_initialized():
                    _backend.init(BackendConfig())

                self._worker = worker_cls(*worker_args, **worker_kwargs)

            @endpoint  # type: ignore[misc]
            async def _invoke(self, method_name: str, args: tuple, kwargs: dict) -> Any:
                method = getattr(self._worker, method_name)
                result = method(*args, **kwargs)
                # If the worker method is a coroutine, await it transparently.
                import asyncio

                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    result = await result
                return result

        _worker_proxy_cls = _MonarchWorkerProxy
        return _worker_proxy_cls


# ---------------------------------------------------------------------------
# Exception monitor (mirrors RayExceptionMonitor but runs locally or as a
# Monarch actor when the runtime is available)
# ---------------------------------------------------------------------------


class _LocalExceptionMonitor:
    """In-process exception monitor used when Monarch actors are unavailable."""

    def __init__(self) -> None:
        self._node_and_err_msg: Dict[str, List[str]] = defaultdict(list)
        self.running = True
        self.stop_count = 0

    def add_error_node_and_msg(self, ip: str, msg: str) -> None:
        self._node_and_err_msg[ip].append(msg)

    def get_error_node_and_msg(self) -> Dict[str, List[str]]:
        return dict(self._node_and_err_msg)

    def get_error_msg(self, ip: str) -> List[str]:
        return self._node_and_err_msg[ip]

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.running = False
        self.stop_count += 1

    def get_stop_count(self) -> int:
        return self.stop_count


# ---------------------------------------------------------------------------
# MonarchBackend
# ---------------------------------------------------------------------------


class MonarchBackend(Backend):
    """Full ``Backend`` implementation backed by Meta's torchmonarch runtime.

    The backend is importable without torchmonarch installed.  Monarch imports
    happen lazily inside the methods that need them, guarded by
    ``try / except ImportError`` so that the class can be instantiated and
    introspected (e.g. for tests or documentation) on any machine.
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._config: Optional[BackendConfig] = None

        # Monarch runtime handles -- set during init().
        self._host_mesh: Any = None

        # Actor name -> ActorHandle registry for get_actor() lookups.
        self._actor_registry: Dict[str, ActorHandle] = {}

        # Local object store: ref_id -> value.
        self._object_store: Dict[int, Any] = {}
        self._next_ref_id: int = 0
        self._store_lock = threading.Lock()

        # Placement groups: pg_id -> _PlacementGroupRef.
        self._placement_groups: Dict[int, _PlacementGroupRef] = {}
        self._next_pg_id: int = 0

        # Exception monitor (created lazily).
        self._exception_monitor: Optional[ActorHandle] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "MonarchBackend has not been initialized. Call init() first."
            )

    def _next_store_id(self) -> int:
        with self._store_lock:
            ref_id = self._next_ref_id
            self._next_ref_id += 1
            return ref_id

    @staticmethod
    def _get_hostname() -> str:
        return socket.gethostname()

    @staticmethod
    def _get_ip() -> str:
        try:
            return socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            return "127.0.0.1"

    @staticmethod
    def _gpu_count() -> int:
        try:
            import torch

            return torch.cuda.device_count()
        except Exception:
            return 0

    @staticmethod
    def _cpu_count() -> int:
        return os.cpu_count() or 1

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def init(self, config: BackendConfig) -> None:
        """Initialize the Monarch transport and host mesh.

        Parameters
        ----------
        config:
            ``BackendConfig`` with optional address, namespace, env_vars.
        """
        if self._initialized:
            logger.info("MonarchBackend is already initialized; skipping re-init.")
            return

        # Apply any requested environment variables before touching Monarch.
        if config.env_vars:
            for key, value in config.env_vars.items():
                os.environ[key] = value

        try:
            from monarch.actor import enable_transport, this_host  # type: ignore[import-untyped]

            try:
                enable_transport("tcp")
            except RuntimeError:
                # Transport already enabled (e.g. inside a Monarch worker
                # process).  This is expected and safe to ignore.
                pass
            self._host_mesh = this_host()
            logger.info("MonarchBackend: transport enabled, host mesh acquired.")
        except ImportError:
            logger.warning(
                "torchmonarch is not installed. MonarchBackend will operate in "
                "degraded mode with local-only object store and no remote actors."
            )
            self._host_mesh = None

        self._config = config
        self._initialized = True
        logger.info("MonarchBackend initialized.")

    def shutdown(self) -> None:
        """Shut down the Monarch runtime and release all resources."""
        if not self._initialized:
            return

        try:
            from monarch.actor import shutdown_context  # type: ignore[import-untyped]

            # shutdown_context() returns a Future; call .get() to block until done.
            shutdown_context().get(timeout=10.0)
            logger.info("MonarchBackend: Monarch context shut down.")
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("MonarchBackend: error during shutdown: %s", exc)

        self._host_mesh = None
        self._actor_registry.clear()
        self._object_store.clear()
        self._placement_groups.clear()
        self._exception_monitor = None
        self._initialized = False
        logger.info("MonarchBackend shut down.")

    def is_initialized(self) -> bool:
        return self._initialized

    # ==================================================================
    # Cluster Introspection
    # ==================================================================

    def nodes(self) -> List[NodeInfo]:
        """Return discovered nodes.

        Monarch does not provide a cluster-wide node listing API comparable to
        ``ray.nodes()``.  We return a single-element list describing the
        current host based on local introspection.
        """
        self._require_initialized()
        return [self._build_local_node_info()]

    def available_resources(self) -> Dict[str, float]:
        """Return locally visible resources (GPU count, CPU count, memory)."""
        self._require_initialized()
        resources: Dict[str, float] = {
            "CPU": float(self._cpu_count()),
            "GPU": float(self._gpu_count()),
        }
        try:
            import torch

            if torch.cuda.is_available():
                # Report total GPU memory in bytes across all devices.
                total_mem = sum(
                    torch.cuda.get_device_properties(i).total_mem
                    for i in range(torch.cuda.device_count())
                )
                resources["GPU_memory"] = float(total_mem)
        except Exception:
            pass
        return resources

    def current_node_id(self) -> str:
        return self._get_hostname()

    def runtime_address(self) -> str:
        self._require_initialized()
        return f"monarch://{self._get_hostname()}"

    def is_driver(self) -> bool:
        """The process calling backend methods is always the driver."""
        return True

    def get_current_node(self) -> NodeInfo:
        """Build ``NodeInfo`` for the current host."""
        self._require_initialized()
        return self._build_local_node_info()

    def _build_local_node_info(self) -> NodeInfo:
        return NodeInfo(
            node_id=self._get_hostname(),
            ip=self._get_ip(),
            resources=self.available_resources() if self._initialized else {},
            alive=True,
        )

    # ==================================================================
    # Actor Lifecycle
    # ==================================================================

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
        """Create a remote actor backed by a Monarch actor mesh.

        Parameters
        ----------
        cls:
            The plain-Python worker class to instantiate remotely.
        args, kwargs:
            Positional / keyword arguments forwarded to ``cls.__init__``.
        name:
            Optional logical name for later lookup via ``get_actor()``.
        namespace:
            Optional namespace qualifier (combined with *name*).
        placement:
            Placement hints (placement group, node affinity, etc.).
        env_vars:
            Extra environment variables to inject into the worker process.
        num_cpus, num_gpus:
            Resource requests.  ``num_gpus`` drives the ProcMesh size.
        max_concurrency:
            Concurrency hint (informational; Monarch actors are single-
            threaded by default).
        get_if_exists:
            If ``True`` and *name* is already registered, return the existing
            handle instead of creating a duplicate.
        """
        self._require_initialized()
        kwargs = kwargs or {}

        # If get_if_exists and a named actor already exists, return it.
        registry_key = self._registry_key(name, namespace)
        if get_if_exists and registry_key and registry_key in self._actor_registry:
            logger.debug("MonarchBackend: returning existing actor %r", registry_key)
            return self._actor_registry[registry_key]

        # env_vars are injected into the worker process via the proxy
        # constructor, not set in the driver.  See _MonarchWorkerProxy.__init__.

        # Determine GPU count for the ProcMesh.
        gpu_count = max(1, int(num_gpus)) if num_gpus > 0 else 0

        # --- Attempt real Monarch spawn ---
        if self._host_mesh is not None:
            try:
                actor_ref = self._spawn_monarch_actor(
                    cls, args, kwargs, name, gpu_count, env_vars=env_vars
                )
                handle = ActorHandle(inner=actor_ref)
                if registry_key:
                    self._actor_registry[registry_key] = handle
                return handle
            except Exception as exc:
                logger.warning(
                    "MonarchBackend: Monarch actor spawn failed (%s). "
                    "Falling back to local proxy.",
                    exc,
                )

        # --- Fallback: local in-process proxy (no Monarch runtime) ---
        actor_ref = self._spawn_local_actor(cls, args, kwargs, name, env_vars=env_vars)
        handle = ActorHandle(inner=actor_ref)
        if registry_key:
            self._actor_registry[registry_key] = handle
        return handle

    @staticmethod
    def _sanitize_monarch_name(name: str) -> str:
        """Sanitize a name for Monarch's Rust-side name parser.

        Monarch actor/proc names must be valid identifiers: alphanumeric plus
        underscores, and must not start with a digit.
        """
        import re

        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"a{sanitized}"
        return sanitized or "unnamed"

    def _spawn_monarch_actor(
        self,
        cls: Type,
        args: tuple,
        kwargs: dict,
        name: Optional[str],
        gpu_count: int,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> _MonarchActorRef:
        """Spawn a real Monarch actor on a fresh ProcMesh.

        GPU isolation is handled by CUDA_VISIBLE_DEVICES in env_vars (set by
        the Cluster layer), NOT by requesting GPUs from Monarch's ProcMesh.
        Requesting GPUs via spawn_procs({"gpus": N}) causes Monarch to assign
        the same GPU to every process, breaking NCCL collectives.
        """
        proc_mesh_kwargs: Dict[str, Any] = {}
        # Don't request GPUs from Monarch -- CUDA_VISIBLE_DEVICES in env_vars
        # handles GPU assignment (same pattern as Ray's fractional GPU).

        # Use the bootstrap callback to inject env_vars at process startup,
        # BEFORE CUDA is initialized.  Setting CUDA_VISIBLE_DEVICES after CUDA
        # init has no effect, so this must happen in bootstrap, not __init__.
        def _bootstrap() -> None:
            import os as _os

            if env_vars:
                for key, value in env_vars.items():
                    _os.environ[key] = str(value)

        proc_mesh = self._host_mesh.spawn_procs(proc_mesh_kwargs, bootstrap=_bootstrap)

        proxy_cls = _get_worker_proxy_cls()
        actor_name = name or f"actor_{id(cls)}_{self._next_store_id()}"
        actor_name = self._sanitize_monarch_name(actor_name)
        actor_mesh = proc_mesh.spawn(actor_name, proxy_cls, cls, args, kwargs, env_vars or {})

        return _MonarchActorRef(
            actor_mesh=actor_mesh,
            proc_mesh=proc_mesh,
            name=name,
        )

    @staticmethod
    def _spawn_local_actor(
        cls: Type,
        args: tuple,
        kwargs: dict,
        name: Optional[str],
        env_vars: Optional[Dict[str, str]] = None,
    ) -> _MonarchActorRef:
        """Create a local (in-process) stand-in when Monarch is unavailable.

        The ``actor_mesh`` slot stores the raw worker instance so that
        ``invoke()`` can call methods on it directly.
        """
        if env_vars:
            for key, value in env_vars.items():
                os.environ[key] = str(value)
        worker = cls(*args, **kwargs)
        return _MonarchActorRef(actor_mesh=worker, proc_mesh=None, name=name)

    # ==================================================================
    # Actor Lookup
    # ==================================================================

    def get_actor(self, name: str, namespace: Optional[str] = None) -> ActorHandle:
        """Look up a previously created named actor.

        Raises ``KeyError`` if no actor with the given name exists.
        """
        self._require_initialized()
        registry_key = self._registry_key(name, namespace)
        if registry_key not in self._actor_registry:
            raise KeyError(
                f"No actor registered with name={name!r}, namespace={namespace!r}. "
                f"Known actors: {list(self._actor_registry.keys())}"
            )
        return self._actor_registry[registry_key]

    @staticmethod
    def _registry_key(name: Optional[str], namespace: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        if namespace:
            return f"{namespace}/{name}"
        return name

    # ==================================================================
    # Remote Invocation
    # ==================================================================

    def invoke(
        self,
        handle: ActorHandle,
        method_name: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> RemoteRef:
        """Invoke a method on a remote actor, returning a ``RemoteRef``.

        For Monarch actors the call goes through the ``_invoke`` endpoint on
        the ``_MonarchWorkerProxy``.  For local fallback actors the method is
        called directly and the result is stashed in the object store.
        """
        self._require_initialized()
        kwargs = kwargs or {}
        actor_ref: _MonarchActorRef = handle._inner

        if not actor_ref._is_local:
            # Real Monarch actor -- dispatch through the proxy endpoint.
            # This works both for freshly created refs (proc_mesh set) and
            # deserialized refs (proc_mesh=None but actor_mesh still valid).
            future = actor_ref.actor_mesh._invoke.call(method_name, args, kwargs)
            return RemoteRef(inner=_MonarchFutureRef(future))

        # Local fallback -- call synchronously and store the result.
        worker = actor_ref.actor_mesh  # raw worker instance
        method = getattr(worker, method_name)
        result = method(*args, **kwargs)
        # Handle coroutine results from async methods.
        import asyncio

        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
        ref_id = self._next_store_id()
        with self._store_lock:
            self._object_store[ref_id] = result
        return RemoteRef(inner=_ObjectStoreRef(ref_id))

    # ==================================================================
    # Blocking Materialization
    # ==================================================================

    def get(
        self,
        refs: Union[RemoteRef, List[RemoteRef]],
        timeout: Optional[float] = None,
    ) -> Any:
        """Block until one or more ``RemoteRef`` values are ready.

        Parameters
        ----------
        refs:
            A single ``RemoteRef`` or a list of them.
        timeout:
            Optional timeout in seconds.  Currently honoured only for Monarch
            futures (where the underlying ``Future.get`` supports it); object
            store lookups are instantaneous.
        """
        self._require_initialized()
        if isinstance(refs, list):
            return [self._get_single(r, timeout) for r in refs]
        return self._get_single(refs, timeout)

    def _get_single(self, ref: RemoteRef, timeout: Optional[float]) -> Any:
        inner = ref._inner

        if isinstance(inner, _ObjectStoreRef):
            with self._store_lock:
                try:
                    return self._object_store[inner.ref_id]
                except KeyError:
                    raise ValueError(
                        f"Object store reference {inner.ref_id} not found. "
                        "It may have been garbage-collected."
                    )

        if isinstance(inner, _MonarchFutureRef):
            return self._resolve_monarch_future(inner.future, timeout)

        # Raw passthrough for any unexpected inner type.
        return inner

    @staticmethod
    def _resolve_monarch_future(future: Any, timeout: Optional[float]) -> Any:
        """Resolve a Monarch ``ValueMesh`` to a plain Python value.

        Monarch's ``Future.get()`` returns a ``ValueMesh``.  We call
        ``.item()`` to unwrap single-element meshes and ``.values()`` for
        multi-element ones, falling back to the raw value if neither works.
        """
        try:
            if timeout is not None:
                value_mesh = future.get(timeout=timeout)
            else:
                value_mesh = future.get()
        except Exception:
            # Re-raise with a friendlier message.
            raise

        # Unwrap ValueMesh -> plain Python object.
        if hasattr(value_mesh, "item"):
            try:
                return value_mesh.item()
            except Exception:
                pass
        if hasattr(value_mesh, "values"):
            try:
                values = value_mesh.values()
                if len(values) == 1:
                    return values[0]
                return values
            except Exception:
                pass
        return value_mesh

    # ==================================================================
    # Object Store
    # ==================================================================

    def put(self, obj: Any) -> RemoteRef:
        """Store an object locally and return a ``RemoteRef`` to it."""
        ref_id = self._next_store_id()
        with self._store_lock:
            self._object_store[ref_id] = obj
        return RemoteRef(inner=_ObjectStoreRef(ref_id))

    # ==================================================================
    # Placement Groups
    # ==================================================================

    def create_placement_group(
        self,
        bundles: List[Dict[str, float]],
    ) -> Any:
        """Create a placement group.

        Monarch ties placement to process creation, so placement groups are
        stored as configuration and consulted when actors reference them.
        The returned object is a ``_PlacementGroupRef`` that can be passed as
        ``PlacementSpec.placement_group``.
        """
        self._require_initialized()
        pg_id = self._next_pg_id
        self._next_pg_id += 1
        pg = _PlacementGroupRef(pg_id=pg_id, bundles=list(bundles))
        self._placement_groups[pg_id] = pg
        logger.debug("MonarchBackend: created placement group %d with %d bundles", pg_id, len(bundles))
        return pg

    def remove_placement_group(self, pg: Any) -> None:
        """Remove a previously created placement group."""
        if isinstance(pg, _PlacementGroupRef):
            self._placement_groups.pop(pg.pg_id, None)
            logger.debug("MonarchBackend: removed placement group %d", pg.pg_id)
        else:
            logger.warning(
                "MonarchBackend: remove_placement_group called with "
                "unrecognized object: %r",
                pg,
            )

    def wait_placement_group_ready(self, pg: Any) -> None:
        """Wait for a placement group to become ready.

        In Monarch, placement is immediate at process-spawn time, so this is
        a no-op.
        """
        pass

    # ==================================================================
    # Profiling
    # ==================================================================

    def timeline(self, filename: str) -> None:
        """Export an execution timeline.

        Monarch has its own telemetry subsystem.  We attempt to use it if
        available, otherwise log a message.
        """
        try:
            from monarch.actor import timeline as monarch_timeline  # type: ignore[import-untyped]

            monarch_timeline(filename)
            logger.info("MonarchBackend: timeline exported to %s", filename)
        except ImportError:
            logger.info(
                "MonarchBackend: timeline export requested (%s) but Monarch "
                "telemetry is not available.",
                filename,
            )
        except Exception as exc:
            logger.warning("MonarchBackend: timeline export failed: %s", exc)

    # ==================================================================
    # Log Monitoring
    # ==================================================================

    def get_log_directory(self) -> str:
        """Return the log directory for the Monarch backend."""
        log_dir = os.environ.get("MONARCH_LOG_DIR", "/tmp/monarch/logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def get_node_ip_address(self) -> str:
        """Return the IP address of the current node."""
        return self._get_ip()

    def create_exception_monitor(self) -> ActorHandle:
        """Create an exception monitor actor.

        If Monarch is fully available the monitor runs as a remote actor;
        otherwise a local in-process instance is used.  Either way the handle
        is cached so repeated calls return the same monitor.
        """
        if self._exception_monitor is not None:
            return self._exception_monitor

        # Attempt to spawn via Monarch, fall back to local.
        try:
            handle = self.create_actor(
                cls=_LocalExceptionMonitor,
                name="ExceptionMonitor",
                get_if_exists=True,
            )
        except Exception:
            # Ultimate fallback: wrap a local instance directly.
            monitor = _LocalExceptionMonitor()
            ref = _MonarchActorRef(actor_mesh=monitor, proc_mesh=None, name="ExceptionMonitor")
            handle = ActorHandle(inner=ref)

        self._exception_monitor = handle
        return handle
