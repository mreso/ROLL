import os
import inspect
from typing import List, Type, Dict, Union, Any

from roll.configs.worker_config import WorkerConfig
from roll.distributed.backend import get_backend
from roll.distributed.backend.types import ActorHandle, PlacementSpec
from roll.distributed.executor.worker import Worker, RankInfo
from roll.distributed.scheduler.decorator import (
    BIND_WORKER_METHOD_FLAG,
    Dispatch,
    get_predefined_dispatch_fn,
    func_generator,
    get_predefined_execute_fn,
    collect_all_to_all,
    dispatch_one_to_all,
)
from roll.platforms import current_platform
from roll.utils.constants import RAY_NAMESPACE
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger


logger = get_logger()


def _has_async_methods(cls: Type) -> bool:
    """Check if a class has async methods (replacement for ray._private.async_compat.has_async_methods)"""
    for name in dir(cls):
        if not name.startswith('_'):
            method = getattr(cls, name, None)
            if callable(method) and inspect.iscoroutinefunction(method):
                return True
    return False


class Cluster:

    def __init__(
        self,
        name,
        worker_cls: Union[Type[Worker], str],
        resource_manager: ResourceManager,
        worker_config: WorkerConfig,
    ):

        self.cluster_name = name
        if isinstance(worker_cls, str):
            worker_cls = safe_import_class(worker_cls)

        self.worker_cls = worker_cls
        self.backend = get_backend()
        self.resource_manager = resource_manager
        self.placement_groups = None
        self.worker_config = worker_config

        self.workers: List[ActorHandle] = []

        self.master_addr = None
        self.master_port = None
        self.world_size = self.worker_config.world_size

        self._create_workers()
        self._bind_worker_method()
        self._worker_rank_info = None
        self.initialized = False

        self.rank2worker = {k: self.workers[k] for k in range(len(self.workers))}
        self.worker2rank = {self.workers[k]: k for k in range(len(self.workers))}

        # Get device info from all workers
        devices_refs = [self.backend.invoke(worker, "get_devices_info") for worker in self.workers]
        devices_info = self.backend.get(devices_refs)
        self.rank2devices = dict(zip(map(lambda worker: self.worker2rank[worker], self.workers), devices_info))

        # Get node IPs from all workers
        node_refs = [self.backend.invoke(worker, "get_node_ip") for worker in self.workers]
        node_ips = self.backend.get(node_refs)
        self.worker2nodes = dict(zip(self.workers, node_ips))

        logger.debug(f"{self.cluster_name} rank2devices {self.rank2devices}")
        # Keep worker_cls for backend create_actor calls
        # Note: Unlike Ray version, we keep worker_cls as it's needed for backend.create_actor()

    @property
    def dp_size(self):
        return self.worker_rank_info[0].dp_size

    @property
    def tp_size(self):
        return self.worker_rank_info[0].tp_size

    @property
    def pp_size(self):
        return self.worker_rank_info[0].pp_size

    @property
    def cp_size(self):
        return self.worker_rank_info[0].cp_size

    @property
    def vp_size(self):
        if (self.worker_config.strategy_args.strategy_config and
            'virtual_pipeline_model_parallel_size' in self.worker_config.strategy_args.strategy_config):
            return self.worker_config.strategy_args.strategy_config['virtual_pipeline_model_parallel_size']
        else:
            return 1

    @property
    def worker_rank_info(self) -> List[RankInfo]:
        if not self._worker_rank_info or not self.initialized:
            # initialize 后RankInfo不能改变了，使用缓存
            self._worker_rank_info: List[RankInfo] = self.execute_all_sync(method_name="get_rank_info")
        return self._worker_rank_info

    def get_rank_info(self, rank):
        assert 0 <= rank < self.world_size, f"rank must be from [0, world_size), Got {rank}"
        return self.worker_rank_info[rank]

    def _create_workers(self):
        placement_groups: List[List[Dict]] = self.resource_manager.allocate_placement_group(
            device_mapping=self.worker_config.device_mapping, world_size=self.worker_config.world_size
        )
        logger.debug(f"placement_groups: {placement_groups}")
        self.placement_groups = placement_groups

        for rank, pgs in enumerate(placement_groups):
            deploy_pg = pgs[0]
            pg_zero_gpu_ranks = sorted([pg["gpu_rank"] for pg in pgs if pg["node_rank"] == deploy_pg["node_rank"]])

            # Include GPU IDs in worker name for timeline visualization
            # Format: actor_train-0-G0 (single GPU) or actor_infer-0-G01 (TP=2)
            if pg_zero_gpu_ranks and deploy_pg["gpu_rank"] is not None:
                gpu_str = "".join(str(g) for g in pg_zero_gpu_ranks)
                worker_name = f"{self.cluster_name}-{rank}-G{gpu_str}"
            else:
                # CPU-only workers
                worker_name = f"{self.cluster_name}-{rank}"
            env_vars = {
                "WORLD_SIZE": str(self.world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(0),
                "CLUSTER_NAME": self.cluster_name,
                "WORKER_NAME": worker_name,
            }

            if rank != 0:
                env_vars["MASTER_ADDR"] = self.master_addr
                env_vars["MASTER_PORT"] = str(self.master_port)
            if deploy_pg["gpu_rank"] is not None:
                current_platform.update_env_vars_for_visible_devices(env_vars=env_vars, gpu_ranks=pg_zero_gpu_ranks)
            if "ROLL_LOG_DIR" in os.environ:
                env_vars["ROLL_LOG_DIR"] = os.environ["ROLL_LOG_DIR"]
            env_vars.update(self.worker_config.system_envs)

            self.worker_config.resource_placement_groups = pgs

            if _has_async_methods(self.worker_cls):
                max_concurrency = (self.worker_config.max_concurrency if self.worker_config.max_concurrency > 1
                                else 1000) # equivalent to DEFAULT_MAX_CONCURRENCY_ASYNC in ray
                logger.info(f"set max_concurrency to {max_concurrency} for worker {type(self.worker_cls)}")
            else:
                assert self.worker_config.max_concurrency == 1
                max_concurrency = 1

            # Set up placement specification
            placement = PlacementSpec(placement_group=deploy_pg["placement_group"])

            # Configure GPU/NPU resources
            num_cpus = 0.01
            num_gpus = 0.0
            if current_platform.ray_device_key == "GPU":
                num_gpus = 0.01 if self.worker_config.device_mapping else 0
            elif current_platform.ray_device_key == "NPU":
                # NPU resources will be handled via custom resources in the future
                # For now, keep num_gpus = 0 and rely on placement groups
                pass

            worker = self.backend.create_actor(
                cls=self.worker_cls,
                args=(self.worker_config,),  # worker_config passed as positional arg
                name=worker_name,
                namespace=RAY_NAMESPACE,
                placement=placement,
                env_vars=env_vars,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_concurrency=max_concurrency,
            )
            self.workers.append(worker)
            if rank == 0:
                master_ref = self.backend.invoke(worker, "get_master_addr_and_port")
                self.master_addr, self.master_port = self.backend.get(master_ref)

    def _bind_worker_method(self):
        """
        magic method: 用Cluster来代理向List[Worker]的请求
        ref: https://github.com/volcengine/verl/blob/27b43eba2b8905fdf18237548e596819e1831fdb/single_controller/base/worker_group.py#L136C9-L136C28
        """
        for method_name in dir(self.worker_cls):
            if method_name.startswith("_"):
                continue
            try:
                method = getattr(self.worker_cls, method_name)
                assert callable(method), f"{method_name} in {self.worker_cls} is not callable"
            except Exception as e:
                logger.debug(str(e))
                continue

            if hasattr(method, BIND_WORKER_METHOD_FLAG):

                attribute = getattr(method, BIND_WORKER_METHOD_FLAG)
                assert isinstance(attribute, Dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, f"attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]

                if isinstance(dispatch_mode, Dispatch):
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                execute_fn_name = execute_mode["execute_fn_name"]

                try:
                    execute_fn = getattr(self, execute_fn_name)
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception as e:
                    logger.warning(str(e))
                    raise

                func = func_generator(
                    self, method_name, dispatch_fn=dispatch_fn, collect_fn=collect_fn, execute_fn=execute_fn
                )
                try:
                    setattr(self, method_name, func)
                except Exception as e:
                    logger.warning(str(e))
                    raise ValueError(f"Fail to set method_name {method_name}")

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        return self.backend.get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        return self.backend.invoke(self.workers[0], method_name, args, kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        return self.backend.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        length = len(self.workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(self.backend.invoke(self.workers[i], method_name, sliced_args, sliced_kwargs))
                return result

        return [self.backend.invoke(worker, method_name, args, kwargs) for worker in self.workers]

    def __getstate__(self):
        """Allow pickling for cross-process transfer (e.g., Monarch backend).

        Only serialize the data needed by remote consumers (workers that
        receive the Cluster for model_update coordination): the worker
        list, config, and cluster metadata.
        """
        return {
            "workers": self.workers,
            "worker_config": self.worker_config,
            "cluster_name": self.cluster_name,
            "world_size": self.world_size,
            "master_addr": getattr(self, "master_addr", None),
            "master_port": getattr(self, "master_port", None),
        }

    def __setstate__(self, state):
        """Reconstruct a lightweight Cluster proxy on the receiving side."""
        self.workers = state["workers"]
        self.worker_config = state["worker_config"]
        self.cluster_name = state["cluster_name"]
        self.world_size = state["world_size"]
        self.master_addr = state.get("master_addr")
        self.master_port = state.get("master_port")
        # Reconstruct backend from singleton
        from roll.distributed.backend import get_backend
        self.backend = get_backend()
        # Rebuild lookup dicts
        self.rank2worker = {k: self.workers[k] for k in range(len(self.workers))}
        self.worker2rank = {}
        for k in range(len(self.workers)):
            self.worker2rank[self.workers[k]] = k
        # These are not needed on the receiving side
        self.resource_manager = None
        self.placement_groups = []
        self.worker2nodes = {}
