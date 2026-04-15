import dataclasses
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from roll.distributed.backend import get_backend
from roll.platforms import current_platform


def _get_visible_gpus(device_control_env_var: str):
    """Get visible GPUs from environment variable."""
    import os
    gpu_env = os.environ.get(device_control_env_var, "")
    if not gpu_env.strip():
        # If CUDA_VISIBLE_DEVICES is empty or not set, return empty list
        return []
    return [gpu.strip() for gpu in gpu_env.split(",") if gpu.strip()]


def _get_node_rank():
    """Get node rank from environment variable."""
    import os
    return int(os.environ.get("NODE_RANK", "0"))


class ResourceManager:
    def __init__(self, num_gpus_per_node, num_nodes):
        """
            The ResourceManager centrally manages the required GPU/CPU resources,
            facilitating backend to deploy Actors on specified GPU devices.
        """
        self.backend = get_backend()
        available_resources = self.backend.available_resources()
        available_gpu = available_resources.get(current_platform.ray_device_key, 0)

        nodes_maybe_used = []
        backend_nodes = self.backend.nodes()
        for node in backend_nodes:
            resource = node.resources
            node_gpu_num = int(resource.get(current_platform.ray_device_key, 0))
            if node_gpu_num >= num_gpus_per_node:
                nodes_maybe_used.append({"Resources": resource})
        nodes_maybe_used = sorted(nodes_maybe_used, key=lambda n: n["Resources"]["CPU"])

        ray_num_nodes = len(nodes_maybe_used)
        if num_nodes is None:
            num_nodes = ray_num_nodes

        assert num_nodes <= ray_num_nodes, (f"The Ray clusters(ray_num_nodes: {ray_num_nodes}) cannot meet the "
                                            f"required number of nodes (`num_nodes`{num_nodes}).")
        self.num_nodes = num_nodes
        self.gpu_per_node = num_gpus_per_node
        self.num_gpus = self.gpu_per_node * self.num_nodes

        if self.gpu_per_node > 0:
            assert self.num_gpus <= available_gpu, f"num_gpus {self.num_gpus} > available_gpu {available_gpu}"
            bundles = []
            for i in range(self.num_nodes):
                node = nodes_maybe_used[i]
                node_cpu = int(node["Resources"]["CPU"])
                bundles.append({current_platform.ray_device_key: self.gpu_per_node, "CPU": max(node_cpu / 2, 1)})

            self.placement_groups = [self.backend.create_placement_group([bundle]) for bundle in bundles]
            for pg in self.placement_groups:
                self.backend.wait_placement_group_ready(pg)
            # Note: These functions read environment variables, so we can call them directly
            # In a full migration, these would need to be scheduled on the placement groups
            gpu_ranks = [_get_visible_gpus(current_platform.device_control_env_var)
                        for _ in self.placement_groups]
            print(f"gpu ranks: {gpu_ranks}")
            self.node_ranks = [_get_node_rank() for _ in self.placement_groups]
            if self.node_ranks.count(0) > 1:
                # NODE_RANK environment variable is not set in the cluster, so a default value is used for NODE_RANK.
                self.node_ranks = list(range(len(self.placement_groups)))

            # Handle GPU ranks - if empty or invalid, use sequential numbering
            self.gpu_ranks = []
            for i, gpu_rank in enumerate(gpu_ranks):
                if gpu_rank and len(gpu_rank) > 0 and gpu_rank[0].isdigit():
                    self.gpu_ranks.append(int(gpu_rank[0]))
                else:
                    # Use sequential GPU numbering starting from 0
                    self.gpu_ranks.append(i)
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group
            print(f"node2pg: {self.node2pg}")
        else:
            assert self.num_nodes == 1
            node = nodes_maybe_used[0]
            node_cpu = int(node["Resources"]["CPU"])
            bundles = [{"CPU": node_cpu}] * self.num_nodes
            self.placement_groups = [self.backend.create_placement_group([bundle]) for bundle in bundles]
            for pg in self.placement_groups:
                self.backend.wait_placement_group_ready(pg)
            self.node_ranks = [0]
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group

    def nodes_placement_group(self, node_rank) -> Any:
        """
        mesh table是 m×n，获取第node_rank nodel上gpu_rank的PlacementGroup，用于把Actor部署到指定的GPU上
        """
        return self.node2pg[node_rank]

    def destroy_placement_group(self):
        for pg in self.placement_groups:
            self.backend.remove_placement_group(pg)

    def allocate_placement_group(self, world_size, device_mapping: List[int] = None) -> List[List[Dict]]:
        """
            Allocate resources according to device_mapping (numbered by GPU RANK)
            - GPUs: Specify required GPU indices via device_mapping
            - CPUs: Specify via world_size

            Return Type: List[List[Dict]]
              Dict Keys:
                - node_rank
                - gpu_rank
                - placement_group
              List[Dict]: Represents GPUs allocated to a worker and access to placement groups
              Example: If num_gpus_per_worker=8, then len(List[Dict])=8

            A Worker is defined as a group of resource owners (can span multiple machines) that can independently use allocated resources to execute computation operations.
        """
        allocated_pg = []
        # Get runtime address from backend for placement group metadata
        # Note: This was 'ray_address' in earlier versions, causing NameError - now fixed
        backend_address = self.backend.runtime_address()
        if not backend_address:
            raise RuntimeError("Backend runtime address is empty - backend may not be initialized")
        if device_mapping:
            num_gpus_per_worker = len(device_mapping) // world_size
            grouped_ranks = [
                list(device_mapping[i : i + num_gpus_per_worker])
                for i in range(0, len(device_mapping), num_gpus_per_worker)
            ]
            for group in grouped_ranks:
                pg_list = []
                for rank in group:
                    node_rank = rank // self.gpu_per_node
                    gpu_rank = rank % self.gpu_per_node

                    assert node_rank < self.num_nodes, (f"device_mapping used gpus are more than "
                                                        f"num_nodes×num_gpus_per_node={self.num_nodes}×{self.gpu_per_node}")

                    pg = self.nodes_placement_group(node_rank)
                    pg_list.append(
                        dict(node_rank=node_rank, gpu_rank=gpu_rank, placement_group=pg, ray_address=backend_address)
                    )
                allocated_pg.append(pg_list)
        else:
            # Try to spread the CPU workers across various nodes to avoid the out-of-memory (OOM) situation caused
            # by the concentration of CPU workers in one place and the resulting peak memory usage.
            for rank in range(world_size):
                node_rank = rank % self.num_nodes
                allocated_pg.append(
                    [
                        dict(
                            node_rank=node_rank,
                            gpu_rank=None,
                            placement_group=self.nodes_placement_group(node_rank),
                            ray_address=backend_address,
                        )
                    ]
                )

        assert len(allocated_pg) == world_size

        return allocated_pg
