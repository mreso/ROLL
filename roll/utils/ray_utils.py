# NOTE: This file contains Ray-specific utility functions that use @ray.remote.
# These are called from strategy files (e.g., vLLM strategy) which depend on
# Ray-specific placement group mechanics. They should NOT be migrated to the
# backend abstraction individually -- the callers need their own migration path.
# When using the Monarch backend, these functions may need adaptation.
import os

import ray


@ray.remote
def get_visible_gpus(device_control_env_var: str):
    return os.environ.get(device_control_env_var, "").split(",")


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
