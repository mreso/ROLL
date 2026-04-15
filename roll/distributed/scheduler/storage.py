from roll.distributed.backend import get_backend
from roll.utils.logging import get_logger
logger = get_logger()


class SharedStorage:

    def __init__(self):
        self._storage = {}
        self.backend = get_backend()

    def put(self, key, data):
        ref = self.backend.put(data)
        self._storage[key] = ref

    def get(self, key):
        ref = self._storage.get(key)
        if ref is None:
            logger.warning(f"{key} is not found in storage")
            return None
        return self.backend.get(ref)

    def put_if_absent(self, key: str, data: any) -> bool:
        if key in self._storage:
            return False
        self._storage[key] = self.backend.put(data)
        return True