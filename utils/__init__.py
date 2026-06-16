import hashlib
import json
import os
import random
from collections import deque
from concurrent.futures import Future
from typing import *

import numpy as np
import torch

T = TypeVar('T')

def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

def seed_everything(seed=0):
    """
    Helper to seed random number generators.

    Note, however, that even with the seeds fixed, some non-determinism is possible.

    For more details read <https://pytorch.org/docs/stable/notes/randomness.html>.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# dictionary utils

def str2dict(dict_str):
    if dict_str.startswith('{') and dict_str.endswith('}'):
        # Replace single quotes with double quotes for JSON parsing
        dict_str = dict_str.replace("'", '"')
        dict_str = dict_str.replace(',}', '}')
        dict_str = dict_str.replace(',]', ']')
        dict_str = dict_str.replace('True', 'true').replace('False', 'false')
        dict = json.loads(dict_str)
        return dict
    else:
        raise ValueError(f"Invalid dictionary string: {dict_str}")

class MockThreadPoolExecutor:
    """
    A single-threaded mock implementation of ThreadPoolExecutor for debugging purposes.
    Executes tasks sequentially in the main thread instead of using a thread pool.
    """

    def __init__(self, max_workers=None, thread_name_prefix='', initializer=None, initargs=()):
        self._shutdown = False
        self._tasks = deque()
        self._initializer = initializer
        self._initargs = initargs

        # Run initializer if provided
        if self._initializer:
            self._initializer(*self._initargs)

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """
        Submit a task for execution.
        Instead of running in a separate thread, executes immediately and returns a completed Future.
        """
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')

        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)

        return future

    def map(self, fn: Callable[..., T], *iterables: Iterable[Any], timeout=None, chunksize=1) -> Iterable[T]:
        """
        Returns an iterator equivalent to map(fn, *iterables).
        Executes tasks sequentially instead of in parallel.
        Important: This implementation ensures all tasks are executed even if their results are not consumed,
        which is necessary for side effects like updating progress bars.
        """
        return list(self._map(fn, *iterables, timeout=timeout, chunksize=chunksize))

    def _map(self, fn: Callable[..., T], *iterables: Iterable[Any], timeout=None, chunksize=1) -> Iterable[T]:
        """
        Returns an iterator equivalent to map(fn, *iterables).
        Executes tasks sequentially instead of in parallel.
        Important: This implementation ensures all tasks are executed even if their results are not consumed,
        which is necessary for side effects like updating progress bars.
        """
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')

        # Create a list to store futures
        futures = []

        # Submit all tasks first to ensure they all execute
        for args in zip(*iterables):
            future = self.submit(fn, *args)
            futures.append(future)

        # Now yield the results
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                yield result
            except Exception as exc:
                # Continue processing even if individual tasks fail
                raise exc
                # continue

    def shutdown(self, wait=True, *, cancel_futures=False):
        """
        Signal the executor that it should free any resources.
        Since this is a mock implementation, it just sets the shutdown flag.
        """
        self._shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False
