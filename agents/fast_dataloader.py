from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
import queue
import threading


class FastDataLoaderWrapperIter(object):
    
    def __init__(self, dataloader, queue_size):
        self._dataloader = dataloader
        self._flush_queue = queue.Queue(queue_size)
        self._flush_thread = threading.Thread(target=self._flush_thread_run)
        self._flush_thread.daemon = True
        self._flush_thread.start()

    def _flush_thread_run(self):
        for data in self._dataloader:
            self._flush_queue.put(data)
        self._flush_queue.put(None)

    def __next__(self):
        print(self._flush_queue.qsize())
        out = self._flush_queue.get()
        if out:
            return out
        else:
            self._shutdown_workers()
            raise StopIteration

    next = __next__

    def _shutdown_workers(self):
        print("in shutdown_workers")
        self._flush_thread.terminate()

    def __del__(self):
        self._shutdown_workers()
         

class FastDataLoaderWrapper(object):
    """A faster wrapper for pytorch dataloader."""

    def __init__(self, dataloader, queue_size):
        self._dataloader = dataloader
        self._queue_size = queue_size

    def __iter__(self):
        return FastDataLoaderWrapperIter(self._dataloader, self._queue_size)

    def __len__(self):
        return len(self._dataloader)
