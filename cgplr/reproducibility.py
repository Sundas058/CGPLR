import hashlib
from pathlib import Path
import threading
import psutil


def source_fingerprints(root):
    root = Path(root)
    files = sorted((root/'cgplr').rglob('*.py'))
    files += sorted((root/'configs').glob('*.json')) if (root/'configs').exists() else []
    return {str(path.relative_to(root)).replace('\\','/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


class ProcessMemory:

    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _sample(self):
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *error):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self._stop.set()
        self.thread.join()
