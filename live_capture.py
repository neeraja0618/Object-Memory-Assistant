"""One replaceable frame slot: slow recognition never accumulates a camera backlog."""
import threading


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._sequence = 0
        self._sample = None

    def publish(self, sample):
        with self._lock:
            self._sequence += 1
            self._sample = sample

    def after(self, sequence):
        with self._lock:
            if self._sequence == sequence:
                return sequence, None
            return self._sequence, self._sample
