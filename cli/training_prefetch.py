#!/usr/bin/env python
"""Transparent wrapper that adds a data-loading prefetch thread to
microwakeword training, then delegates to model_train_eval.

Instead of:
    python -m microwakeword.model_train_eval --training_config ...
Use:
    python training_prefetch.py --training_config ...

How it works:
    model.train_on_batch() releases the GIL while running TF/GPU ops.
    get_data() does mmap reads + numpy work that also (largely) releases
    the GIL.  By running get_data() in a background thread we overlap
    data prep with GPU execution — free speed for zero risk.

    If the monkey-patch fails for any reason, training runs normally
    without prefetch (graceful fallback).
"""

import sys
import threading
import queue as _queue_mod

# ---------------------------------------------------------------------------
# Monkey-patch FeatureHandler *before* the training module imports it.
# ---------------------------------------------------------------------------

_PREFETCH_DEPTH = 2          # batches to keep ready in the queue
_PREFETCH_ENABLED = False     # flipped to True on successful patch

try:
    import microwakeword.data as _mww_data

    _OrigFeatureHandler = _mww_data.FeatureHandler

    class _PrefetchFeatureHandler(_OrigFeatureHandler):
        """FeatureHandler with 1-thread / N-batch-ahead prefetch for training."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._pf_q      = _queue_mod.Queue(maxsize=_PREFETCH_DEPTH)
            self._pf_stop   = threading.Event()
            self._pf_thread = None
            self._pf_key    = None          # hashable key of last-prefetched args

        # -- internal helpers -----------------------------------------------

        @staticmethod
        def _args_key(mode, kwargs):
            """Hashable snapshot of call arguments."""
            return repr((mode, sorted((k, repr(v)) for k, v in kwargs.items())))

        def _worker(self, mode, kwargs):
            """Background thread: continuously produce batches."""
            try:
                while not self._pf_stop.is_set():
                    batch = _OrigFeatureHandler.get_data(self, mode, **kwargs)
                    try:
                        self._pf_q.put(batch, timeout=0.5)
                    except _queue_mod.Full:
                        if self._pf_stop.is_set():
                            break
            except Exception:
                pass            # thread dies → get_data falls back to sync

        def _start_prefetch(self, mode, kwargs):
            self._stop_prefetch()
            self._pf_key = self._args_key(mode, kwargs)
            self._pf_stop.clear()
            t = threading.Thread(
                target=self._worker,
                args=(mode, dict(kwargs)),   # copy kwargs for the thread
                daemon=True,
            )
            t.start()
            self._pf_thread = t

        def _stop_prefetch(self):
            if self._pf_thread is not None and self._pf_thread.is_alive():
                self._pf_stop.set()
                # drain so the worker can unblock on put()
                while True:
                    try:
                        self._pf_q.get_nowait()
                    except _queue_mod.Empty:
                        break
                self._pf_thread.join(timeout=2)
            self._pf_thread = None
            self._pf_key = None

        # -- public API (drop-in replacement) --------------------------------

        def get_data(self, mode, **kwargs):
            # Only prefetch for training batches.
            if mode != "training":
                self._stop_prefetch()
                return _OrigFeatureHandler.get_data(self, mode, **kwargs)

            key = self._args_key(mode, kwargs)

            # Happy path: prefetch running with matching args → grab a batch.
            if (self._pf_key == key
                    and self._pf_thread is not None
                    and self._pf_thread.is_alive()):
                try:
                    return self._pf_q.get(timeout=30)
                except _queue_mod.Empty:
                    pass          # fall through to sync call

            # First call, arg change, or dead thread → sync + (re)start.
            self._stop_prefetch()
            result = _OrigFeatureHandler.get_data(self, mode, **kwargs)
            self._start_prefetch(mode, kwargs)
            return result

    # Install the patch.
    _mww_data.FeatureHandler = _PrefetchFeatureHandler
    _PREFETCH_ENABLED = True
    print(f"ℹ️  Training data prefetch enabled ({_PREFETCH_DEPTH} batches ahead)")

except Exception as exc:
    print(f"⚠️  Could not enable data prefetch ({exc}); continuing without")

# ---------------------------------------------------------------------------
# Delegate to the real training entry-point.
# ---------------------------------------------------------------------------
import runpy
runpy.run_module("microwakeword.model_train_eval", run_name="__main__", alter_sys=True)
