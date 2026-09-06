"""Private, temporary storage for a single Streamlit server process.

Only activity/operation bookkeeping is shared; PDF contents and indexes are not.
An operation lease keeps cleanup from deleting files during a running app script,
including time spent waiting for the shared model. Browser idleness does not renew it.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

_TOKEN = re.compile(r"[0-9a-f]{32}")
_ACTIVITY = ".last_activity"


@dataclass(frozen=True)
class Session:
    token: str
    path: str
    reset: bool


class SessionStore:
    def __init__(self, root=None, ttl_seconds=3600, cleanup_interval=60):
        if ttl_seconds <= 0 or cleanup_interval <= 0:
            raise ValueError("Session lifetime and cleanup interval must be positive")
        self.root = os.path.abspath(os.fspath(root or os.path.join(
            tempfile.gettempdir(), "doc_finder_sessions")))
        if os.path.islink(self.root):
            raise ValueError("Session storage must not be a symbolic link")
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self._lock = threading.RLock()
        self._active: Dict[str, int] = {}
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def _path(self, token):
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise ValueError("Invalid session identifier")
        # A validated token is a single path component, never a client-supplied path.
        return os.path.join(self.root, token)

    @staticmethod
    def _directory(path):
        return os.path.isdir(path) and not os.path.islink(path)

    @staticmethod
    def _last_activity(path):
        marker = os.path.join(path, _ACTIVITY)
        if os.path.isfile(marker) and not os.path.islink(marker):
            return os.stat(marker, follow_symlinks=False).st_mtime
        # Covers interrupted directory creation without following a marker symlink.
        return os.stat(path, follow_symlinks=False).st_mtime

    @staticmethod
    def _touch(path):
        marker = os.path.join(path, _ACTIVITY)
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(marker, flags, 0o600)
        try:
            os.utime(fd, None)
        finally:
            os.close(fd)

    @contextmanager
    def session(self, token=None):
        """Lease this session's directory; replace expired/missing/invalid sessions."""
        with self._lock:
            path = self._path(token) if isinstance(token, str) and _TOKEN.fullmatch(token) else None
            reset = True
            if path and self._directory(path):
                if self._active.get(token) or time.time() - self._last_activity(path) < self.ttl_seconds:
                    reset = False
                else:
                    shutil.rmtree(path)
            if reset:
                while True:
                    token = secrets.token_hex(16)
                    path = self._path(token)
                    try:
                        os.mkdir(path, 0o700)
                    except FileExistsError:
                        continue
                    break
            self._touch(path)
            self._active[token] = self._active.get(token, 0) + 1
        try:
            yield Session(token=token, path=path, reset=reset)
        finally:
            with self._lock:
                self._active[token] -= 1
                if not self._active[token]:
                    del self._active[token]
                # Clear may have removed the directory during this operation.
                if self._directory(path):
                    self._touch(path)

    def expired(self, token):
        """Read-only check for an idle browser; never renew the session lifetime."""
        with self._lock:
            path = self._path(token)
            if self._active.get(token):
                return False
            return (not self._directory(path)
                    or time.time() - self._last_activity(path) >= self.ttl_seconds)

    def clear(self, token):
        """Remove only the supplied session. A lease exit will not recreate it."""
        with self._lock:
            path = self._path(token)
            if self._directory(path):
                shutil.rmtree(path)

    def cleanup(self, now=None):
        """Delete expired, inactive session directories; ignore unrelated entries."""
        cutoff = (time.time() if now is None else now) - self.ttl_seconds
        removed = 0
        with self._lock:
            with os.scandir(self.root) as entries:
                for entry in entries:
                    if (not _TOKEN.fullmatch(entry.name) or self._active.get(entry.name)
                            or not entry.is_dir(follow_symlinks=False)):
                        continue
                    try:
                        if self._last_activity(entry.path) <= cutoff:
                            shutil.rmtree(entry.path)
                            removed += 1
                    except OSError:
                        # A transient filesystem error should not kill future sweeps.
                        continue
        return removed

    def start_cleanup(self):
        """Start at most one daemon, so cleanup also happens without new visitors."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(target=self._cleanup_loop,
                                            name="doc-finder-cleanup", daemon=True)
            self._worker.start()

    def _cleanup_loop(self):
        while not self._stop.wait(self.cleanup_interval):
            try:
                self.cleanup()
            except OSError:
                continue

    def close(self):
        """Stop the cleanup daemon (primarily for tests)."""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
