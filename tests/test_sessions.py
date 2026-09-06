"""Session isolation and lifecycle checks; no downloaded model is needed."""
import json
import os
from pathlib import Path
import re
import time

import pytest

import core
from session_storage import SessionStore


@pytest.fixture
def store(tmp_path):
    manager = SessionStore(tmp_path / "sessions", ttl_seconds=60)
    yield manager
    manager.close()


def expire(path):
    old = time.time() - 120
    os.utime(Path(path) / ".last_activity", (old, old))


def test_session_paths_are_random_private_and_distinct(store):
    with store.session() as a, store.session() as b:
        assert a.path != b.path and a.token != b.token
        assert a.reset and b.reset
        assert re.fullmatch(r"[0-9a-f]{32}", a.token)
        assert Path(a.path).parent == Path(store.root)
        assert Path(a.path).stat().st_mode & 0o777 == 0o700
        assert Path(store.root).stat().st_mode & 0o777 == 0o700
    with store.session(a.token) as same:
        assert same.path == a.path and not same.reset


def test_sessions_cannot_list_search_or_delete_each_others_documents(store, monkeypatch):
    import faiss
    import numpy as np

    with store.session() as a, store.session() as b:
        # Identical corpus IDs deliberately model the strongest accidental collision.
        cid = "0123abcd"
        for session, text in ((a, "A private passage"), (b, "B private passage")):
            folder = core.corpus_folder(session.path, cid)
            os.mkdir(folder)
            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1, 0]], dtype="float32"))
            faiss.write_index(index, os.path.join(folder, core.INDEX_FILENAME))
            Path(folder, core.META_FILENAME).write_text(json.dumps({
                "chunks": [{"page": 1, "text": text}],
            }))
            Path(folder, core.PDF_FILENAME).write_bytes(b"private PDF placeholder")
            core.save_collections(session.path, [{"id": cid, "folder": cid, "name": text}])

        assert [c["name"] for c in core.load_collections(a.path)] == ["A private passage"]
        assert [c["name"] for c in core.load_collections(b.path)] == ["B private passage"]
        monkeypatch.setattr(core, "embed", lambda *args, **kwargs: np.array([[1, 0]], dtype="float32"))
        for session, text in ((a, "A private passage"), (b, "B private passage")):
            corpus = core.load_corpus(core.corpus_folder(session.path, cid))
            assert [h.text for h in core.search(corpus, None, "query", 1)] == [text]

        core.delete_corpus(a.path, cid)
        assert core.load_collections(a.path) == []
        assert core.corpus_files_exist(core.corpus_folder(b.path, cid))
        assert core.load_collections(b.path)[0]["name"] == "B private passage"


def test_expired_and_missing_session_get_new_tokens(store):
    with store.session() as first:
        pass
    expire(first.path)
    with store.session(first.token) as replacement:
        assert replacement.reset and replacement.token != first.token
        assert not Path(first.path).exists()
    store.clear(replacement.token)
    with store.session(replacement.token) as missing:
        assert missing.reset and missing.token != replacement.token


def test_cleanup_skips_active_operations_and_refreshes_on_exit(store):
    with store.session() as active:
        expire(active.path)
        assert store.cleanup() == 0
        assert Path(active.path).exists()
    assert store.cleanup() == 0
    expire(active.path)
    assert store.cleanup() == 1
    assert not Path(active.path).exists()


def test_script_stop_or_rerun_releases_lease(store):
    class ScriptControlFlow(BaseException):
        pass

    with pytest.raises(ScriptControlFlow):
        with store.session() as session:
            raise ScriptControlFlow()
    expire(session.path)
    assert store.cleanup() == 1


def test_browser_expiry_checks_do_not_keep_session_alive(store):
    with store.session() as session:
        expire(session.path)
        assert not store.expired(session.token)  # active job still owns its files
    assert not store.expired(session.token)
    expire(session.path)
    marker = Path(session.path, ".last_activity")
    before = marker.stat().st_mtime
    assert store.expired(session.token)
    assert marker.stat().st_mtime == before
    store.clear(session.token)
    assert store.expired(session.token)


def test_clear_does_not_recreate_directory_or_affect_other_sessions(store):
    with store.session() as a, store.session() as b:
        Path(a.path, "private.txt").write_text("A")
        Path(b.path, "private.txt").write_text("B")
        store.clear(a.token)
        assert not Path(a.path).exists()
        assert Path(b.path, "private.txt").read_text() == "B"
    assert not Path(a.path).exists()
    assert Path(b.path).is_dir()


def test_cleanup_handles_interrupted_creation_and_ignores_unrelated_paths(store, tmp_path):
    interrupted = Path(store.root, "a" * 32)
    interrupted.mkdir()
    old = time.time() - 120
    os.utime(interrupted, (old, old))
    unrelated = Path(store.root, "not-a-session")
    unrelated.mkdir()
    os.utime(unrelated, (old, old))
    outside = tmp_path / "outside"
    outside.mkdir()
    Path(outside, "keep.txt").write_text("keep")
    Path(store.root, "b" * 32).symlink_to(outside, target_is_directory=True)
    Path(store.root, "c" * 32).write_text("unrelated regular file")
    assert store.cleanup() == 1
    assert not interrupted.exists()
    assert unrelated.exists()
    assert Path(outside, "keep.txt").read_text() == "keep"
    assert Path(store.root, "b" * 32).is_symlink()
    assert Path(store.root, "c" * 32).is_file()


def test_invalid_token_cannot_escape_root(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    for token in ("../outside", str(outside), "x" * 32, "a" * 32 + "\n"):
        with store.session(token) as session:
            assert session.reset and Path(session.path).parent == Path(store.root)
        with pytest.raises(ValueError):
            store.clear(token)
    assert outside.is_dir()


def test_root_symlink_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "sessions"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        SessionStore(link)


def test_background_cleanup_runs_without_another_visitor(tmp_path):
    manager = SessionStore(tmp_path / "sessions", ttl_seconds=0.05, cleanup_interval=0.01)
    try:
        with manager.session() as session:
            pass
        manager.start_cleanup()
        worker = manager._worker
        manager.start_cleanup()
        assert manager._worker is worker
        deadline = time.monotonic() + 2
        while Path(session.path).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(session.path).exists()
    finally:
        manager.close()
    assert not worker.is_alive()
