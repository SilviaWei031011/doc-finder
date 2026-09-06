"""Smoke test for doc_finder: PDF -> index -> Chinese query -> English chunk + page, plus the Streamlit UI.

Run:   DOC_FINDER_MODEL=/path/to/bge-m3  python -m pytest tests -q
Without DOC_FINDER_MODEL the default BAAI/bge-m3 is downloaded from the Hugging Face Hub (~2.3 GB).
The test PDF is generated on the fly from original sentences written for this test.
"""
import os
import shutil
import sys
import time

import fitz
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core  # noqa: E402

PAGES = [
    [  # page 1
        "The lighthouse keeper counted the fishing boats every morning before the fog lifted from the harbour. "
        "He wrote the numbers in a green notebook that his daughter had given him.",
        "Later that spring, the town council voted to replace the old oil lamp with an electric beacon, "
        "and the keeper wondered whether anyone would still need him.",
    ],
    [  # page 2
        "The botanist kept a small greenhouse behind the railway station, where she grew tomatoes that "
        "ripened two weeks earlier than anyone else's.",
        "Her neighbours suspected a secret fertiliser, but the real reason was simpler: she talked to the "
        "plants every evening while the trains went past.",
    ],
    [  # page 3
        "On the last day of the exhibition, the museum guard discovered that the painting of the blue horse "
        "had been hanging upside down for eleven years.",
        "Nobody had complained, and the curator decided, after a long silence, to leave it exactly as it was.",
    ],
]

# (Chinese query, expected 1-based page, English substring expected in the top hit)
QUERIES = [
    ("灯塔看守人每天早晨在雾从港口散去之前清点渔船的数量。", 1, "counted the fishing boats"),
    ("植物学家在火车站后面有一个小温室，她种的番茄比别人早两周成熟。", 2, "greenhouse"),
    ("展览的最后一天，博物馆保安发现那幅蓝马的画已经倒挂了十一年。", 3, "upside down"),
]


def make_pdf(path: str, pages) -> None:
    doc = fitz.open()
    for paragraphs in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 523, 770), "\n\n".join(paragraphs), fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


@pytest.fixture(scope="session")
def model():
    return core.load_model()


@pytest.fixture(scope="session")
def corpus_env(tmp_path_factory, model):
    base = tmp_path_factory.mktemp("corpora")
    cid = "0123abcd"
    folder = base / cid
    folder.mkdir()
    pdf = folder / core.PDF_FILENAME
    make_pdf(str(pdf), PAGES)
    result = core.build_corpus(str(pdf), str(folder), model)
    core.save_collections(str(base), [{
        "id": cid, "name": "Smoke Test Book", "pdf_filename": core.PDF_FILENAME,
        "folder": cid, "created_at": time.time(),
    }])
    return {"base": str(base), "id": cid, "folder": str(folder), "pdf": str(pdf), "result": result}


# ----------------------------------------------------------------------------- pipeline

def test_split_text_to_chunks():
    text = "\n".join(["word " * 20] * 10)  # 10 lines of 100 chars
    chunks = core.split_text_to_chunks(text, max_chars=300)
    assert chunks and all(len(c) <= 300 for c in chunks)
    assert core.split_text_to_chunks("") == []
    assert core.split_text_to_chunks("\n  \n") == []


def test_model_uses_cls_pooling_and_normalises(model):
    assert model[1].get_pooling_mode_str() == "cls"
    v = core.embed(model, ["hello world"])
    assert v.shape == (1, model.get_sentence_embedding_dimension())
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-4


def test_build_writes_v2_meta(corpus_env):
    corpus = core.load_corpus(corpus_env["folder"])
    assert corpus.meta["format_version"] == core.FORMAT_VERSION
    assert corpus.meta["pooling"] == "cls" and corpus.meta["normalize"] is True
    assert corpus.index.ntotal == len(corpus.chunks) == corpus_env["result"].num_chunks
    assert corpus_env["result"].num_pages == 3
    assert {c["page"] for c in corpus.chunks} == {1, 2, 3}


def test_chinese_query_returns_english_chunk_and_correct_page(corpus_env, model):
    corpus = core.load_corpus(corpus_env["folder"])
    assert core.compatibility_error(corpus, model) is None
    for zh, page, substring in QUERIES:
        hits = core.search(corpus, model, zh, top_k=3)
        assert hits, zh
        assert hits[0].page == page, (zh, hits[0])
        assert substring in hits[0].text, (zh, hits[0])
        assert hits[0].score > 0.5


def test_english_exact_query(corpus_env, model):
    corpus = core.load_corpus(corpus_env["folder"])
    hits = core.search(corpus, model, corpus.chunks[0]["text"], top_k=1)
    assert hits[0].chunk_id == 0 and hits[0].score > 0.99


def test_top_k_is_clamped_and_ids_valid(corpus_env, model):
    corpus = core.load_corpus(corpus_env["folder"])
    hits = core.search(corpus, model, "任意查询", top_k=500)
    assert len(hits) == corpus.index.ntotal
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(0 <= h.chunk_id < len(corpus.chunks) for h in hits)
    assert core.search(corpus, model, "   ", top_k=5) == []


def test_empty_pdf_raises_and_leaves_no_files(tmp_path, model):
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    with pytest.raises(core.NoTextError):
        core.build_corpus(str(pdf), str(tmp_path), model)
    assert not core.corpus_files_exist(str(tmp_path))


def test_legacy_index_is_detected(tmp_path, model):
    import faiss
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    index.add(np.zeros((2, index.d), dtype="float32"))
    faiss.write_index(index, str(tmp_path / core.INDEX_FILENAME))
    (tmp_path / core.META_FILENAME).write_text('[{"page": 1, "text": "a"}, {"page": 1, "text": "b"}]')
    corpus = core.load_corpus(str(tmp_path))
    assert corpus.meta["format_version"] == 1
    assert core.compatibility_error(corpus, model) is not None


def test_render_highlight_returns_png(corpus_env):
    corpus = core.load_corpus(corpus_env["folder"])
    png = core.render_highlight(corpus_env["pdf"], corpus.chunks[0]["page"], corpus.chunks[0]["text"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_registry_rejects_bad_ids(tmp_path):
    with pytest.raises(ValueError):
        core.corpus_folder(str(tmp_path), "../etc")
    core.save_collections(str(tmp_path), [{"id": "../x", "folder": "../x"}, {"id": "0123abcd", "folder": "0123abcd"}])
    assert [c["id"] for c in core.load_collections(str(tmp_path))] == ["0123abcd"]


# ----------------------------------------------------------------------------- Streamlit UI (AppTest)

def test_streamlit_ui_select_search_highlight(corpus_env, model, monkeypatch, tmp_path):
    import streamlit as st
    import session_storage
    from streamlit.testing.v1 import AppTest

    store = session_storage.SessionStore(tmp_path / "sessions")
    monkeypatch.setattr(session_storage, "SessionStore", lambda: store)
    # Reuse the actual BGE-M3 fixture, not a second multi-GB model in the UI cache.
    monkeypatch.setattr(core, "load_model", lambda *args, **kwargs: model)
    st.cache_resource.clear()
    # The legacy local corpus directory must not become visible in the web app.
    monkeypatch.setenv("DOC_FINDER_CORPUS_DIR", corpus_env["base"])
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("请在左侧选择" in i.value for i in at.info)
    assert not at.expander
    with store.session(at.session_state["session_token"]) as a:
        shutil.copytree(corpus_env["base"], a.path, dirs_exist_ok=True)
    at.run()

    at.button(key=f"choose_{corpus_env['id']}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Smoke Test Book" in s.value for s in at.success)

    zh, page, substring = QUERIES[2]
    at.text_area(key="query").input(zh).run()
    at.button(key="search").click().run()
    assert not at.exception, [e.value for e in at.exception]
    md = "\n".join(m.value for m in at.markdown)
    assert "候选 1" in md and f"第 {page} 页" in md
    assert any(substring in str(getattr(el, "value", "")) for el in at.main)

    # highlight checkbox lives outside the search button, so it must work on the next rerun
    at.checkbox(key="hl_1_1").check().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("imgs")) == 1

    # results and the highlight survive further interactions (e.g. moving the slider)
    at.slider(key="top_k").set_value(2).run()
    assert not at.exception, [e.value for e in at.exception]
    md = "\n".join(m.value for m in at.markdown)
    assert "候选 1" in md and len(at.get("imgs")) == 1

    # Independent browser session cannot list or select the first one's corpus,
    # even when a stale/forged selection ID is supplied.
    other = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600).run()
    assert not other.exception
    assert other.session_state["session_token"] != at.session_state["session_token"]
    other.session_state["selected_corpus_id"] = corpus_env["id"]
    other.run()
    assert not other.expander and not other.text_area
    assert other.session_state["selected_corpus_id"] is None
    with store.session(other.session_state["session_token"]) as b:
        shutil.copytree(corpus_env["base"], b.path, dirs_exist_ok=True)
    other.run()

    # Clear resets PDF uploader generation and all result/query/highlight state;
    # another session's files and the old CLI corpus survive.
    old_generation = at.session_state["upload_generation"]
    at.button(key="clear_documents").click().run()
    assert not at.exception
    assert not os.path.exists(a.path)
    assert os.path.isfile(os.path.join(b.path, core.COLLECTIONS_FILENAME))
    assert os.path.isfile(corpus_env["pdf"])
    assert at.session_state["results"] is None and not at.get("imgs")
    assert at.session_state["upload_generation"] > old_generation
    assert not at.expander and not at.text_area

    # Re-entering an expired session discards all stale selections and results.
    other.button(key=f"choose_{corpus_env['id']}").click().run()
    other.session_state["results"] = {"corpus_id": corpus_env["id"], "hits": []}
    store.cleanup(now=time.time() + store.ttl_seconds + 1)
    other.run()
    assert not other.exception and not other.expander
    assert other.session_state["results"] is None
    assert any("已过期" in msg.value for msg in other.info)
    visible = "\n".join(str(getattr(el, "value", "")) for el in other)
    assert other.session_state["session_token"] not in visible
    assert str(store.root) not in visible
    store.close()
    st.cache_resource.clear()


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", __file__]))
