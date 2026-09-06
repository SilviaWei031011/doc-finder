"""Public upload boundaries: validate before embedding and leave no partial data."""
import io
from pathlib import Path

import pytest

import core
import demo


def pdf_bytes(pages=1, text="This is an English text document."):
    with core.fitz.open() as doc:
        for _ in range(pages):
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        return doc.tobytes()


def upload(data):
    value = io.BytesIO(data)
    value.name = "../paper.pdf"
    return value


def no_build(*args):
    pytest.fail("Invalid input must be rejected before model loading/indexing")


@pytest.mark.parametrize("data,error", [
    (b"", core.InvalidPDFError),
    (b"not a PDF", core.fitz.FileDataError),
    (b"x" * (demo.MAX_UPLOAD_BYTES + 1), core.PDFLimitError),
    (pdf_bytes(101), core.PDFLimitError),
    (pdf_bytes(text=""), core.NoTextError),
])
def test_rejected_upload_leaves_no_files(tmp_path, data, error):
    with pytest.raises(error):
        demo.upload_document(str(tmp_path), upload(data), "", no_build)
    assert list(tmp_path.iterdir()) == []


def test_page_limit_checked_before_extraction_and_embedding(tmp_path, monkeypatch):
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(pdf_bytes(101))
    monkeypatch.setattr(core.fitz.Page, "get_text", no_build)
    with pytest.raises(core.PDFLimitError, match="100"):
        core.build_corpus(str(pdf), str(tmp_path / "index"), None, max_pages=100)
    assert not (tmp_path / "index").exists()


def test_page_limit_boundary(tmp_path):
    pdf = tmp_path / "allowed.pdf"
    pdf.write_bytes(pdf_bytes(100))
    chunks, pages = core.chunk_pdf(str(pdf), max_pages=100, max_chunks=2500)
    assert pages == len(chunks) == 100


def test_chunk_limit_and_boundary_before_embedding(tmp_path, monkeypatch):
    data = pdf_bytes()
    # Simulate a dense page without committing a large test document.
    text = "\n".join(["x" * 300] * 2500)
    monkeypatch.setattr(core.fitz.Page, "get_text", lambda self: text)
    pdf = tmp_path / "dense.pdf"
    pdf.write_bytes(data)
    assert len(core.chunk_pdf(str(pdf), max_chunks=2500)[0]) == 2500
    text += "\n" + "x" * 300
    monkeypatch.setattr(core, "embed", no_build)
    with pytest.raises(core.PDFLimitError, match="2500"):
        core.build_corpus(str(pdf), str(tmp_path / "index"), None, max_chunks=2500)
    pdf.unlink()
    with pytest.raises(core.PDFLimitError):
        demo.upload_document(str(tmp_path), upload(data), "", no_build)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_interrupted_build_cleans_pdf_and_partial_index(tmp_path, error):
    def failed_build(pdf, folder):
        Path(folder, "index.faiss.tmp").write_bytes(b"partial")
        Path(folder, "meta.json").write_text("partial")
        raise error("interrupted")
    with pytest.raises(error):
        demo.upload_document(str(tmp_path), upload(pdf_bytes()), "", failed_build)
    assert list(tmp_path.iterdir()) == []


def test_registration_failure_rolls_back(tmp_path, monkeypatch):
    def failed_register(base, entry):
        Path(base, core.COLLECTIONS_FILENAME + ".tmp").write_text("partial")
        raise OSError("write failed")
    monkeypatch.setattr(core, "register_corpus", failed_register)
    with pytest.raises(OSError):
        demo.upload_document(str(tmp_path), upload(pdf_bytes()), "", fake_build)
    assert list(tmp_path.iterdir()) == []


def fake_build(pdf, folder):
    # Upload lifecycle tests do not need the actual encoder; retrieval has its
    # own full BGE-M3 integration tests in test_smoke.py.
    Path(folder, core.INDEX_FILENAME).write_bytes(b"index")
    Path(folder, core.META_FILENAME).write_text("{}")
    return core.BuildResult(1, 1, 0)


def test_three_document_limit_and_delete_frees_slot(tmp_path):
    for _ in range(3):
        entry, _ = demo.upload_document(str(tmp_path), upload(pdf_bytes()), "", fake_build)
        assert entry["original_filename"] == "paper.pdf"
        assert Path(tmp_path, entry["id"], core.PDF_FILENAME).is_file()
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    with pytest.raises(core.PDFLimitError, match="3"):
        demo.upload_document(str(tmp_path), upload(pdf_bytes()), "", no_build)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    core.delete_corpus(str(tmp_path), entry["id"])
    demo.upload_document(str(tmp_path), upload(pdf_bytes()), "", fake_build)
    assert len(core.load_collections(str(tmp_path))) == 3


def test_encrypted_upload_is_rejected(tmp_path):
    with core.fitz.open(stream=pdf_bytes(), filetype="pdf") as doc:
        data = doc.tobytes(encryption=core.fitz.PDF_ENCRYPT_AES_256,
                          owner_pw="owner", user_pw="reader")
    with pytest.raises(core.InvalidPDFError):
        demo.upload_document(str(tmp_path), upload(data), "", no_build)
    assert list(tmp_path.iterdir()) == []


def test_public_highlight_bounds_oversized_page_raster(tmp_path):
    pdf = tmp_path / "large-page.pdf"
    with core.fitz.open() as doc:
        page = doc.new_page(width=20000, height=20000)
        page.insert_text((72, 72), "A very large page.")
        doc.save(str(pdf))
    png = core.render_highlight(str(pdf), 1, "A very large page.", max_side=1600)
    pixmap = core.fitz.Pixmap(png)
    assert max(pixmap.width, pixmap.height) <= 1600
