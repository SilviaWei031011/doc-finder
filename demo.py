"""Public-demo upload limits and rollback; no Streamlit or global user-data cache."""
import os
import shutil
import threading
import time

import core

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PAGES = 100
MAX_CHUNKS = 2500
MAX_DOCUMENTS = 3

# Shared by every Streamlit session in this process. Queries use the same lock
# so model inference cannot overlap an index build or another query.
MODEL_LOCK = threading.Lock()


def upload_document(base, uploaded, custom_name, build):
    """Validate before loading the model; roll back all files on any failure.

    `build(pdf_path, folder)` is supplied by the UI to display lock/progress state.
    The caller holds its session's operation lease for this whole transaction.
    """
    if len(core.load_collections(base)) >= MAX_DOCUMENTS:
        raise core.PDFLimitError(f"每个会话最多保留 {MAX_DOCUMENTS} 份文献，请先删除或清空。")
    data = uploaded.getbuffer()
    if data.nbytes > MAX_UPLOAD_BYTES:
        raise core.PDFLimitError("每份 PDF 最大 25 MB，请缩小文件后重试。")
    if not data.nbytes:
        raise core.InvalidPDFError("文件为空，请上传有效的 PDF。")

    cid = core.new_corpus_id(base)
    folder = core.corpus_folder(base, cid)
    original_name = core.sanitize_display_name(uploaded.name)
    name = core.sanitize_display_name(custom_name) if custom_name.strip() else original_name
    os.mkdir(folder, mode=0o700)
    try:
        pdf_path = os.path.join(folder, core.PDF_FILENAME)
        with open(pdf_path, "wb") as f:
            f.write(data)
        # Reject page/chunk limits and scans before an expensive model load.
        chunks, _ = core.chunk_pdf(pdf_path, max_pages=MAX_PAGES, max_chunks=MAX_CHUNKS)
        if not chunks:
            raise core.NoTextError("PDF 中没有可提取的文本（可能是扫描件或纯图片 PDF）")
        del chunks
        result = build(pdf_path, folder)
        entry = {
            "id": cid, "name": name, "pdf_filename": core.PDF_FILENAME,
            "original_filename": original_name, "folder": cid,
            "created_at": time.time(), "num_pages": result.num_pages,
            "num_chunks": result.num_chunks,
        }
        core.register_corpus(base, entry)
        return entry, result
    except BaseException:
        # Includes Streamlit stop/rerun interruptions during model loading.
        shutil.rmtree(folder)
        registry_tmp = os.path.join(base, core.COLLECTIONS_FILENAME + ".tmp")
        if os.path.isfile(registry_tmp):
            os.remove(registry_tmp)
        raise
