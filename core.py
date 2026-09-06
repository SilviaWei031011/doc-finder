"""doc_finder core pipeline (no Streamlit dependency).

PDF ──PyMuPDF──▶ per-page text ──▶ fixed-size chunks (≤300 chars, page number kept)
    ──BGE-M3 (CLS pooling, L2-normalised)──▶ FAISS inner-product index
query ──same encoder──▶ top-k chunks with page numbers ──PyMuPDF──▶ highlighted page image

Configuration (environment variables):
  DOC_FINDER_MODEL       sentence-transformers model id or local directory (default: BAAI/bge-m3)
  DOC_FINDER_DEVICE      torch device for the encoder (default: cpu)
  DOC_FINDER_CORPUS_DIR  where uploaded PDFs and indexes are stored (default: ./corpora next to this file)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# IMPORTANT: import torch (through sentence_transformers) BEFORE faiss.
# faiss-cpu and torch each ship their own OpenMP runtime (libomp.dylib). On macOS, importing
# faiss first makes the encoder's first forward pass segfault inside libomp
# (reproduced 2026-09-05 on an M1 Pro with torch 2.8.0 / faiss-cpu 1.13.0).
import torch  # noqa: F401  (must stay above `import faiss`)
from sentence_transformers import SentenceTransformer, models

import faiss  # noqa: E402
import fitz  # PyMuPDF  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-m3"
POOLING = "cls"          # same as BAAI/bge-m3's 1_Pooling/config.json (pooling_mode_cls_token = true)
MAX_SEQ_LENGTH = 1024    # tokens; chunks are ≤300 characters and queries are short
CHUNK_MAX_CHARS = 300
BATCH_SIZE = 32
FORMAT_VERSION = 2       # meta.json layout; 1 = the pre-2026-09 list layout (built with mean pooling)

PDF_FILENAME = "document.pdf"
INDEX_FILENAME = "index.faiss"
META_FILENAME = "meta.json"
COLLECTIONS_FILENAME = "collections.json"
_CORPUS_ID_RE = re.compile(r"^[0-9a-f]{8}$")

ProgressCallback = Callable[[int, int], None]


class NoTextError(ValueError):
    """The PDF contains no extractable text (scanned / image-only PDF)."""


class PDFLimitError(ValueError):
    """A PDF exceeds a configured public-demo limit."""


class InvalidPDFError(ValueError):
    """An upload is not an unlocked PDF."""


class RegistryError(RuntimeError):
    """collections.json exists but cannot be read."""


# ----------------------------------------------------------------------------- configuration

def model_name() -> str:
    return os.environ.get("DOC_FINDER_MODEL", "").strip() or DEFAULT_MODEL


def corpus_dir() -> str:
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora")
    return os.environ.get("DOC_FINDER_CORPUS_DIR", "").strip() or default


# ----------------------------------------------------------------------------- model

def load_model(name_or_path: Optional[str] = None, device: Optional[str] = None) -> SentenceTransformer:
    """Load BGE-M3 (or a compatible encoder) as an explicit, fixed pipeline:
    Transformer -> CLS pooling -> L2 normalisation.

    The pipeline is built explicitly instead of relying on the model directory's modules.json,
    so a local copy of the weights that contains only the transformer files still uses the
    correct pooling. For BAAI/bge-m3 this is identical to the official configuration.
    """
    name = name_or_path or model_name()
    device = device or os.environ.get("DOC_FINDER_DEVICE", "").strip() or "cpu"
    transformer = models.Transformer(name, max_seq_length=MAX_SEQ_LENGTH)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=POOLING)
    model = SentenceTransformer(modules=[transformer, pooling, models.Normalize()], device=device)
    model.doc_finder_name = name  # recorded in meta.json for compatibility checks
    return model


def embed(model: SentenceTransformer, texts: List[str], batch_size: int = BATCH_SIZE,
          progress_cb: Optional[ProgressCallback] = None) -> np.ndarray:
    """Encode texts to L2-normalised float32 vectors. Used for both documents and queries."""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    parts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        parts.append(model.encode(batch, batch_size=batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False))
        if progress_cb:
            progress_cb(min(start + batch_size, len(texts)), len(texts))
    return np.vstack(parts).astype("float32")


# ----------------------------------------------------------------------------- text

def split_text_to_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """Greedy line accumulation: consecutive non-empty lines are joined with a space until the
    chunk would exceed max_chars. A single line longer than max_chars becomes its own chunk."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: List[str] = []
    current = ""
    for ln in lines:
        if len(current) + len(ln) + 1 <= max_chars:
            current = f"{current} {ln}" if current else ln
        else:
            if current:
                chunks.append(current)
            current = ln
    if current:
        chunks.append(current)
    return chunks


def extract_pages(pdf_path: str) -> List[str]:
    with fitz.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def chunk_pdf(pdf_path: str, max_pages: Optional[int] = None,
              max_chunks: Optional[int] = None):
    """Return (chunks, num_pages); each chunk is {"page": 1-based page number, "text": str}."""
    chunks = []
    with fitz.open(pdf_path) as doc:
        if not doc.is_pdf or doc.needs_pass:
            raise InvalidPDFError("请上传有效且未加密的 PDF 文件。")
        num_pages = len(doc)
        if max_pages is not None and num_pages > max_pages:
            raise PDFLimitError(f"公开演示每份 PDF 最多 {max_pages} 页。")
        for i, page in enumerate(doc):
            for text in split_text_to_chunks(page.get_text()):
                if max_chunks is not None and len(chunks) >= max_chunks:
                    raise PDFLimitError(f"公开演示每份 PDF 最多 {max_chunks} 个文本片段，请缩短文档。")
                chunks.append({"page": i + 1, "text": text})
    return chunks, num_pages


# ----------------------------------------------------------------------------- index build / load

@dataclass
class BuildResult:
    num_pages: int
    num_chunks: int
    seconds: float


def _paths(folder: str):
    return os.path.join(folder, INDEX_FILENAME), os.path.join(folder, META_FILENAME)


def corpus_files_exist(folder: str) -> bool:
    return all(os.path.isfile(p) for p in _paths(folder))


def corpus_mtime(folder: str) -> float:
    return max(os.path.getmtime(p) for p in _paths(folder))


def build_corpus(pdf_path: str, out_dir: str, model: SentenceTransformer,
                 progress_cb: Optional[ProgressCallback] = None, *,
                 max_pages: Optional[int] = None,
                 max_chunks: Optional[int] = None) -> BuildResult:
    """Extract, chunk, embed and index one PDF; writes index.faiss + meta.json into out_dir.
    Files are written to temporary names first, so a failure never leaves a half-written index."""
    t0 = time.time()
    chunks, num_pages = chunk_pdf(pdf_path, max_pages=max_pages, max_chunks=max_chunks)
    if not chunks:
        raise NoTextError("PDF 中没有可提取的文本（可能是扫描件或纯图片 PDF）")
    vectors = embed(model, [c["text"] for c in chunks], progress_cb=progress_cb)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    meta = {
        "format_version": FORMAT_VERSION,
        "model": getattr(model, "doc_finder_name", model_name()),
        "pooling": POOLING,
        "normalize": True,
        "dim": int(vectors.shape[1]),
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }
    os.makedirs(out_dir, exist_ok=True)
    index_path, meta_path = _paths(out_dir)
    faiss.write_index(index, index_path + ".tmp")
    with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(index_path + ".tmp", index_path)
    os.replace(meta_path + ".tmp", meta_path)
    return BuildResult(num_pages=num_pages, num_chunks=len(chunks), seconds=time.time() - t0)


@dataclass
class Corpus:
    index: "faiss.Index"
    chunks: List[Dict]
    meta: Dict


def load_corpus(folder: str) -> Corpus:
    index_path, meta_path = _paths(folder)
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):  # legacy layout written before FORMAT_VERSION existed
        meta = {"format_version": 1, "chunks": raw}
    else:
        meta = raw
    return Corpus(index=index, chunks=meta["chunks"], meta=meta)


def compatibility_error(corpus: Corpus, model: SentenceTransformer) -> Optional[str]:
    """Why `corpus` cannot be searched with `model` (needs a rebuild), or None if it can."""
    m = corpus.meta
    if m.get("format_version", 1) < FORMAT_VERSION:
        return "此索引由旧版本程序建立（mean pooling），与当前的编码方式不一致"
    if m.get("pooling") != POOLING or not m.get("normalize", False):
        return f"此索引的 pooling 方式（{m.get('pooling')}）与当前程序（{POOLING}）不一致"
    if m.get("dim") != corpus.index.d or corpus.index.ntotal != len(corpus.chunks):
        return "索引文件与 meta.json 不一致（可能已损坏）"
    if corpus.index.d != model.get_sentence_embedding_dimension():
        return f"索引维度 {corpus.index.d} 与当前模型维度 {model.get_sentence_embedding_dimension()} 不一致"
    return None


def model_mismatch_note(corpus: Corpus, model: SentenceTransformer) -> Optional[str]:
    """Soft warning when the index was built with a differently named model."""
    built_with = corpus.meta.get("model")
    current = getattr(model, "doc_finder_name", model_name())
    if built_with and built_with != current:
        return (f"此索引由模型「{built_with}」建立，当前加载的是「{current}」。"
                "若两者是同一模型的不同路径可忽略，否则请删除后重新上传。")
    return None


# ----------------------------------------------------------------------------- search

@dataclass
class Hit:
    rank: int
    score: float
    page: int
    text: str
    chunk_id: int


def search(corpus: Corpus, model: SentenceTransformer, query: str, top_k: int) -> List[Hit]:
    query = query.strip()
    if not query or corpus.index.ntotal == 0:
        return []
    k = max(1, min(int(top_k), corpus.index.ntotal))
    scores, ids = corpus.index.search(embed(model, [query]), k)
    hits: List[Hit] = []
    for score, idx in zip(scores[0], ids[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(corpus.chunks):  # FAISS pads with -1 when fewer than k vectors exist
            continue
        c = corpus.chunks[idx]
        hits.append(Hit(rank=len(hits) + 1, score=float(score), page=int(c["page"]),
                        text=c["text"], chunk_id=idx))
    return hits


# ----------------------------------------------------------------------------- highlight

def render_highlight(pdf_path: str, page_number: int, text: str, dpi: int = 110,
                     max_side: Optional[int] = None) -> bytes:
    """Return a PNG of `page_number` (1-based) with `text` highlighted. The PDF is not modified."""
    with fitz.open(pdf_path) as doc:
        if not 1 <= page_number <= len(doc):
            raise ValueError(f"页码 {page_number} 超出范围（共 {len(doc)} 页）")
        page = doc[page_number - 1]
        needle = " ".join(text.split())
        rects = page.search_for(needle) if needle else []
        if not rects and len(needle) > 80:  # fall back to the first ~80 characters, cut at a word boundary
            short = needle[:80].rsplit(" ", 1)[0] or needle[:80]
            rects = page.search_for(short)
        for r in rects:
            page.add_highlight_annot(r)
        if max_side is not None:
            scale = min(dpi / 72, max_side / max(page.rect.width, page.rect.height))
            return page.get_pixmap(matrix=fitz.Matrix(scale, scale)).tobytes("png")
        return page.get_pixmap(dpi=dpi).tobytes("png")


# ----------------------------------------------------------------------------- registry (collections.json)

def is_valid_corpus_id(value) -> bool:
    return isinstance(value, str) and bool(_CORPUS_ID_RE.match(value))


def corpus_folder(base: str, corpus_id: str) -> str:
    """Folder of a corpus; refuses anything that is not a plain 8-hex id (no path tricks)."""
    if not is_valid_corpus_id(corpus_id):
        raise ValueError(f"非法的文献 ID：{corpus_id!r}")
    return os.path.join(base, corpus_id)


def new_corpus_id(base: str) -> str:
    while True:
        cid = uuid.uuid4().hex[:8]
        if not os.path.exists(os.path.join(base, cid)):
            return cid


def sanitize_display_name(name: str, max_len: int = 120) -> str:
    """Display-only name: basename, printable characters only, length-capped."""
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:max_len] or "未命名文献"


def load_collections(base: str) -> List[Dict]:
    path = os.path.join(base, COLLECTIONS_FILENAME)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise RegistryError(f"无法读取 {path}：{e}") from e
    if not isinstance(data, list):
        raise RegistryError(f"{path} 的内容不是列表")
    return [c for c in data
            if isinstance(c, dict) and is_valid_corpus_id(c.get("id")) and is_valid_corpus_id(c.get("folder"))]


def save_collections(base: str, items: List[Dict]) -> None:
    path = os.path.join(base, COLLECTIONS_FILENAME)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)


def register_corpus(base: str, entry: Dict) -> None:
    items = load_collections(base)
    items.append(entry)
    save_collections(base, items)


def delete_corpus(base: str, corpus_id: str) -> None:
    folder = corpus_folder(base, corpus_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    save_collections(base, [c for c in load_collections(base) if c.get("id") != corpus_id])
