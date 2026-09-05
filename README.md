# Doc Finder

**Find the English source passage—and its PDF page—behind a Chinese translation, paraphrase, or reading note.**

`v0.1` · early-stage research utility · runs locally · no external inference API

Doc Finder is a small research utility for people who read English sources and take notes or write in Chinese. Upload an English PDF, paste a Chinese translation, paraphrase, or reading note based on the document, and Doc Finder retrieves the most semantically similar passages from the original English text together with their PDF page numbers. It can also render the corresponding PDF page with the matched passage highlighted.

The tool does not translate the Chinese query into English. Instead, the Chinese query and English passages are embedded into the same multilingual vector space using [BGE-M3](https://huggingface.co/BAAI/bge-m3) and compared by cosine similarity. Cross-lingual semantic representations are provided by BGE-M3; Doc Finder implements the document-processing, indexing, retrieval, page-mapping, corpus-management, and user-facing workflow around it.

## What it does

- Upload an English, text-based PDF.
- Extract text page by page while preserving the PDF page associated with each text chunk.
- Split the document into approximately 300-character chunks.
- Encode document chunks and queries with BGE-M3.
- Build a local FAISS index for semantic retrieval.
- Enter a Chinese translation, paraphrase, or reading note and retrieve the top-k most similar English passages.
- Show the cosine similarity score and 1-based PDF page number for each result.
- Optionally render the PDF page with the matched passage highlighted.
- Maintain multiple local document corpora and switch between them from the Streamlit sidebar.

A roughly 300-page book takes about one minute to index on an Apple M1 Pro CPU in the current implementation.

## How it works

```text
English PDF
    │
    ▼
PyMuPDF page-by-page text extraction
    │
    ▼
Page-aware text chunks (≤ ~300 characters)
    │
    ▼
BGE-M3 embeddings
(CLS pooling + L2 normalization)
    │
    ▼
FAISS IndexFlatIP
    │
    │
Chinese query ── same encoder ──► query embedding
    │
    ▼
Cosine-similarity retrieval
    │
    ▼
Top-k English passages + PDF pages
    │
    ▼
Optional PyMuPDF text search + page highlight
```

| Step | Implementation | Where |
|---|---|---|
| PDF parsing | PyMuPDF `page.get_text()`, page by page | `core.extract_pages` |
| Chunking | Consecutive text lines are joined into chunks of up to roughly 300 characters while preserving the source page | `core.split_text_to_chunks`, `core.chunk_pdf` |
| Embedding | `BAAI/bge-m3` loaded as Transformer → CLS pooling → L2 normalization | `core.load_model`, `core.embed` |
| Retrieval | Exact inner-product search over normalized vectors using `faiss.IndexFlatIP`, equivalent to cosine-similarity ranking | `core.search` |
| Page mapping | Chunk-to-page metadata is stored alongside the FAISS index | `core.build_corpus` |
| Display and highlighting | Streamlit UI with PyMuPDF-based text location, highlighting, and page rendering | `app.py`, `core.render_highlight` |

Documents and queries pass through the same `core.embed` function. Each index records metadata about the embedding configuration used to build it. The application detects incompatible index settings such as pooling, format version, or vector dimension and requires a rebuild before searching. A difference in model identifier alone is reported as a warning, since the same model may be referenced by either a Hugging Face ID or a local path.

## Installation

### Requirements

- Python 3.9 or newer
- Python 3.11 recommended for a new environment
- Approximately 3 GB of free disk space for BGE-M3
- Approximately 3 GB of available RAM
- No GPU required

The project has been developed and tested primarily on macOS with Apple Silicon. Linux is expected to work but has not yet been formally tested.

```bash
git clone <this-repository-url>
cd doc_finder

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

By default, the first run downloads `BAAI/bge-m3` from the Hugging Face Hub. The model is approximately 2.3 GB and is stored in the standard Hugging Face cache. Later runs can use the cached model offline.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DOC_FINDER_MODEL` | `BAAI/bge-m3` | Hugging Face model ID or local model directory |
| `DOC_FINDER_CORPUS_DIR` | `./corpora` | Directory used for uploaded PDFs, metadata, and FAISS indexes |
| `DOC_FINDER_DEVICE` | `cpu` | PyTorch device used for embedding |

Example using a local copy of BGE-M3:

```bash
DOC_FINDER_MODEL=/path/to/bge-m3 streamlit run app.py
```

Other compatible transformer models may be used, but indexes are tied to the model configuration with which they were created. The current pooling configuration is designed for BGE-M3, so other models may require code changes rather than only changing the model name.

## Command-line indexing

`build_index.py` exposes the same indexing pipeline without the Streamlit interface.

Index a PDF into a specified directory:

```bash
python build_index.py paper.pdf --out /tmp/paper_index
```

Rebuild an existing corpus:

```bash
python build_index.py --rebuild <corpus_id>
```

Rebuild all local corpora:

```bash
python build_index.py --rebuild-all
```

## Data and privacy

Doc Finder is designed to run locally.

When a PDF is uploaded, the application stores a local copy together with its extracted text metadata and FAISS index under the configured corpus directory. Document text and queries are not sent to an external inference API.

Network access is normally required only to download the embedding model when it is not already available locally. Streamlit anonymous usage statistics are disabled in `.streamlit/config.toml`.

The local corpus directory is excluded from Git version control. Uploaded PDFs, extracted document text, FAISS indexes, local model files, environment files, and secrets are not intended to be committed to this repository.

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the smoke tests:

```bash
DOC_FINDER_MODEL=/path/to/bge-m3 python -m pytest tests -q
```

If `DOC_FINDER_MODEL` is omitted, the model may be downloaded automatically.

The test suite includes a generated multi-page PDF, index construction, Chinese-to-English retrieval with page verification, empty-document handling, oversized `top_k`, legacy-index detection, malformed corpus IDs, and Streamlit UI behavior.

## Limitations

Doc Finder is an early-stage research utility. Please verify retrieved text and page information against the original document before using them in academic citations.

- **Text-based PDFs only.** Scanned or image-only PDFs are not currently supported. There is no OCR pipeline.
- **PDF page numbers, not printed page labels.** Page 1 means the first physical page of the PDF file. In books with front matter, this may differ from the page number printed on the page.
- **Chunk-level rather than sentence-level retrieval.** Text is currently divided into approximately 300-character chunks. A result may contain the desired sentence together with surrounding text, and a sentence can occasionally cross chunk boundaries.
- **PDF extraction artifacts remain.** Running headers, page numbers, line-break artifacts, and hyphenation may appear in extracted text.
- **Loose paraphrases are harder.** In small informal tests, relatively faithful translations were generally retrieved successfully, while short or substantially rewritten notes could rank the relevant passage much lower.
- **No calibrated “no match” decision yet.** The system always returns the nearest passages. A high-ranking result does not guarantee that the intended source is actually present in the document.
- **No formal retrieval benchmark yet.** Current behavior has been checked with smoke tests and a small number of manually constructed examples rather than a dedicated evaluation dataset.
- **Local single-user design.** The current storage architecture is intended for local use. It does not yet provide authentication, per-user isolation, quotas, or automatic data expiration and should not be exposed unchanged as a public multi-user service.
- **Resource requirements.** BGE-M3 requires roughly 2.5 GB of RAM in the current setup. Indexing runs on the CPU and may take around one minute for a 300-page document on an Apple M1 Pro.
- **English source documents are the current target.** BGE-M3 is multilingual, so other source languages may work, but they have not yet been evaluated.

## Roadmap

Potential next steps include:

- sentence-aware chunking;
- de-hyphenation and cleaner PDF text normalization;
- detection of printed page numbers or page labels;
- a small public Chinese-to-English source-localization evaluation set;
- Recall@1 / Recall@5 reporting;
- calibrated no-match detection;
- per-session document isolation and automatic cleanup for web deployment.

## Third-party components

Doc Finder builds on established open-source components rather than training a new embedding model. Its contribution is the application pipeline for page-aware, cross-lingual source localization in PDFs: document processing, page-aware chunking, embedding and index orchestration, corpus management, retrieval workflow, result presentation, and PDF-page highlighting.

| Component | Used for | License | Reference |
|---|---|---|---|
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) (BAAI) | Multilingual embeddings and cross-lingual semantic representation | MIT | Chen et al., 2024 |
| [sentence-transformers](https://www.sbert.net/) | Model loading and text encoding | Apache-2.0 | Reimers and Gurevych, 2019 |
| [PyTorch](https://pytorch.org/) | Model inference | BSD-3-Clause | |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Transformer model infrastructure | Apache-2.0 | |
| [FAISS](https://github.com/facebookresearch/faiss) (Meta) | Vector indexing and similarity search | MIT | Douze et al., 2024 |
| [PyMuPDF](https://pymupdf.readthedocs.io/) (Artifex) | PDF text extraction, text search, highlighting, and page rendering | AGPL-3.0 or commercial | |
| [Streamlit](https://streamlit.io/) | Local web interface | Apache-2.0 | |
| [NumPy](https://numpy.org/) | Array handling | BSD-3-Clause | |

### BGE-M3

If BGE-M3 is relevant to academic work using this tool, please refer to the original model paper:

```bibtex
@misc{bge-m3,
  title         = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  author        = {Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
  year          = {2024},
  eprint        = {2402.03216},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## License

Doc Finder is currently distributed under the [GNU Affero General Public License v3.0](LICENSE).

This licensing choice reflects the project's current use of the AGPL-licensed edition of PyMuPDF. Other major dependencies listed above use permissive open-source licenses such as MIT, BSD, and Apache-2.0.

If the PDF-processing backend is replaced with a permissively licensed alternative in a future version, the licensing options for future versions of Doc Finder may be reconsidered.

## 中文简介

Doc Finder 是一个面向研究者的本地文献辅助工具。

用户上传英文 PDF，再输入一段根据该文献写下的中文翻译、转述或阅读笔记，程序会利用 BGE-M3 的多语言语义表示，在英文原文中检索最相关的文本片段，并返回对应的 PDF 页码。用户还可以查看高亮后的 PDF 页面，以便在论文写作或引用时快速核对原文和页码。

Doc Finder 不先把中文翻译成英文，而是直接将中文查询和英文文本映射到同一个多语言向量空间中进行语义匹配。

当前 v0.1 主要支持文本型英文 PDF。扫描版 PDF、印刷页码识别、句子级定位、严格的“未找到”判断以及公开多用户部署仍属于后续工作。
