# Doc Finder

**Find the English source passage—and its PDF page—behind a Chinese translation, paraphrase, or reading note.**

**在线使用 / Live app：[打开 Doc Finder](https://doc-finder.streamlit.app/)**

`v0.1` · early-stage research utility · local or hosted Streamlit interface · no external inference API

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
- Keep up to three documents in a private temporary browser session and switch between them from the Streamlit sidebar.

The Streamlit interface accepts text-based PDFs up to 25 MB, 100 pages, and 2,500 extracted chunks per document. Larger local jobs can use the command-line indexer; a roughly 300-page book previously took about one minute to index on an Apple M1 Pro CPU.

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
git clone https://github.com/SilviaWei031011/doc-finder.git
cd doc-finder

python3.11 -m venv .venv
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

By default, the first indexing or search operation downloads `BAAI/bge-m3` from the Hugging Face Hub. The model is approximately 2.3 GB and is stored in the standard Hugging Face cache. Later runs can use the cached model offline.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DOC_FINDER_MODEL` | `BAAI/bge-m3` | Hugging Face model ID or local model directory |
| `DOC_FINDER_CORPUS_DIR` | `./corpora` | Persistent corpus directory for command-line use only; the Streamlit app ignores this setting |
| `DOC_FINDER_DEVICE` | `cpu` | PyTorch device used for embedding |

Example using a local copy of BGE-M3:

```bash
DOC_FINDER_MODEL=/path/to/bge-m3 streamlit run app.py
```

Other compatible transformer models may be used, but indexes are tied to the model configuration with which they were created. The current pooling configuration is designed for BGE-M3, so other models may require code changes rather than only changing the model name.

## Deploying on Streamlit Community Cloud

The repository is prepared for a free Community Cloud deployment. Deployment and successful model inference on Community Cloud have not yet been verified.

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app** → **Yup, I have an app**.
2. Set the repository to `SilviaWei031011/doc-finder`, branch to `main`, and main file path to `app.py`.
3. Request the `doc-finder` subdomain if it is available; otherwise choose another available subdomain.
4. Open **Advanced settings**, select **Python 3.11**, and save. No secrets or environment variables are required: the defaults are `BAAI/bge-m3` and CPU inference.
5. Choose **Deploy** and inspect the build logs. The first model use downloads approximately 2.3 GB from Hugging Face.
6. Verify PDF upload, indexing, Chinese-query retrieval, page highlighting, separate browser sessions, and the session clear button before sharing the app.

Community Cloud reads the root `requirements.txt` and `.streamlit/config.toml`. The Linux dependency selects CPU-only PyTorch; macOS keeps its existing PyTorch version. Python is selected in the deployment dialog. See the official [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

BGE-M3 may exceed the free service's available memory. Streamlit's [published resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#resource-limits) give an approximate 690 MB–2.7 GB RAM range, explicitly dated February 2024 and subject to change. The model already uses roughly 2.5 GB in the local setup, before upload and inference overhead. CPU-only dependencies, upload limits, and serialized inference reduce overhead but do not guarantee that the model will fit. If model loading or a small indexing job exhausts memory, this deployment needs more memory or a separately evaluated model change.

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

In the Streamlit interface, each browser session receives a separate temporary document directory. Uploaded PDFs, extracted text, indexes, results, and rendered pages belong to that session; the sidebar does not expose another session's documents. The app ignores `DOC_FINDER_CORPUS_DIR`, so it does not expose persistent command-line corpora.

Use the session clear button to remove the current session's documents immediately. Inactive session directories expire approximately 60 minutes after their last interaction. A background sweep checks every 60 seconds while the process is running, and active processing is protected from cleanup. An open page checks expiry every 60 seconds without renewing activity, then clears stale results and upload state. Closing a browser tab does not immediately delete its files; timers may pause while a browser or host is suspended. Temporary documents may also disappear after a restart or redeployment, so keep your original PDFs.

The BGE-M3 model is shared for read-only inference, and indexing and search are serialized to limit concurrent memory use. The app allows at most three documents per session, with a 25 MB, 100-page, and 2,500-chunk limit per document. Session separation is not account authentication or a guarantee against abuse of a public service.

When hosted, PDFs and queries are sent to the server running the app and processed there. The hosting provider operates that server; do not upload confidential or unpublished material to a public demo. Document text and queries are not sent to an external inference API. Network access is needed to download the embedding model when it is not cached; Streamlit anonymous usage statistics are disabled in `.streamlit/config.toml`.

Command-line indexing keeps persistent files in the requested output or corpus directory; those files are not subject to the Streamlit session cleanup.

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

The test suite includes a generated multi-page PDF, real BGE-M3 Chinese-to-English retrieval with page verification, highlighting, empty-document handling, oversized `top_k`, legacy-index detection, malformed corpus IDs, independent Streamlit sessions, upload limits and rollback, and automatic/manual cleanup. Public page previews are capped at 1,600 pixels on their longest side.

## Limitations

Doc Finder is an early-stage research utility. Please verify retrieved text and page information against the original document before using them in academic citations.

- **Text-based PDFs only.** Scanned or image-only PDFs are not currently supported. There is no OCR pipeline.
- **PDF page numbers, not printed page labels.** Page 1 means the first physical page of the PDF file. In books with front matter, this may differ from the page number printed on the page.
- **Chunk-level rather than sentence-level retrieval.** Text is currently divided into approximately 300-character chunks. A result may contain the desired sentence together with surrounding text, and a sentence can occasionally cross chunk boundaries.
- **PDF extraction artifacts remain.** Running headers, page numbers, line-break artifacts, and hyphenation may appear in extracted text.
- **Loose paraphrases are harder.** In small informal tests, relatively faithful translations were generally retrieved successfully, while short or substantially rewritten notes could rank the relevant passage much lower.
- **No calibrated “no match” decision yet.** The system always returns the nearest passages. A high-ranking result does not guarantee that the intended source is actually present in the document.
- **No formal retrieval benchmark yet.** Current behavior has been checked with smoke tests and a small number of manually constructed examples rather than a dedicated evaluation dataset.
- **Temporary sessions, no accounts.** Streamlit documents are isolated by browser session and automatically expire. There is no authentication, permanent online library, or service-wide abuse prevention; repeated sessions can still consume server resources.
- **Resource requirements.** BGE-M3 requires roughly 2.5 GB of RAM in the current setup, with additional memory needed for inference and documents. CPU indexing and search are serialized, so simultaneous users may wait. Free Community Cloud operation still requires live verification.
- **English source documents are the current target.** BGE-M3 is multilingual, so other source languages may work, but they have not yet been evaluated.

## Roadmap

Potential next steps include:

- sentence-aware chunking;
- de-hyphenation and cleaner PDF text normalization;
- detection of printed page numbers or page labels;
- a small public Chinese-to-English source-localization evaluation set;
- Recall@1 / Recall@5 reporting;
- calibrated no-match detection;
- deployment memory measurements and service-wide resource controls.

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
| [Streamlit](https://streamlit.io/) | Web interface | Apache-2.0 | |
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

Doc Finder 是一个面向研究者的文献辅助工具，可在本地运行，也可通过 Streamlit 网页使用。

用户上传英文 PDF，再输入一段根据该文献写下的中文翻译、转述或阅读笔记，程序会利用 BGE-M3 的多语言语义表示，在英文原文中检索最相关的文本片段，并返回对应的 PDF 页码。用户还可以查看高亮后的 PDF 页面，以便在论文写作或引用时快速核对原文和页码。

Doc Finder 不先把中文翻译成英文，而是直接将中文查询和英文文本映射到同一个多语言向量空间中进行语义匹配。

当前 v0.1 主要支持文本型英文 PDF，不提供 OCR。网页中的文档按浏览器会话隔离，每个会话最多保存 3 份文档，每份限 25 MB、100 页、2,500 个文本片段。可手动清空；会话约 60 分钟无操作后，运行中的后台清理程序会删除临时文件。关闭页面不代表立即删除。托管使用时，PDF 和查询会上传到应用服务器处理，不会发送到外部推理 API。

仓库已准备 Streamlit Community Cloud 部署配置，但尚未验证云端运行；BGE-M3 的内存需求可能超过免费实例额度。印刷页码识别、句子级定位和严格的“未找到”判断仍属于后续工作。
