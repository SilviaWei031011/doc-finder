"""英文文献反查系统 — Streamlit UI.

Run:   streamlit run app.py
Model: set DOC_FINDER_MODEL to a sentence-transformers model id or a local directory
       (default BAAI/bge-m3, downloaded from the Hugging Face Hub on first use, ~2.3 GB).
Data:  uploaded PDFs and their indexes live in ./corpora (override with DOC_FINDER_CORPUS_DIR).
"""
import os
import shutil
import time

import streamlit as st

import core  # imports torch before faiss — see the note in core.py

st.set_page_config(page_title="英文文献反查系统", page_icon="📚", layout="wide")
st.title("📚 英文文献反查系统（中文 → 英文原文 + 页码 + 高亮）")

CORPUS_DIR = core.corpus_dir()
os.makedirs(CORPUS_DIR, exist_ok=True)


@st.cache_resource(show_spinner="正在加载 BGE-M3 编码模型（首次使用可能需要下载约 2.3 GB）…")
def get_model(name: str):
    return core.load_model(name)


@st.cache_resource(show_spinner=False)
def get_corpus(folder: str, _mtime: float):
    # _mtime is part of the cache key, so a rebuilt index is reloaded automatically
    return core.load_corpus(folder)


@st.cache_data(show_spinner=False, max_entries=64)
def render_page(pdf_path: str, page: int, text: str) -> bytes:
    return core.render_highlight(pdf_path, page, text)


def load_model_or_stop():
    try:
        return get_model(core.model_name())
    except Exception as e:  # noqa: BLE001
        st.error(
            f"无法加载编码模型「{core.model_name()}」：{e}\n\n"
            "请检查网络（首次需从 Hugging Face 下载），或用环境变量 DOC_FINDER_MODEL 指向本地模型目录。"
        )
        st.stop()


def run_build(pdf_path: str, folder: str, model, where):
    bar = where.progress(0.0, text="正在解析 PDF …")

    def cb(done, total):
        bar.progress(done / total, text=f"正在计算向量 {done}/{total} 片段")

    result = core.build_corpus(pdf_path, folder, model, progress_cb=cb)
    bar.progress(1.0, text="索引构建完成")
    return result


ss = st.session_state
ss.setdefault("selected_corpus_id", None)
ss.setdefault("results", None)
ss.setdefault("search_seq", 0)
ss.setdefault("flash", None)

if ss.flash:
    st.success(ss.flash)
    ss.flash = None

# ---------------------------------------------------------------- sidebar: library
try:
    collections = core.load_collections(CORPUS_DIR)
except core.RegistryError as e:
    st.error(f"文献登记表读取失败（未做任何修改）：{e}")
    st.stop()
collections.sort(key=lambda c: c.get("created_at", 0), reverse=True)

st.sidebar.title("📁 文献库管理")
st.sidebar.markdown("### 📚 当前文献库")
if not collections:
    st.sidebar.info("当前还没有任何文献，请上传。")

for c in collections:
    with st.sidebar.expander(f"📖 {c['name']}", expanded=False):
        st.write(f"ID: `{c['id']}`")
        st.write(f"文件: `{c.get('original_filename') or c['pdf_filename']}`")
        col1, col2 = st.columns([2, 1])
        if col1.button("➡️ 选择此文献", key=f"choose_{c['id']}"):
            ss.selected_corpus_id = c["id"]
            ss.results = None
            st.rerun()
        if col2.button("🗑 删除", key=f"delete_{c['id']}"):
            try:
                core.delete_corpus(CORPUS_DIR, c["id"])
            except Exception as e:  # noqa: BLE001
                st.error(f"删除失败：{e}")
            else:
                if ss.selected_corpus_id == c["id"]:
                    ss.selected_corpus_id = None
                    ss.results = None
                ss.flash = f"已删除文献：{c['name']}"
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 上传新文献（自动构建索引）")
custom_name = st.sidebar.text_input("给这本文献起一个名称（可选，默认用文件名）", value="")
uploaded = st.sidebar.file_uploader("选择 PDF 文件（论文 / 书籍）", type=["pdf"])

if uploaded is not None and st.sidebar.button("开始上传并建立索引", key="upload"):
    model = load_model_or_stop()
    corpus_id = core.new_corpus_id(CORPUS_DIR)
    folder = core.corpus_folder(CORPUS_DIR, corpus_id)
    original_name = core.sanitize_display_name(uploaded.name)
    display_name = core.sanitize_display_name(custom_name) if custom_name.strip() else original_name
    try:
        os.makedirs(folder)
        # The PDF is stored under a fixed name; the original filename is kept as metadata only.
        pdf_path = os.path.join(folder, core.PDF_FILENAME)
        with open(pdf_path, "wb") as f:
            f.write(uploaded.getvalue())
        result = run_build(pdf_path, folder, model, st.sidebar)
        core.register_corpus(CORPUS_DIR, {
            "id": corpus_id,
            "name": display_name,
            "pdf_filename": core.PDF_FILENAME,
            "original_filename": original_name,
            "folder": corpus_id,
            "created_at": time.time(),
            "num_pages": result.num_pages,
            "num_chunks": result.num_chunks,
        })
    except core.NoTextError as e:
        shutil.rmtree(folder, ignore_errors=True)
        st.sidebar.error(f"未建立索引：{e}")
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(folder, ignore_errors=True)
        st.sidebar.error(f"上传 / 索引失败：{e}")
    else:
        ss.selected_corpus_id = corpus_id
        ss.results = None
        ss.flash = (f"上传并索引完成：{display_name}"
                    f"（{result.num_pages} 页，{result.num_chunks} 个片段，{result.seconds:.0f} 秒）")
        st.rerun()

# ---------------------------------------------------------------- main: search
selected = next((c for c in collections if c["id"] == ss.selected_corpus_id), None)
if selected is None:
    ss.selected_corpus_id = None
    st.info("请在左侧选择或上传一本文献后使用检索功能。")
    st.stop()

st.success(f"当前文献：**{selected['name']}**")
folder = core.corpus_folder(CORPUS_DIR, selected["folder"])
pdf_path = os.path.join(folder, os.path.basename(selected["pdf_filename"]))
if not core.corpus_files_exist(folder):
    st.error("该文献的索引文件缺失，请删除后重新上传。")
    st.stop()

model = load_model_or_stop()
corpus = get_corpus(folder, core.corpus_mtime(folder))

problem = core.compatibility_error(corpus, model)
if problem:
    st.warning(f"⚠️ {problem}。需要用当前模型重建索引后才能检索（PDF 不会被删除）。")
    if not os.path.isfile(pdf_path):
        st.error("找不到该文献的 PDF 文件，无法重建，请删除后重新上传。")
    elif st.button("🔄 重建索引", key="rebuild"):
        try:
            result = run_build(pdf_path, folder, model, st)
        except Exception as e:  # noqa: BLE001
            st.error(f"重建失败（原索引未改动）：{e}")
        else:
            get_corpus.clear()
            ss.results = None
            ss.flash = f"索引已重建：{result.num_pages} 页，{result.num_chunks} 个片段，{result.seconds:.0f} 秒"
            st.rerun()
    st.stop()

note = core.model_mismatch_note(corpus, model)
if note:
    st.caption(f"ℹ️ {note}")
st.caption(f"共 {corpus.meta.get('num_pages', '?')} 页，{corpus.index.ntotal} 个文本片段。"
           "页码为 PDF 物理页码（从 1 起），可能与书上印刷的页码不同。")

query = st.text_area("请输入中文译文 / 转述 / 笔记（也可直接输入英文片段）：", height=150, key="query")
max_k = min(20, corpus.index.ntotal)
if max_k > 1:
    top_k = st.slider("显示候选数量 Top-K", 1, max_k, min(5, max_k), 1, key="top_k")
else:
    top_k = 1

if st.button("开始检索", key="search"):
    if not query.strip():
        st.error("请输入内容。")
    else:
        hits = core.search(corpus, model, query, top_k)
        ss.search_seq += 1
        ss.results = {"corpus_id": selected["id"], "query": query.strip(), "hits": hits, "seq": ss.search_seq}

# Results live in session_state, so they survive the reruns triggered by the buttons below.
res = ss.results
if res and res["corpus_id"] == selected["id"]:
    st.markdown("### 🔍 检索结果（按相似度排序）")
    if not res["hits"]:
        st.info("没有返回结果。")
    for h in res["hits"]:
        st.markdown(f"**候选 {h.rank}（相似度 {h.score:.4f}）** — 📄 第 {h.page} 页")
        st.write(h.text)
        if st.checkbox(f"显示候选 {h.rank} 的高亮页面", key=f"hl_{res['seq']}_{h.rank}"):
            try:
                st.image(render_page(pdf_path, h.page, h.text), caption=f"第 {h.page} 页（高亮）")
            except Exception as e:  # noqa: BLE001
                st.error(f"生成高亮页面时出错：{e}")
        st.divider()
