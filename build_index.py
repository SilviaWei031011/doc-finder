#!/usr/bin/env python
"""Build or rebuild doc_finder indexes from the command line (no UI).

Examples
  python build_index.py paper.pdf --out /tmp/paper_index   # index one PDF into a directory
  python build_index.py --rebuild c95d4e10                   # rebuild one corpus in ./corpora
  python build_index.py --rebuild-all                        # rebuild every corpus in ./corpora

Uses exactly the same pipeline as the web app (core.build_corpus), so the indexes are interchangeable.
"""
import argparse
import os
import sys

import core


def _progress(done: int, total: int) -> None:
    print(f"\r  向量化 {done}/{total}", end="", flush=True)
    if done >= total:
        print()


def build_one(pdf_path: str, out_dir: str, model) -> None:
    print(f"PDF: {pdf_path}\n输出目录: {out_dir}")
    result = core.build_corpus(pdf_path, out_dir, model, progress_cb=_progress)
    print(f"完成：{result.num_pages} 页，{result.num_chunks} 个片段，{result.seconds:.0f} 秒")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="为英文 PDF 构建语义检索索引（与 app.py 完全一致的流程）")
    parser.add_argument("pdf", nargs="?", help="PDF 文件路径")
    parser.add_argument("--out", help="索引输出目录（默认：PDF 所在目录）")
    parser.add_argument("--rebuild", metavar="CORPUS_ID", help="重建文献库中某一文献的索引")
    parser.add_argument("--rebuild-all", action="store_true", help="重建文献库中所有文献的索引")
    parser.add_argument("--corpus-dir", default=core.corpus_dir(), help="文献库目录（默认 ./corpora）")
    args = parser.parse_args(argv)
    if not (args.pdf or args.rebuild or args.rebuild_all):
        parser.error("请提供 PDF 路径，或使用 --rebuild / --rebuild-all")

    print(f"模型: {core.model_name()}  (pooling={core.POOLING}, normalize=True)")
    model = core.load_model()

    if args.pdf:
        if not os.path.isfile(args.pdf):
            sys.exit(f"PDF 文件不存在：{args.pdf}")
        build_one(args.pdf, args.out or os.path.dirname(os.path.abspath(args.pdf)), model)

    if args.rebuild or args.rebuild_all:
        targets = [c for c in core.load_collections(args.corpus_dir)
                   if args.rebuild_all or c["id"] == args.rebuild]
        if not targets:
            sys.exit(f"文献库中没有匹配的文献：{args.rebuild or '(空)'}")
        for c in targets:
            folder = core.corpus_folder(args.corpus_dir, c["folder"])
            pdf_path = os.path.join(folder, os.path.basename(c["pdf_filename"]))
            if not os.path.isfile(pdf_path):
                print(f"跳过 {c['id']}（{c['name']}）：找不到 PDF {pdf_path}")
                continue
            print(f"\n[{c['id']}] {c['name']}")
            build_one(pdf_path, folder, model)


if __name__ == "__main__":
    main()
