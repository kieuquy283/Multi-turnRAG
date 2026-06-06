from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipelines.factory import build_chat_pipeline


DEMO_QUESTIONS = [
    "Thời hiệu yêu cầu Tòa án giải quyết tranh chấp đất đai là bao lâu?",
    "Điều kiện để được cấp Giấy chứng nhận quyền sử dụng đất là gì?",
    "Tranh chấp đất đai phải qua hòa giải ở cơ sở trước khi khởi kiện không?",
    "Quy định về bồi thường khi Nhà nước thu hồi đất như thế nào?",
    "Người sử dụng đất có những quyền và nghĩa vụ gì?",
]


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            if hasattr(o, "to_dict"):
                return o.to_dict()
            return dataclasses.asdict(o)
        return super().default(o)


def _dump_json(obj) -> str:
    return json.dumps(obj, cls=_Encoder, ensure_ascii=False, indent=2)


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _color(text, "1")


def _cyan(text: str) -> str:
    return _color(text, "36")


def _yellow(text: str) -> str:
    return _color(text, "33")


def _green(text: str) -> str:
    return _color(text, "32")


def _red(text: str) -> str:
    return _color(text, "31")


def _dim(text: str) -> str:
    return _color(text, "2")


def _print_separator() -> None:
    print(_dim("─" * 70))


def _print_banner() -> None:
    banner = r"""
 ╔══════════════════════════════════════════════════════════════╗
 ║          MULTI-TURN RAG  —  Hỏi Đáp Demo                   ║
 ║          Hệ thống hỏi đáp pháp luật Việt Nam               ║
 ╚══════════════════════════════════════════════════════════════╝
"""
    print(_cyan(banner))


def _print_help() -> None:
    cmds = [
        ("số (1-5)", "Chọn câu hỏi demo có sẵn"),
        ("demo", "Chạy tất cả câu hỏi demo"),
        ("reset", "Xóa lịch sử hội thoại"),
        ("history", "Xem lịch sử hội thoại"),
        ("exit / quit", "Thoát"),
    ]
    print(_bold("Commands:"))
    for cmd, desc in cmds:
        print(f"  {_yellow(cmd):30s} {desc}")
    print()


def _print_top_files(top_files: list[dict]) -> None:
    if not top_files:
        return
    print(_bold("Top Files:"))
    for i, f in enumerate(top_files, 1):
        source = f.get("source_file", "?")
        score = f.get("best_score", 0.0)
        hits = f.get("hits", 0)
        print(f"  {_cyan(str(i))}. {source}  {_dim(f'(score={score:.4f}, hits={hits})')}")
    print()


def _print_metadata(result: dict) -> None:
    used_rewrite = result.get("used_rewrite", False)
    show_rewrite = result.get("show_rewritten_query", False)
    mode = result.get("mode", "")
    grounded = result.get("grounded", False)
    warning = result.get("warning", "")
    metadata = result.get("metadata", {})
    latency = metadata.get("latency_seconds", 0)

    parts = []
    if mode:
        parts.append(f"Mode: {_yellow(mode)}")
    if show_rewrite and used_rewrite:
        rq = result.get("rewritten_query", "")
        parts.append(f"Rewrite: {_cyan(rq)}")
    parts.append(f"Grounded: {_green('Yes') if grounded else _red('No')}")
    parts.append(f"Latency: {latency:.2f}s")

    print(_bold("Info:"))
    print(f"  {' | '.join(parts)}")

    if warning:
        print(f"  {_yellow('Warning:')} {warning}")
    print()


def _run_single_question(pipeline, question: str, history: list, *, verbose: bool = True) -> dict:
    print(_bold(f"Q: {question}"))
    _print_separator()

    t0 = time.perf_counter()
    result = pipeline.chat(question=question, history=history)
    elapsed = time.perf_counter() - t0

    if verbose:
        _print_metadata(result)
        _print_top_files(result.get("top_files", []))

    answer = result.get("answer", "")
    print(_bold("A:"))
    print(f"  {answer}\n")
    print(_dim(f"  [{elapsed:.2f}s]"))
    _print_separator()
    print()

    return result


def _run_demo(pipeline, *, verbose: bool = True) -> None:
    history: list[dict] = []
    total_time = 0.0

    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(_bold(f"\n{'═' * 70}"))
        print(_bold(f"  DEMO {i}/{len(DEMO_QUESTIONS)}"))
        print(_bold(f"{'═' * 70}\n"))

        t0 = time.perf_counter()
        result = _run_single_question(pipeline, q, history, verbose=verbose)
        total_time += time.perf_counter() - t0

        history = result.get("history", history)

    print(_bold(f"\nDemo hoàn tất — Tổng thời gian: {total_time:.2f}s\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn RAG Q&A Demo")
    parser.add_argument("--index-dir", default=None, help="Path to index directory")
    parser.add_argument("--mode", default=None, help="Pipeline mode: adaptive|modular|legacy")
    parser.add_argument("--demo", action="store_true", help="Chạy tự động tất cả câu hỏi demo")
    parser.add_argument("--quiet", action="store_true", help="Chỉ hiện câu trả lời, không hiện metadata")
    args = parser.parse_args()

    kwargs = {}
    if args.index_dir:
        kwargs["index_dir"] = args.index_dir
    if args.mode:
        kwargs["pipeline_mode"] = args.mode

    print(_dim("Đang khởi tạo pipeline..."))
    pipeline = build_chat_pipeline(**kwargs)
    verbose = not args.quiet

    _print_banner()

    if args.demo:
        _run_demo(pipeline, verbose=verbose)
        return

    _print_help()

    history: list[dict] = []

    while True:
        try:
            raw = input(_bold("Bạn: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not raw:
            continue

        if raw.lower() in {"exit", "quit", "q"}:
            print("Thoát.")
            break

        if raw.lower() == "reset":
            history = []
            print(_green("Đã xóa lịch sử hội thoại.\n"))
            continue

        if raw.lower() == "history":
            if not history:
                print(_dim("Chưa có lịch sử.\n"))
            else:
                for msg in history:
                    role = _cyan("Q") if msg["role"] == "user" else _green("A")
                    content = msg["content"][:120] + ("..." if len(msg["content"]) > 120 else "")
                    print(f"  {role}: {content}")
                print()
            continue

        if raw.lower() == "demo":
            _run_demo(pipeline, verbose=verbose)
            continue

        if raw.isdigit() and 1 <= int(raw) <= len(DEMO_QUESTIONS):
            question = DEMO_QUESTIONS[int(raw) - 1]
        else:
            question = raw

        try:
            result = _run_single_question(pipeline, question, history, verbose=verbose)
            history = result.get("history", history)
        except Exception as exc:
            print(_red(f"\n[ERROR] {exc}\n"))


if __name__ == "__main__":
    main()
