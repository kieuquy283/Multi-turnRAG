from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import font as tkfont, messagebox, scrolledtext, simpledialog, ttk

from code.chat_api import run_chat
from code.config import BASE_DIR
from code.vectorstore import load_vectorstore

SESSION_FILE = BASE_DIR / "data" / "chat_sessions.json"


def ensure_data_dir() -> None:
    data_dir = SESSION_FILE.parent
    data_dir.mkdir(parents=True, exist_ok=True)


def load_sessions() -> list[dict[str, Any]]:
    ensure_data_dir()
    if not SESSION_FILE.exists():
        return []

    try:
        with SESSION_FILE.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError:
        return []


def save_sessions(sessions: list[dict[str, Any]]) -> None:
    ensure_data_dir()
    with SESSION_FILE.open("w", encoding="utf-8") as stream:
        json.dump(sessions, stream, ensure_ascii=False, indent=2)


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def build_session(name: str | None = None) -> dict[str, Any]:
    title = name.strip() if name and name.strip() else f"Phiên chat mới {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    return {
        "id": str(uuid.uuid4()),
        "name": title,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "history": [],
    }


def append_message(session: dict[str, Any], role: str, content: str) -> None:
    session["history"].append({"role": role, "content": content})
    session["updated_at"] = now_iso()


def find_best_font(root: tk.Tk) -> str:
    available = set(tkfont.families(root))
    for family in ("Nimbus Sans L", "DejaVu Sans", "Liberation Sans", "Ubuntu", "Arial", "Sans"):
        if family in available:
            return family
    return "TkDefaultFont"


def find_session(session_id: str, sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for session in sessions:
        if session.get("id") == session_id:
            return session
    return None


class ChatDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Multi-turn RAG Chat Desktop")
        self.geometry("1000x640")
        self.configure(bg="#f5f7fb")

        self.font_family = find_best_font(self)
        self.sessions: list[dict[str, Any]] = []
        self.active_session: dict[str, Any] | None = None
        self.vectorstore = load_vectorstore()

        self.create_widgets()
        self.load_sessions()

    def create_widgets(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Sidebar.TFrame", background="#eef2ff")
        style.configure("Content.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#eef2ff", font=(self.font_family, 13, "bold"))
        style.configure("TLabel", background="#ffffff", font=(self.font_family, 10))
        style.configure("TButton", font=(self.font_family, 10, "bold"), padding=8)
        style.configure("Status.TLabel", foreground="#6b7280", background="#ffffff")

        left_frame = ttk.Frame(self, style="Sidebar.TFrame", padding=(14, 14, 12, 14))
        left_frame.grid(row=0, column=0, sticky="nsew")
        right_frame = ttk.Frame(self, style="Content.TFrame", padding=(16, 14, 16, 14))
        right_frame.grid(row=0, column=1, sticky="nsew")

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        title_label = ttk.Label(left_frame, text="Phiên chat", style="Header.TLabel")
        title_label.grid(row=0, column=0, sticky="w")

        button_frame = ttk.Frame(left_frame, style="Sidebar.TFrame")
        button_frame.grid(row=1, column=0, sticky="ew", pady=(12, 14))
        button_frame.columnconfigure((0, 1), weight=1)

        new_button = ttk.Button(button_frame, text="Tạo mới", command=self.create_session)
        new_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        delete_button = ttk.Button(button_frame, text="Xóa", command=self.delete_session)
        delete_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.session_scroll = ttk.Scrollbar(left_frame, orient="vertical")
        self.session_listbox = tk.Listbox(
            left_frame,
            width=32,
            height=24,
            activestyle="none",
            selectbackground="#2563eb",
            selectforeground="#ffffff",
            bg="#f8fafc",
            bd=0,
            highlightthickness=0,
            yscrollcommand=self.session_scroll.set,
            font=(self.font_family, 10),
        )
        self.session_scroll.config(command=self.session_listbox.yview)
        self.session_listbox.grid(row=2, column=0, sticky="nsew")
        self.session_scroll.grid(row=2, column=1, sticky="ns")
        self.session_listbox.bind("<<ListboxSelect>>", self.on_session_select)

        left_frame.grid_rowconfigure(2, weight=1)

        self.status_label = ttk.Label(right_frame, text="Chọn một phiên để bắt đầu chat.", style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.messages_text = scrolledtext.ScrolledText(
            right_frame,
            wrap="word",
            state="disabled",
            bg="#f8fafc",
            fg="#111827",
            relief="flat",
            borderwidth=0,
            height=22,
            font=(self.font_family, 10),
        )
        self.messages_text.grid(row=1, column=0, sticky="nsew")
        self.messages_text.tag_configure("user", background="#2563eb", foreground="#ffffff", lmargin1=10, lmargin2=10, rmargin=10, spacing3=8)
        self.messages_text.tag_configure("assistant", background="#e2e8f0", foreground="#111827", lmargin1=10, lmargin2=10, rmargin=10, spacing3=8)
        self.messages_text.tag_configure("meta", foreground="#6b7280", font=(self.font_family, 8, "italic"))

        input_frame = ttk.Frame(right_frame, style="Content.TFrame")
        input_frame.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        input_frame.grid_columnconfigure(0, weight=1)

        self.question_entry = tk.Text(
            input_frame,
            height=5,
            wrap="word",
            bg="#ffffff",
            bd=1,
            relief="solid",
            font=(self.font_family, 10),
        )
        self.question_entry.grid(row=0, column=0, sticky="nsew")
        self.question_entry.bind('<Control-Return>', self.on_send_question)

        send_button = ttk.Button(input_frame, text="Gửi (Ctrl+Enter)", command=self.on_send_question)
        send_button.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        right_frame.grid_rowconfigure(1, weight=1)

    def load_sessions(self) -> None:
        self.sessions = load_sessions()
        self.session_listbox.delete(0, tk.END)
        for session in self.sessions:
            self.session_listbox.insert(tk.END, session.get("name", "Phiên không tên"))

        if self.sessions:
            self.session_listbox.selection_set(0)
            self.on_session_select()

    def save_sessions(self) -> None:
        save_sessions(self.sessions)

    def create_session(self) -> None:
        name = simpledialog.askstring("Tạo phiên mới", "Tên phiên:", parent=self)
        session = build_session(name)
        self.sessions.insert(0, session)
        self.save_sessions()
        self.load_sessions()
        self.status_label.config(text=f"Phiên: {session['name']}")

    def delete_session(self) -> None:
        if not self.active_session:
            messagebox.showwarning("Không có phiên", "Vui lòng chọn một phiên để xóa.")
            return
        confirm = messagebox.askyesno("Xóa phiên", f"Bạn có chắc muốn xóa phiên '{self.active_session['name']}'?")
        if not confirm:
            return
        self.sessions = [s for s in self.sessions if s["id"] != self.active_session["id"]]
        self.active_session = None
        self.save_sessions()
        self.load_sessions()
        self.render_messages()

    def on_session_select(self, event: Any | None = None) -> None:
        selection = self.session_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self.active_session = self.sessions[index]
        self.render_messages()

    def render_messages(self) -> None:
        self.messages_text.configure(state="normal")
        self.messages_text.delete("1.0", tk.END)

        if not self.active_session:
            self.status_label.config(text="Chọn một phiên để bắt đầu chat.")
            self.messages_text.insert(tk.END, "Chọn hoặc tạo phiên mới để bắt đầu.")
            self.messages_text.configure(state="disabled")
            return

        self.status_label.config(text=f"Phiên: {self.active_session['name']}")

        if not self.active_session["history"]:
            self.messages_text.insert(tk.END, "Phiên chưa có tin nhắn. Gõ câu hỏi để bắt đầu.", "meta")
        else:
            for message in self.active_session["history"]:
                role = message["role"]
                content = message["content"].strip()
                tag = "user" if role == "user" else "assistant"
                header = "Bạn" if role == "user" else "Bot"
                self.messages_text.insert(tk.END, f"{header}:\n", (tag,))
                self.messages_text.insert(tk.END, f"{content}\n\n", (tag,))

        self.messages_text.see(tk.END)
        self.messages_text.configure(state="disabled")

    def on_send_question(self, event: Any | None = None) -> None:
        if event is not None:
            event.widget.master.focus_set()
        question = self.question_entry.get("1.0", tk.END).strip()
        if not question:
            return
        if not self.active_session:
            messagebox.showwarning("Không có phiên", "Vui lòng chọn hoặc tạo phiên chat.")
            return

        try:
            reply = run_chat(
                question=question,
                history=self.active_session["history"],
                vectorstore=self.vectorstore,
            )
        except Exception as exc:
            messagebox.showerror("Lỗi", str(exc))
            return

        append_message(self.active_session, "user", question)
        append_message(self.active_session, "assistant", reply.get("answer", ""))
        self.save_sessions()
        self.render_messages()
        self.question_entry.delete("1.0", tk.END)


if __name__ == "__main__":
    app = ChatDesktopApp()
    app.mainloop()
