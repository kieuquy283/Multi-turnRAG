from __future__ import annotations

import json
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from code.chat_api import run_chat
from code.config import BASE_DIR
from code.vectorstore import load_vectorstore

SESSION_FILE = BASE_DIR / "data" / "chat_sessions.json"
HTML_FILE = Path(__file__).resolve().parent / "chat_ui.html"
HOST = "0.0.0.0"
PORT = 8500


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


def find_session(session_id: str, sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for session in sessions:
        if session.get("id") == session_id:
            return session
    return None


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def build_session(name: str | None = None) -> dict[str, Any]:
    title = name.strip() if name and name.strip() else f"New chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
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


class ChatServerHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            try:
                html = HTML_FILE.read_text(encoding="utf-8")
                self._send_text(html)
            except FileNotFoundError:
                self._send_json({"error": "Giao diện không tìm thấy."}, status=HTTPStatus.NOT_FOUND)
            return

        if self.path == "/api/sessions":
            sessions = load_sessions()
            self._send_json({"sessions": sessions})
            return

        if self.path.startswith("/api/sessions/"):
            session_id = self.path.split("/", 3)[-1]
            sessions = load_sessions()
            session = find_session(session_id, sessions)
            if not session:
                self._send_json({"error": "Session không tồn tại."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"session": session})
            return

        self._send_json({"error": "Không tìm thấy endpoint."}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        if self.path.startswith("/api/sessions/"):
            session_id = self.path.split("/", 3)[-1]
            sessions = load_sessions()
            session = find_session(session_id, sessions)
            if not session:
                self._send_json({"error": "Session không tồn tại."}, status=HTTPStatus.NOT_FOUND)
                return

            sessions = [s for s in sessions if s.get("id") != session_id]
            save_sessions(sessions)
            self._send_json({"deleted": session_id})
            return

        self._send_json({"error": "Không tìm thấy endpoint."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/sessions":
            body = self._read_json()
            session = build_session(body.get("name"))
            sessions = load_sessions()
            sessions.insert(0, session)
            save_sessions(sessions)
            self._send_json({"session": session}, status=HTTPStatus.CREATED)
            return

        if self.path == "/api/chat":
            body = self._read_json()
            session_id = body.get("session_id")
            question = str(body.get("question", "")).strip()

            if not session_id or not question:
                self._send_json(
                    {"error": "session_id và question là bắt buộc."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            sessions = load_sessions()
            session = find_session(session_id, sessions)
            if not session:
                self._send_json({"error": "Session không tồn tại."}, status=HTTPStatus.NOT_FOUND)
                return

            try:
                reply = run_chat(question=question, history=session["history"], vectorstore=VECTORSTORE)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            append_message(session, "user", question)
            append_message(session, "assistant", reply.get("answer", ""))
            save_sessions(sessions)

            self._send_json({"session": session, "reply": reply})
            return

        self._send_json({"error": "Không tìm thấy endpoint."}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server() -> None:
    global VECTORSTORE
    print("Đang load FAISS index...")
    VECTORSTORE = load_vectorstore()
    print(f"Server chạy tại http://{HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), ChatServerHandler)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
