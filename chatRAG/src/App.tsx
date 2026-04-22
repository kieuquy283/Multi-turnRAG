import { FormEvent, KeyboardEvent, MouseEvent, useEffect, useState } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChatIcon, ChartAnalysisIcon, Settings01Icon, UserIcon, SquareArrowLeft02Icon, Delete02Icon } from "@hugeicons/core-free-icons";

type Message = {
  role: "user" | "assistant";
  content: string;
  time?: string;
};

type ChatResponse = {
  answer: string;
  rewritten_query: string;
  used_rewrite: boolean;
  show_rewritten_query: boolean;
  grounded: boolean;
  warning: string;
  mode: string;
  top_files: Array<{ source_file: string; best_score: number; hits: number }>;
  history: Message[];
};

type EvaluationStats = {
  name: string;
  top_k: number;
  eval_path: string;
  sample_count: number;
  hit: number;
  recall: number;
  mrr: number;
};

type EvaluationResponse = {
  results: EvaluationStats[];
};

type ChatSession = {
  id: string;
  name: string;
  createdAt: string;
  messages: Message[];
};

type TabKey = "chat" | "evaluation" | "settings";

const DEFAULT_MESSAGES: Message[] = [
  {
    role: "assistant",
    content:
      "Xin chào. Tôi có thể giúp gì cho bạn?",
  },
];

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("chat");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>(DEFAULT_MESSAGES);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [chatDropdownOpen, setChatDropdownOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState<EvaluationStats[]>([]);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const SESSIONS_API_URL = "http://127.0.0.1:8000/sessions";

  const makeSessionTitle = (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) return "Cuộc chat mới";
    return trimmed.length > 32 ? `${trimmed.slice(0, 32)}...` : trimmed;
  };

  const isDefaultSessionTitle = (title: string) => title.startsWith("Session ") || title === "Cuộc chat mới";

  const persistSessions = (updatedSessions: ChatSession[]) => {
    setSessions(updatedSessions);
    fetch(SESSIONS_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updatedSessions),
    }).catch(() => {
      // Nếu việc lưu file bị lỗi thì vẫn giữ local state.
    });
  };

  const loadSession = (sessionId: string) => {
    const session = sessions.find((item) => item.id === sessionId);
    if (!session) return;
    setActiveSessionId(sessionId);
    setMessages(session.messages);
    setActiveTab("chat");
  };

  const createNewSession = () => {
    const name = `Session ${sessions.length + 1}`;
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      name,
      createdAt: new Date().toLocaleString(),
      messages: DEFAULT_MESSAGES,
    };
    persistSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
    setMessages(newSession.messages);
    setChatDropdownOpen((true));
  };

  const deleteSession = (sessionId: string) => {
    const remaining = sessions.filter((session) => session.id !== sessionId);
    if (remaining.length === 0) {
      const newSession: ChatSession = {
        id: `session-${Date.now()}`,
        name: "Session 1",
        createdAt: new Date().toLocaleString(),
        messages: DEFAULT_MESSAGES,
      };
      persistSessions([newSession]);
      setActiveSessionId(newSession.id);
      setMessages(newSession.messages);
      return;
    }

    persistSessions(remaining);
    if (activeSessionId === sessionId) {
      setActiveSessionId(remaining[0].id);
      setMessages(remaining[0].messages);
    }
  };

  const updateCurrentSessionMessages = (updatedMessages: Message[]) => {
    setMessages(updatedMessages);
    const updatedSessions = sessions.map((session) => {
      if (session.id !== activeSessionId) return session;
      const updatedSession = { ...session, messages: updatedMessages };

      const hasOneGreetingMessage = session.messages.length === 1 && session.messages[0].role === "assistant";
      const firstUserMessage = updatedMessages.find((message) => message.role === "user");
      if (
        hasOneGreetingMessage &&
        firstUserMessage &&
        isDefaultSessionTitle(session.name)
      ) {
        updatedSession.name = makeSessionTitle(firstUserMessage.content);
      }
      return updatedSession;
    });
    persistSessions(updatedSessions);
  };

  useEffect(() => {
    const loadSessionsFromServer = async () => {
      try {
        const response = await fetch(SESSIONS_API_URL);
        if (response.ok) {
          const parsed = (await response.json()) as ChatSession[];
          if (parsed.length > 0) {
            setSessions(parsed);
            setActiveSessionId(parsed[0].id);
            setMessages(parsed[0].messages);
            return;
          }
        }
      } catch {
        // ignore fetch errors
      }
      createNewSession();
    };

    loadSessionsFromServer();
  }, []);

  const sendMessage = async (event?: FormEvent<HTMLFormElement> | MouseEvent<HTMLButtonElement>) => {
    event?.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const userMessage: Message = { role: "user", content: trimmed, time: timestamp };
    const updatedMessages = [...messages, userMessage];
    updateCurrentSessionMessages(updatedMessages);
    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmed, history: updatedMessages }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "API error");
      }

      const data: ChatResponse = await response.json();
      const answerTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const assistantMessage: Message = { role: "assistant", content: data.answer, time: answerTime };
      const updatedMessagesWithAnswer = [...updatedMessages, assistantMessage];
      updateCurrentSessionMessages(updatedMessagesWithAnswer);
    } catch (err) {
      const errorMessage = (err as Error).message;
      const errorMessageText = `Lỗi: ${errorMessage}`;
      const errorMessageEntry: Message = {
        role: "assistant",
        content: errorMessageText,
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      const updatedMessagesWithError = [...updatedMessages, errorMessageEntry];
      updateCurrentSessionMessages(updatedMessagesWithError);
    } finally {
      setLoading(false);
    }
  };

  const fetchEvaluation = async () => {
    setEvaluationLoading(true);
    setEvaluationError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/evaluation");
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "API error");
      }
      const data: EvaluationResponse = await response.json();
      setEvaluationResults(data.results);
    } catch (err) {
      setEvaluationError((err as Error).message);
    } finally {
      setEvaluationLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "evaluation" && evaluationResults.length === 0 && !evaluationLoading) {
      fetchEvaluation();
    }
  }, [activeTab]);

  useEffect(() => {
    if (!sidebarOpen) {
      setChatDropdownOpen(false);
    }
  }, [sidebarOpen]);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : "collapsed"}`} onClick={() => !sidebarOpen && setSidebarOpen(true)}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <div className="brand-avatar">
              L
            </div>
            <div>
              <div className="brand-title">LUMINAL</div>
              <div className="brand-subtitle">AI CHATBOX</div>
            </div>
            <button
              className="sidebar-toggle"
              type="button"
              onClick={() => {
                setSidebarOpen(!sidebarOpen);
                if (sidebarOpen) {
                  setChatDropdownOpen(false);
                }
              }}
              aria-label={sidebarOpen ? "Đóng menu" : "Mở menu"}
            >
              <HugeiconsIcon icon={SquareArrowLeft02Icon} size={18} />
            </button>
          </div>

          <button className="new-chat-button" onClick={createNewSession}>
          Cuộc chat mới
          </button>
        </div>

        <div className="sidebar-section">
          <p className="sidebar-section-title">Danh mục</p>
          <nav className="sidebar-nav">
            <button
              type="button"
              className={activeTab === "chat" ? "nav-button active" : "nav-button"}
              onClick={() => {
                setActiveTab("chat");
                setChatDropdownOpen((prev) => !prev);
              }}
            >
              <span className="nav-icon">
                <HugeiconsIcon icon={ChatIcon} size={18} />
              </span>
              <span className="nav-label">Trò chuyện</span>
            </button>
            {chatDropdownOpen && (
              <div className="session-dropdown">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={session.id === activeSessionId ? "session-item-wrapper active" : "session-item-wrapper"}
                  >
                    <button
                      type="button"
                      className={session.id === activeSessionId ? "session-item active" : "session-item"}
                      onClick={() => loadSession(session.id)}
                    >
                      {session.name}
                    </button>
                    <button
                      type="button"
                      className="session-delete-button"
                      onClick={(event) => {
                        event.stopPropagation();
                        deleteSession(session.id);
                      }}
                      aria-label={`Xóa ${session.name}`}
                    >
                      <HugeiconsIcon icon={Delete02Icon} size={16} />
                    </button>
                  </div>
                ))}
                <button type="button" className="session-item add-session" onClick={createNewSession}>
                  + Tạo session mới
                </button>
              </div>
            )}
          <button
            type="button"
            className={activeTab === "evaluation" ? "nav-button active" : "nav-button"}
            onClick={() => {
              setActiveTab("evaluation");
              setChatDropdownOpen(false);
            }}
          >
            <span className="nav-icon">
              <HugeiconsIcon icon={ChartAnalysisIcon} size={18} />
            </span>
            <span className="nav-label">Tóm tắt đánh giá</span>
          </button>
          <button
            type="button"
            className={activeTab === "settings" ? "nav-button active" : "nav-button"}
            onClick={() => setActiveTab("settings")}
          >
            <span className="nav-icon">
              <HugeiconsIcon icon={Settings01Icon} size={18} />
            </span>
            <span className="nav-label">Cài đặt</span>
          </button>
        </nav>
        </div>

        <div className="sidebar-footer">
          <div className="profile-card">
            <div className="profile-icon">
              <HugeiconsIcon icon={UserIcon} size={20} />
            </div>
            <div>
              <div className="profile-name">Người dùng</div>
              <div className="profile-role">Miễn phí</div>
            </div>
          </div>
        </div>
      </aside>

      <main className="content-area">
        {activeTab === "chat" && (
          <section className="chat-screen">
            <div className="chat-topbar">
              <div>
                <p className="chat-topic">{sessions.find((session) => session.id === activeSessionId)?.name || "Trò chuyện"}</p>
              </div>
            </div>

            <div className="chat-messages">
              {messages.map((message, index) => (
                <div key={index} className={`message-row ${message.role}`}>
                  {message.role === "assistant" && <div className="message-avatar assistant-avatar">L</div>}
                  <div className="message-bubble">
                    <div className="message-header">
                      <span>{message.role === "assistant" ? "LUMINAL" : "Bạn"}</span>
                      <span>{message.time}</span>
                    </div>
                    <div className="message-body">{message.content}</div>
                  </div>
                  {message.role === "user" && <div className="message-avatar user-avatar">Y</div>}
                </div>
              ))}
            </div>

            <div className="chat-footer">
              <div className="input-panel">
                <div className="input-wrapper">
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void sendMessage();
                      }
                    }}
                    placeholder="Hỏi LUMINAL một câu hỏi..."
                    rows={2}
                  />
                </div>
                <button className="action-send" type="button" disabled={loading || !question.trim()} onClick={() => void sendMessage()}>
                  ➤
                </button>
              </div>
              <div className="chat-quote">
                Create by Văn Quyền, Thiện Quý, Đăng Đạo, Huy Thực, Đình Minh
              </div>
            </div>
          </section>
        )}

        {activeTab === "evaluation" && (
          <section className="evaluation-screen">
            <div className="evaluation-header">
              <div>
                <h1>Tóm tắt đánh giá</h1>
                <p>Hiệu suất hệ thống theo các hình thức truy xuất.</p>
              </div>
              <div className="evaluation-range">
                <button className="range-button active">7 ngày gần nhất</button>
                <button className="range-button">30 ngày</button>
              </div>
            </div>

            {evaluationLoading ? (
              <div className="empty-state">Đang tải dữ liệu đánh giá...</div>
            ) : evaluationError ? (
              <div className="warning-block">{evaluationError}</div>
            ) : evaluationResults.length === 0 ? (
              <div className="empty-state">Không có dữ liệu đánh giá. Nhấn làm mới để tải lại.</div>
            ) : (
              <>
                <div className="evaluation-cards-row">
                  {evaluationResults.map((item) => (
                    <div key={item.name} className="evaluation-card">
                      <div className="card-label">{item.name}</div>
                      <div className="card-metrics">
                        <div className="card-metric-row">
                          <span>Top-k</span>
                          <strong>{item.top_k}</strong>
                        </div>
                        <div className="card-metric-row">
                          <span>Hit@k</span>
                          <strong>{(item.hit * 100).toFixed(4)}</strong>
                        </div>
                        <div className="stat-row">
                          <span>Recall@k</span>
                          <strong>{item.recall.toFixed(4)}</strong>
                        </div>
                        <div className="progress-pill">
                          <div className="progress-fill" style={{ width: `${Math.min(item.recall * 100, 100)}%` }} />
                        </div>
                        <div className="stat-row">
                          <span>MRR</span>
                          <strong>{item.mrr.toFixed(4)}</strong>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="performance-section">
                  <div className="performance-header">
                    <div>
                      <h2>Xu hướng hiệu suất</h2>
                      <p>Xem xét hiệu suất truy xuất qua các khoảng đánh giá gần đây.</p>
                    </div>
                    <div className="trend-filters">
                      <button className="filter-pill active">Recall</button>
                      <button className="filter-pill">Hit@k</button>
                      <button className="filter-pill">MRR</button>
                    </div>
                  </div>

                  <div className="trend-card">
                    <div className="trend-graph">
                      <div className="trend-bar" style={{ height: "42%" }} />
                      <div className="trend-bar" style={{ height: "50%" }} />
                      <div className="trend-bar" style={{ height: "62%" }} />
                      <div className="trend-bar" style={{ height: "70%" }} />
                      <div className="trend-bar" style={{ height: "82%" }} />
                      <div className="trend-bar" style={{ height: "88%" }} />
                    </div>
                  </div>
                </div>
              </>
            )}
          </section>
        )}

        {activeTab === "settings" && (
          <section className="settings-screen">
            <h1>Cài đặt</h1>
            <p>Nội dung trang cài đặt sẽ được thêm vào ở đây.</p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
