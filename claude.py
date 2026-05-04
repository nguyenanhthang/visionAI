"""
Claude Opus AI Chat Application - Full Featured
Built with PySide6 + Anthropic SDK
Author: GitHub Copilot
"""
from openai import OpenAI
import sys
import os
import json
import base64
import mimetypes
import datetime
import markdown
from pathlib import Path
from threading import Thread

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QFileDialog, QDialog, QFormLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QMessageBox, QToolBar, QStatusBar, QMenu, QSystemTrayIcon,
    QScrollArea, QFrame, QSizePolicy, QTextBrowser, QToolButton, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QObject, QSize, QThread, QTimer
from PySide6.QtGui import (
    QAction, QIcon, QFont, QPixmap, QColor, QPalette, QKeySequence,
    QTextCursor, QDesktopServices, QShortcut
)

# import anthropic


# ========================= STYLES =========================

DARK_STYLE = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
    font-size: 14px;
}
QSplitter::handle {
    background-color: #16213e;
    width: 2px;
}

/* Sidebar */
#sidebar {
    background-color: #16213e;
    border-right: 1px solid #0f3460;
}
#sidebar QLabel {
    color: #e94560;
    font-size: 18px;
    font-weight: bold;
    padding: 15px;
}
#sidebar QPushButton {
    background-color: #e94560;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 15px;
    font-weight: bold;
    margin: 5px 10px;
}
#sidebar QPushButton:hover {
    background-color: #ff6b6b;
}
QListWidget {
    background-color: #16213e;
    border: none;
    padding: 5px;
}
QListWidget::item {
    background-color: #1a1a2e;
    color: #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    margin: 3px 5px;
}
QListWidget::item:selected {
    background-color: #0f3460;
    border-left: 3px solid #e94560;
}
QListWidget::item:hover {
    background-color: #0f3460;
}

/* Chat Area */
#chatArea {
    background-color: #1a1a2e;
}
QTextBrowser {
    background-color: #1a1a2e;
    border: none;
    padding: 15px;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.6;
}

/* Input Area */
#inputFrame {
    background-color: #16213e;
    border: 2px solid #0f3460;
    border-radius: 12px;
    margin: 10px 15px;
    padding: 8px;
}
#inputFrame:focus-within {
    border-color: #e94560;
}
QLineEdit {
    background-color: transparent;
    border: none;
    color: #e0e0e0;
    font-size: 15px;
    padding: 8px;
}
QLineEdit::placeholder {
    color: #666;
}

/* Buttons */
QPushButton#sendBtn {
    background-color: #e94560;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 14px;
    min-width: 80px;
}
QPushButton#sendBtn:hover {
    background-color: #ff6b6b;
}
QPushButton#sendBtn:disabled {
    background-color: #444;
}
QPushButton#attachBtn {
    background-color: transparent;
    border: 1px solid #0f3460;
    border-radius: 20px;
    padding: 8px 15px;
    color: #e0e0e0;
}
QPushButton#attachBtn:hover {
    background-color: #0f3460;
}

/* Toolbar */
QToolBar {
    background-color: #16213e;
    border-bottom: 1px solid #0f3460;
    padding: 5px;
    spacing: 5px;
}
QToolBar QToolButton {
    background-color: transparent;
    color: #e0e0e0;
    border: none;
    border-radius: 5px;
    padding: 8px 12px;
    font-size: 13px;
}
QToolBar QToolButton:hover {
    background-color: #0f3460;
}

/* StatusBar */
QStatusBar {
    background-color: #16213e;
    color: #888;
    border-top: 1px solid #0f3460;
}

/* Dialog */
QDialog {
    background-color: #1a1a2e;
}
QDialog QLabel {
    color: #e0e0e0;
}
QDialog QLineEdit, QDialog QComboBox, QDialog QSpinBox, QDialog QDoubleSpinBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 8px;
    color: #e0e0e0;
}
QDialog QPushButton {
    background-color: #e94560;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 20px;
    font-weight: bold;
}
QDialog QPushButton:hover {
    background-color: #ff6b6b;
}

/* ScrollBar */
QScrollBar:vertical {
    background-color: #1a1a2e;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background-color: #0f3460;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #e94560;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QComboBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 5px 10px;
    color: #e0e0e0;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #e0e0e0;
    selection-background-color: #0f3460;
}
"""

CHAT_HTML_STYLE = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; color: #e0e0e0; }
    .msg-user {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 60px 8px 15%;
        color: #fff;
        word-wrap: break-word;
    }
    .msg-assistant {
        background: linear-gradient(135deg, #1a1a2e, #222244);
        border: 1px solid #0f3460;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 8px 15% 8px 60px;
        margin-left: 5px;
        color: #e0e0e0;
        word-wrap: break-word;
    }
    .msg-system {
        text-align: center;
        color: #888;
        font-style: italic;
        margin: 10px 20%;
        font-size: 12px;
    }
    .sender {
        font-weight: bold;
        font-size: 12px;
        margin-bottom: 4px;
        color: #e94560;
    }
    .sender-user {
        text-align: right;
        color: #4fc3f7;
    }
    .time {
        font-size: 10px;
        color: #666;
        margin-top: 4px;
    }
    .time-user { text-align: right; }
    pre {
        background-color: #0d1117;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px;
        overflow-x: auto;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 13px;
        color: #e6edf3;
    }
    code {
        background-color: #0d1117;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 13px;
        color: #ff7b72;
    }
    a { color: #58a6ff; }
    img { max-width: 300px; border-radius: 8px; margin: 5px 0; }
    ul, ol { padding-left: 20px; }
    blockquote {
        border-left: 3px solid #e94560;
        padding-left: 12px;
        color: #aaa;
        margin: 8px 0;
    }
    table { border-collapse: collapse; margin: 8px 0; }
    th, td { border: 1px solid #333; padding: 6px 12px; }
    th { background-color: #0f3460; }
</style>
"""


# ===================== WORKER THREAD =====================

class StreamWorker(QObject):
    """Worker để stream response từ Claude API trong background thread."""
    token_received = Signal(str)
    stream_finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, client, messages, model, max_tokens, temperature, system_prompt):
        super().__init__()
        self.client = client
        self.messages = messages
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            full_response = ""
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": self.messages,
                "stream": True,
            }
            if self.system_prompt:
                kwargs["system"] = self.system_prompt

            with self.client.messages.stream(**{k: v for k, v in kwargs.items() if k != "stream"}) as stream:
                for text in stream.text_stream:
                    if self._is_cancelled:
                        break
                    full_response += text
                    self.token_received.emit(text)

            if not self._is_cancelled:
                self.stream_finished.emit(full_response)
        except Exception as e:
            self.error_occurred.emit(str(e))


# =================== SETTINGS DIALOG ====================

class SettingsDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Cài đặt")
        self.setMinimumSize(450, 400)
        self.settings = settings or {}

        layout = QFormLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-ant-api...")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setText(self.settings.get("api_key", ""))
        layout.addRow("🔑 API Key:", self.api_key_input)

        # Model
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "claude-opus-4-0-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ])
        current_model = self.settings.get("model", "claude-opus-4-0-20250514")
        idx = self.model_combo.findText(current_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setEditText(current_model)
        self.model_combo.setEditable(True)
        layout.addRow("🤖 Model:", self.model_combo)

        # Max Tokens
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 128000)
        self.max_tokens_spin.setValue(self.settings.get("max_tokens", 4096))
        self.max_tokens_spin.setSingleStep(256)
        layout.addRow("📝 Max Tokens:", self.max_tokens_spin)

        # Temperature
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setValue(self.settings.get("temperature", 0.7))
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setDecimals(2)
        layout.addRow("🌡️ Temperature:", self.temp_spin)

        # System Prompt
        self.system_prompt = QTextEdit()
        self.system_prompt.setPlaceholderText("Nhập system prompt (tùy chọn)...")
        self.system_prompt.setText(self.settings.get("system_prompt", ""))
        self.system_prompt.setMaximumHeight(120)
        layout.addRow("📋 System Prompt:", self.system_prompt)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Lưu")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("❌ Hủy")
        cancel_btn.setStyleSheet("background-color: #555;")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)

    def get_settings(self):
        return {
            "api_key": self.api_key_input.text().strip(),
            "model": self.model_combo.currentText(),
            "max_tokens": self.max_tokens_spin.value(),
            "temperature": self.temp_spin.value(),
            "system_prompt": self.system_prompt.toPlainText().strip(),
        }


# =================== MAIN WINDOW ========================

class ClaudeChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Claude Opus AI Chat")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)

        # State
        self.conversations = {}  # {id: {"title": str, "messages": list}}
        self.current_conv_id = None
        self.attached_files = []  # list of {"path": str, "type": str, "data": str}
        self.is_streaming = False
        self.stream_worker = None
        self.stream_thread = None

        # Settings
        self.settings = self._load_settings()
        self.client = None
        self._init_client()

        # Build UI
        self._build_ui()
        self._load_conversations()

        # Nếu chưa có conversation, tạo mới
        if not self.conversations:
            self._new_conversation()

        self.statusBar().showMessage("✅ Sẵn sàng chat với Claude Opus!")

    def _init_client(self):
        api_key = self.settings.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            self.client = OpenAI(api_key="your_openai_key")
        else:
            self.client = None

    def _load_settings(self):
        settings_path = Path.home() / ".claude_chat_settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "api_key": "",
            "model": "claude-opus-4-0-20250514",
            "max_tokens": 4096,
            "temperature": 0.7,
            "system_prompt": "",
        }

    def _save_settings(self):
        settings_path = Path.home() / ".claude_chat_settings.json"
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2)

    def _conversations_path(self):
        p = Path.home() / ".claude_chat_conversations.json"
        return p

    def _load_conversations(self):
        path = self._conversations_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
                for conv_id, conv in self.conversations.items():
                    item = QListWidgetItem(f"💬 {conv['title']}")
                    item.setData(Qt.UserRole, conv_id)
                    self.conv_list.addItem(item)
                if self.conversations:
                    first_id = list(self.conversations.keys())[0]
                    self._switch_conversation(first_id)
                    self.conv_list.setCurrentRow(0)
            except Exception:
                pass

    def _save_conversations(self):
        path = self._conversations_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ===== SIDEBAR =====
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo
        logo_label = QLabel("🤖 Claude Chat")
        logo_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo_label)

        # New chat button
        new_chat_btn = QPushButton("➕ Cuộc hội thoại mới")
        new_chat_btn.clicked.connect(self._new_conversation)
        sidebar_layout.addWidget(new_chat_btn)

        # Conversation list
        self.conv_list = QListWidget()
        self.conv_list.itemClicked.connect(self._on_conv_clicked)
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self._conv_context_menu)
        sidebar_layout.addWidget(self.conv_list)

        splitter.addWidget(sidebar)

        # ===== CHAT AREA =====
        chat_widget = QWidget()
        chat_widget.setObjectName("chatArea")
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))

        self.model_label = QLabel(f"  Model: {self.settings.get('model', 'N/A')}  ")
        self.model_label.setStyleSheet("color: #e94560; font-weight: bold;")
        toolbar.addWidget(self.model_label)

        toolbar.addSeparator()

        settings_action = QAction("⚙️ Cài đặt", self)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)

        export_action = QAction("📤 Export Chat", self)
        export_action.triggered.connect(self._export_chat)
        toolbar.addAction(export_action)

        clear_action = QAction("🗑️ Xóa Chat", self)
        clear_action.triggered.connect(self._clear_chat)
        toolbar.addAction(clear_action)

        chat_layout.addWidget(toolbar)

        # Chat display
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)

        # Attachment preview
        self.attach_label = QLabel("")
        self.attach_label.setStyleSheet("color: #4fc3f7; padding: 5px 15px; font-size: 12px;")
        self.attach_label.setVisible(False)
        chat_layout.addWidget(self.attach_label)

        # Input area
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(5, 5, 5, 5)
        input_layout.setSpacing(8)

        # Attach button
        attach_btn = QPushButton("📎")
        attach_btn.setObjectName("attachBtn")
        attach_btn.setToolTip("Đính kèm file/ảnh")
        attach_btn.clicked.connect(self._attach_file)
        input_layout.addWidget(attach_btn)

        # Text input
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Nhập tin nhắn... (Enter để gửi, Ctrl+Enter cho dòng mới)")
        self.input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.input_field)

        # Send button
        self.send_btn = QPushButton("Gửi ➤")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)

        # Stop button
        self.stop_btn = QPushButton("⏹ Dừng")
        self.stop_btn.setObjectName("sendBtn")
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._stop_streaming)
        input_layout.addWidget(self.stop_btn)

        chat_layout.addWidget(input_frame)

        splitter.addWidget(chat_widget)
        splitter.setStretchFactor(1, 1)

        # StatusBar
        self.setStatusBar(QStatusBar())

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+N"), self, self._new_conversation)
        QShortcut(QKeySequence("Ctrl+E"), self, self._export_chat)
        QShortcut(QKeySequence("Ctrl+,"), self, self._open_settings)

    # ============== CONVERSATION MANAGEMENT ===============

    def _new_conversation(self):
        conv_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        title = f"Chat {len(self.conversations) + 1}"
        self.conversations[conv_id] = {"title": title, "messages": []}
        item = QListWidgetItem(f"💬 {title}")
        item.setData(Qt.UserRole, conv_id)
        self.conv_list.insertItem(0, item)
        self.conv_list.setCurrentRow(0)
        self._switch_conversation(conv_id)
        self._save_conversations()

    def _switch_conversation(self, conv_id):
        self.current_conv_id = conv_id
        self._refresh_chat_display()

    def _on_conv_clicked(self, item):
        conv_id = item.data(Qt.UserRole)
        self._switch_conversation(conv_id)

    def _conv_context_menu(self, pos):
        item = self.conv_list.itemAt(pos)
        if not item:
            return
        conv_id = item.data(Qt.UserRole)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #16213e; color: #e0e0e0; border: 1px solid #0f3460; }
            QMenu::item:selected { background-color: #0f3460; }
        """)

        rename_action = menu.addAction("✏️ Đổi tên")
        delete_action = menu.addAction("🗑️ Xóa")

        action = menu.exec(self.conv_list.mapToGlobal(pos))
        if action == rename_action:
            new_name, ok = QInputDialog.getText(self, "Đổi tên", "Tên mới:", text=self.conversations[conv_id]["title"])
            if ok and new_name:
                self.conversations[conv_id]["title"] = new_name
                item.setText(f"💬 {new_name}")
                self._save_conversations()
        elif action == delete_action:
            reply = QMessageBox.question(self, "Xóa", "Bạn có chắc muốn xóa cuộc hội thoại này?")
            if reply == QMessageBox.Yes:
                del self.conversations[conv_id]
                self.conv_list.takeItem(self.conv_list.row(item))
                if self.current_conv_id == conv_id:
                    if self.conversations:
                        first_id = list(self.conversations.keys())[0]
                        self._switch_conversation(first_id)
                    else:
                        self._new_conversation()
                self._save_conversations()

    # ============== CHAT DISPLAY ==========================

    def _refresh_chat_display(self):
        if not self.current_conv_id or self.current_conv_id not in self.conversations:
            self.chat_display.setHtml("")
            return

        messages = self.conversations[self.current_conv_id]["messages"]
        html = f"<html><head>{CHAT_HTML_STYLE}</head><body>"

        if not messages:
            html += """
            <div style="text-align:center; margin-top: 100px;">
                <div style="font-size: 48px;">🤖</div>
                <h2 style="color: #e94560;">Xin chào! Tôi là Claude Opus</h2>
                <p style="color: #888;">Hãy hỏi tôi bất cứ điều gì. Tôi có thể giúp bạn viết code,
                phân tích, sáng tạo, và nhiều hơn nữa!</p>
                <p style="color: #666; font-size: 12px;">
                    📎 Đính kèm file/ảnh &nbsp;|&nbsp; ⚙️ Ctrl+, để cài đặt &nbsp;|&nbsp;
                    ➕ Ctrl+N tạo chat mới
                </p>
            </div>
            """

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("display_content", msg.get("content", ""))
            timestamp = msg.get("timestamp", "")

            if role == "user":
                sender_html = f'<div class="sender sender-user">👤 Bạn</div>'
                time_html = f'<div class="time time-user">{timestamp}</div>'
                content_html = self._format_content(content)
                html += f'<div class="msg-user">{sender_html}{content_html}{time_html}</div>'
            elif role == "assistant":
                sender_html = f'<div class="sender">🤖 Claude</div>'
                time_html = f'<div class="time">{timestamp}</div>'
                content_html = self._format_content(content)
                html += f'<div class="msg-assistant">{sender_html}{content_html}{time_html}</div>'
            elif role == "system":
                html += f'<div class="msg-system">{content}</div>'

        html += "</body></html>"
        self.chat_display.setHtml(html)

        # Auto scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)

    def _format_content(self, content):
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block["text"])
                    elif block.get("type") == "image":
                        parts.append("[📷 Ảnh đính kèm]")
                else:
                    parts.append(str(block))
            content = "\n".join(parts)

        # Convert markdown to HTML
        try:
            html = markdown.markdown(
                str(content),
                extensions=["fenced_code", "codehilite", "tables", "nl2br", "sane_lists"]
            )
        except Exception:
            html = str(content).replace("\n", "<br>")
        return html

    # ============== FILE ATTACHMENT =======================

    def _attach_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file đính kèm",
            "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.gif *.webp);;Text (*.txt *.py *.js *.md *.json *.csv)"
        )
        if not file_path:
            return

        mime_type, _ = mimetypes.guess_type(file_path)
        file_name = os.path.basename(file_path)

        if mime_type and mime_type.startswith("image/"):
            with open(file_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            self.attached_files.append({
                "type": "image",
                "media_type": mime_type,
                "data": data,
                "name": file_name
            })
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_data = f.read()
                self.attached_files.append({
                    "type": "text",
                    "data": text_data,
                    "name": file_name
                })
            except Exception:
                with open(file_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                self.attached_files.append({
                    "type": "document",
                    "data": data,
                    "name": file_name
                })

        names = ", ".join([f["name"] for f in self.attached_files])
        self.attach_label.setText(f"📎 Đính kèm: {names}  (Click 📎 để thêm)")
        self.attach_label.setVisible(True)
        self.statusBar().showMessage(f"📎 Đã đính kèm: {file_name}")

    # ============== SEND MESSAGE ==========================

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text and not self.attached_files:
            return

        if not self.client:
            QMessageBox.warning(self, "Lỗi", "Chưa cấu hình API Key!\nVào ⚙️ Cài đặt để nhập API Key.")
            self._open_settings()
            return

        if self.is_streaming:
            return

        # Build content
        content_blocks = []
        display_parts = []

        # Add attached files
        for f in self.attached_files:
            if f["type"] == "image":
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f["media_type"],
                        "data": f["data"],
                    }
                })
                display_parts.append(f"📷 [{f['name']}]")
            elif f["type"] == "text":
                content_blocks.append({
                    "type": "text",
                    "text": f"📄 File: {f['name']}\n```\n{f['data']}\n```"
                })
                display_parts.append(f"📄 [{f['name']}]")

        if text:
            content_blocks.append({"type": "text", "text": text})
            display_parts.append(text)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Add message to conversation
        user_message = {
            "role": "user",
            "content": content_blocks if len(content_blocks) > 1 else text,
            "display_content": "\n".join(display_parts),
            "timestamp": timestamp,
        }
        self.conversations[self.current_conv_id]["messages"].append(user_message)

        # Auto-rename conversation based on first message
        if len(self.conversations[self.current_conv_id]["messages"]) == 1:
            short_title = text[:40] + ("..." if len(text) > 40 else "")
            self.conversations[self.current_conv_id]["title"] = short_title
            for i in range(self.conv_list.count()):
                item = self.conv_list.item(i)
                if item.data(Qt.UserRole) == self.current_conv_id:
                    item.setText(f"💬 {short_title}")
                    break

        # Clear input
        self.input_field.clear()
        self.attached_files.clear()
        self.attach_label.setVisible(False)
        self._refresh_chat_display()

        # Build API messages (exclude display_content and timestamp)
        api_messages = []
        for msg in self.conversations[self.current_conv_id]["messages"]:
            api_msg = {"role": msg["role"], "content": msg["content"]}
            api_messages.append(api_msg)

        # Start streaming
        self._start_streaming(api_messages)

    def _start_streaming(self, api_messages):
        self.is_streaming = True
        self.send_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.input_field.setEnabled(False)
        self.statusBar().showMessage("🤖 Claude đang suy nghĩ...")

        # Add placeholder for assistant message
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "display_content": "",
            "timestamp": timestamp,
        }
        self.conversations[self.current_conv_id]["messages"].append(assistant_msg)

        # Worker
        self.stream_worker = StreamWorker(
            client=self.client,
            messages=api_messages,
            model=self.settings.get("model", "claude-opus-4-0-20250514"),
            max_tokens=self.settings.get("max_tokens", 4096),
            temperature=self.settings.get("temperature", 0.7),
            system_prompt=self.settings.get("system_prompt", ""),
        )

        self.stream_thread = QThread()
        self.stream_worker.moveToThread(self.stream_thread)

        self.stream_thread.started.connect(self.stream_worker.run)
        self.stream_worker.token_received.connect(self._on_token_received)
        self.stream_worker.stream_finished.connect(self._on_stream_finished)
        self.stream_worker.error_occurred.connect(self._on_stream_error)
        self.stream_worker.stream_finished.connect(self.stream_thread.quit)
        self.stream_worker.error_occurred.connect(self.stream_thread.quit)

        self.stream_thread.start()

    def _on_token_received(self, token):
        if self.current_conv_id and self.conversations[self.current_conv_id]["messages"]:
            last_msg = self.conversations[self.current_conv_id]["messages"][-1]
            if last_msg["role"] == "assistant":
                last_msg["content"] += token
                last_msg["display_content"] += token
                self._refresh_chat_display()

    def _on_stream_finished(self, full_response):
        self.is_streaming = False
        self.send_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        self._save_conversations()

        # Count tokens roughly
        token_count = len(full_response.split())
        self.statusBar().showMessage(f"✅ Hoàn thành! (~{token_count} từ)")

    def _on_stream_error(self, error_msg):
        self.is_streaming = False
        self.send_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

        # Remove empty assistant message
        if self.conversations[self.current_conv_id]["messages"]:
            last_msg = self.conversations[self.current_conv_id]["messages"][-1]
            if last_msg["role"] == "assistant" and not last_msg["content"]:
                self.conversations[self.current_conv_id]["messages"].pop()

        # Add error as system message
        self.conversations[self.current_conv_id]["messages"].append({
            "role": "system",
            "content": f"❌ Lỗi: {error_msg}",
            "display_content": f"❌ Lỗi: {error_msg}",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        })
        self._refresh_chat_display()
        self.statusBar().showMessage(f"❌ Lỗi: {error_msg[:80]}")

    def _stop_streaming(self):
        if self.stream_worker:
            self.stream_worker.cancel()
        self.is_streaming = False
        self.send_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        self._save_conversations()
        self.statusBar().showMessage("⏹ Đã dừng streaming")

    # ============== SETTINGS ==============================

    def _open_settings(self):
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec() == QDialog.Accepted:
            self.settings = dialog.get_settings()
            self._save_settings()
            self._init_client()
            self.model_label.setText(f"  Model: {self.settings.get('model', 'N/A')}  ")
            self.statusBar().showMessage("✅ Đã lưu cài đặt!")

    # ============== EXPORT / CLEAR ========================

    def _export_chat(self):
        if not self.current_conv_id:
            return
        messages = self.conversations[self.current_conv_id]["messages"]
        if not messages:
            QMessageBox.information(self, "Export", "Không có tin nhắn để export!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Chat",
            f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "Markdown (*.md);;Text (*.txt);;JSON (*.json)"
        )
        if not file_path:
            return

        if file_path.endswith(".json"):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        else:
            lines = []
            title = self.conversations[self.current_conv_id]["title"]
            lines.append(f"# {title}\n")
            lines.append(f"_Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n---\n")
            for msg in messages:
                role = "👤 **Bạn**" if msg["role"] == "user" else "🤖 **Claude**"
                content = msg.get("display_content", msg.get("content", ""))
                timestamp = msg.get("timestamp", "")
                lines.append(f"### {role} ({timestamp})\n\n{content}\n\n---\n")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        self.statusBar().showMessage(f"📤 Đã export: {file_path}")

    def _clear_chat(self):
        if not self.current_conv_id:
            return
        reply = QMessageBox.question(self, "Xóa Chat", "Bạn có chắc muốn xóa toàn bộ tin nhắn?")
        if reply == QMessageBox.Yes:
            self.conversations[self.current_conv_id]["messages"] = []
            self._refresh_chat_display()
            self._save_conversations()
            self.statusBar().showMessage("🗑️ Đã xóa chat")

    # ============== CLOSE EVENT ===========================

    def closeEvent(self, event):
        self._save_conversations()
        self._save_settings()
        if self.stream_worker:
            self.stream_worker.cancel()
        if self.stream_thread and self.stream_thread.isRunning():
            self.stream_thread.quit()
            self.stream_thread.wait(2000)
        event.accept()


# ======================== MAIN ============================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1a1a2e"))
    palette.setColor(QPalette.WindowText, QColor("#e0e0e0"))
    palette.setColor(QPalette.Base, QColor("#16213e"))
    palette.setColor(QPalette.AlternateBase, QColor("#1a1a2e"))
    palette.setColor(QPalette.Text, QColor("#e0e0e0"))
    palette.setColor(QPalette.Button, QColor("#16213e"))
    palette.setColor(QPalette.ButtonText, QColor("#e0e0e0"))
    palette.setColor(QPalette.Highlight, QColor("#e94560"))
    palette.setColor(QPalette.HighlightedText, QColor("#fff"))
    app.setPalette(palette)

    window = ClaudeChatApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()