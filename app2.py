import os
import json
import base64
import random
import re
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import yaml
import pandas as pd
import altair as alt
from pypdf import PdfReader

try:
    from docx import Document  # python-docx
except ImportError:
    Document = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except ImportError:
    canvas = None
    letter = None

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx


# ============================================================
# 0) Streamlit page config
# ============================================================
st.set_page_config(
    page_title="Antigravity AI Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1) Models & Providers
# ============================================================
ALL_MODELS = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    # Anthropic (examples; can extend freely)
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    # xAI Grok
    "grok-4-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
]

OPENAI_MODELS = {"gpt-4o-mini", "gpt-4.1-mini"}
GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
}
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
}
GROK_MODELS = {"grok-4-fast-reasoning", "grok-4-1-fast-non-reasoning"}


def get_provider(model: str) -> str:
    if model in OPENAI_MODELS:
        return "openai"
    if model in GEMINI_MODELS:
        return "gemini"
    if model in ANTHROPIC_MODELS:
        return "anthropic"
    if model in GROK_MODELS:
        return "grok"
    raise ValueError(f"Unknown/unsupported model: {model}")


# ============================================================
# 2) i18n (English / zh-TW)
# ============================================================
I18N: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "Antigravity AI Workspace",
        "top_tagline": "A WOW workspace for agents, dashboards, notes, and art styles",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "language": "Language",
        "english": "English",
        "zh_tw": "Traditional Chinese (繁體中文)",
        "style_engine": "Style Engine",
        "painter_style": "Painter Style",
        "jackpot": "Jackpot",
        "global_settings": "Global Settings",
        "default_model": "Default Model",
        "default_max_tokens": "Default max_tokens",
        "temperature": "Temperature",
        "api_keys": "API Keys",
        "active_env": "Active (Env)",
        "missing": "Missing",
        "provided_session": "Provided (Session)",
        "agents_catalog": "Agents Catalog (agents.yaml)",
        "upload_agents_yaml": "Upload custom agents.yaml",
        "dashboard": "Dashboard",
        "tw_premarket": "TW Premarket Application",
        "fda_510k": "510(k) Intelligence",
        "pdf_md": "PDF → Markdown",
        "pipeline": "510(k) Review Pipeline",
        "note_keeper": "Note Keeper & Magics",
        "agents_config": "Agents Config Studio",
        "workflow_studio": "Agent Workflow Studio",
        "status_wall": "WOW Status Wall",
        "run_agent": "Run Agent",
        "prompt": "Prompt",
        "model": "Model",
        "input_text": "Input Text / Markdown",
        "output": "Output",
        "view_mode": "View mode",
        "markdown": "Markdown",
        "plain_text": "Plain text",
        "api_pulse": "API Connection Pulse",
        "token_meter": "Token Usage Meter",
        "agent_status": "Agent Status",
        "idle": "Idle",
        "thinking": "Thinking",
        "done": "Done",
        "error": "Error",
        "clear_history": "Clear history",
        "export_history": "Export history (CSV)",
        "note_upload": "Upload note file (.pdf/.txt/.md)",
        "note_paste": "Paste your notes (text/markdown)",
        "note_transform": "Transform to organized Markdown + coral keywords",
        "note_color": "Keyword color",
        "ai_magics": "AI Magics",
        "magic_keywords": "AI Keywords",
        "magic_summarize": "Summarize",
        "magic_polish": "Polisher",
        "magic_critique": "Critique",
        "magic_poet": "Poet Mode",
        "magic_translate": "Translate",
        "apply": "Apply",
        "run": "Run",
        "reset": "Reset",
        "step": "Step",
        "run_step": "Run step",
        "run_next": "Run next",
        "workflow_input": "Workflow Input",
        "workflow_output": "Workflow Output",
        "add_step": "Add step",
        "remove_step": "Remove last step",
        "load_defaults": "Load recommended workflow",
        "download_md": "Download Markdown",
    },
    "zh-tw": {
        "app_title": "Antigravity AI 工作空間",
        "top_tagline": "WOW 級：代理工作流、互動儀表板、筆記魔法、藝術主題",
        "theme": "主題",
        "light": "淺色",
        "dark": "深色",
        "language": "語言",
        "english": "英文",
        "zh_tw": "繁體中文",
        "style_engine": "風格引擎",
        "painter_style": "畫家風格",
        "jackpot": "拉霸",
        "global_settings": "全域設定",
        "default_model": "預設模型",
        "default_max_tokens": "預設 max_tokens",
        "temperature": "溫度(創造力)",
        "api_keys": "API 金鑰",
        "active_env": "已啟用（環境變數）",
        "missing": "缺少",
        "provided_session": "已提供（本次會話）",
        "agents_catalog": "代理目錄（agents.yaml）",
        "upload_agents_yaml": "上傳自訂 agents.yaml",
        "dashboard": "儀表板",
        "tw_premarket": "第二、三等級醫療器材查驗登記",
        "fda_510k": "510(k) 智能分析",
        "pdf_md": "PDF → Markdown",
        "pipeline": "510(k) 審查全流程",
        "note_keeper": "筆記助手與魔法",
        "agents_config": "代理設定工作室",
        "workflow_studio": "代理工作流工作室",
        "status_wall": "WOW 狀態牆",
        "run_agent": "執行代理",
        "prompt": "提示詞",
        "model": "模型",
        "input_text": "輸入（文字/Markdown）",
        "output": "輸出",
        "view_mode": "檢視模式",
        "markdown": "Markdown",
        "plain_text": "純文字",
        "api_pulse": "API 連線脈動",
        "token_meter": "Token 用量儀表",
        "agent_status": "代理狀態",
        "idle": "待命",
        "thinking": "思考中",
        "done": "完成",
        "error": "錯誤",
        "clear_history": "清除紀錄",
        "export_history": "匯出紀錄（CSV）",
        "note_upload": "上傳筆記檔（.pdf/.txt/.md）",
        "note_paste": "貼上筆記（文字/Markdown）",
        "note_transform": "整理成結構化 Markdown + 珊瑚色關鍵字",
        "note_color": "關鍵字顏色",
        "ai_magics": "AI 魔法",
        "magic_keywords": "AI 關鍵字",
        "magic_summarize": "摘要",
        "magic_polish": "潤稿",
        "magic_critique": "評論",
        "magic_poet": "詩人模式",
        "magic_translate": "翻譯",
        "apply": "套用",
        "run": "執行",
        "reset": "重置",
        "step": "步驟",
        "run_step": "執行本步驟",
        "run_next": "執行下一步",
        "workflow_input": "工作流輸入",
        "workflow_output": "工作流輸出",
        "add_step": "新增步驟",
        "remove_step": "刪除最後一步",
        "load_defaults": "載入建議工作流",
        "download_md": "下載 Markdown",
    },
}


def lang_code() -> str:
    return st.session_state.settings.get("language", "zh-tw")


def t(key: str) -> str:
    return I18N.get(lang_code(), I18N["en"]).get(key, key)


# ============================================================
# 3) Style Engine: 20 painter styles + Jackpot
# ============================================================
PAINTER_STYLES_20 = [
    "Van Gogh",
    "Picasso",
    "Monet",
    "Da Vinci",
    "Dali",
    "Mondrian",
    "Warhol",
    "Rembrandt",
    "Klimt",
    "Hokusai",
    "Munch",
    "O'Keeffe",
    "Basquiat",
    "Matisse",
    "Pollock",
    "Kahlo",
    "Hopper",
    "Magritte",
    "Cyberpunk",
    "Bauhaus",
]

# Theme tokens (CSS variables) per painter style
STYLE_TOKENS: Dict[str, Dict[str, str]] = {
    "Van Gogh": {
        "--bg1": "#0b1020",
        "--bg2": "#1f3b73",
        "--accent": "#f7c948",
        "--accent2": "#60a5fa",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Picasso": {
        "--bg1": "#2b2b2b",
        "--bg2": "#7c2d12",
        "--accent": "#f59e0b",
        "--accent2": "#a3e635",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Monet": {
        "--bg1": "#a1c4fd",
        "--bg2": "#c2e9fb",
        "--accent": "#2563eb",
        "--accent2": "#0ea5e9",
        "--card": "rgba(255,255,255,0.35)",
        "--border": "rgba(255,255,255,0.45)",
    },
    "Da Vinci": {
        "--bg1": "#f6f0d9",
        "--bg2": "#cbb38b",
        "--accent": "#7c2d12",
        "--accent2": "#1f2937",
        "--card": "rgba(255,255,255,0.35)",
        "--border": "rgba(17,24,39,0.18)",
    },
    "Dali": {
        "--bg1": "#0f172a",
        "--bg2": "#b91c1c",
        "--accent": "#fbbf24",
        "--accent2": "#38bdf8",
        "--card": "rgba(255,255,255,0.12)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Mondrian": {
        "--bg1": "#f8fafc",
        "--bg2": "#e2e8f0",
        "--accent": "#ef4444",
        "--accent2": "#2563eb",
        "--card": "rgba(255,255,255,0.60)",
        "--border": "rgba(0,0,0,0.18)",
    },
    "Warhol": {
        "--bg1": "#0b1020",
        "--bg2": "#6d28d9",
        "--accent": "#22c55e",
        "--accent2": "#f472b6",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Rembrandt": {
        "--bg1": "#07050a",
        "--bg2": "#2c1810",
        "--accent": "#f59e0b",
        "--accent2": "#fbbf24",
        "--card": "rgba(255,255,255,0.08)",
        "--border": "rgba(245,158,11,0.20)",
    },
    "Klimt": {
        "--bg1": "#0b1020",
        "--bg2": "#3b2f0b",
        "--accent": "#fbbf24",
        "--accent2": "#fde68a",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(251,191,36,0.25)",
    },
    "Hokusai": {
        "--bg1": "#061a2b",
        "--bg2": "#1e3a8a",
        "--accent": "#60a5fa",
        "--accent2": "#93c5fd",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(147,197,253,0.25)",
    },
    "Munch": {
        "--bg1": "#1f2937",
        "--bg2": "#7f1d1d",
        "--accent": "#fb7185",
        "--accent2": "#fde047",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "O'Keeffe": {
        "--bg1": "#fff7ed",
        "--bg2": "#fecdd3",
        "--accent": "#db2777",
        "--accent2": "#f97316",
        "--card": "rgba(255,255,255,0.55)",
        "--border": "rgba(219,39,119,0.18)",
    },
    "Basquiat": {
        "--bg1": "#111827",
        "--bg2": "#f59e0b",
        "--accent": "#22c55e",
        "--accent2": "#60a5fa",
        "--card": "rgba(255,255,255,0.12)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Matisse": {
        "--bg1": "#ffedd5",
        "--bg2": "#fde68a",
        "--accent": "#ea580c",
        "--accent2": "#2563eb",
        "--card": "rgba(255,255,255,0.60)",
        "--border": "rgba(234,88,12,0.20)",
    },
    "Pollock": {
        "--bg1": "#0b1020",
        "--bg2": "#111827",
        "--accent": "#f97316",
        "--accent2": "#22c55e",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.20)",
    },
    "Kahlo": {
        "--bg1": "#064e3b",
        "--bg2": "#7f1d1d",
        "--accent": "#fbbf24",
        "--accent2": "#22c55e",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Hopper": {
        "--bg1": "#0b1020",
        "--bg2": "#0f766e",
        "--accent": "#60a5fa",
        "--accent2": "#fbbf24",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Magritte": {
        "--bg1": "#0b1020",
        "--bg2": "#1d4ed8",
        "--accent": "#e2e8f0",
        "--accent2": "#fbbf24",
        "--card": "rgba(255,255,255,0.10)",
        "--border": "rgba(255,255,255,0.22)",
    },
    "Cyberpunk": {
        "--bg1": "#050816",
        "--bg2": "#1b0033",
        "--accent": "#22d3ee",
        "--accent2": "#a78bfa",
        "--card": "rgba(255,255,255,0.08)",
        "--border": "rgba(34,211,238,0.25)",
    },
    "Bauhaus": {
        "--bg1": "#f8fafc",
        "--bg2": "#e2e8f0",
        "--accent": "#111827",
        "--accent2": "#ef4444",
        "--card": "rgba(255,255,255,0.70)",
        "--border": "rgba(17,24,39,0.15)",
    },
}


def apply_style_engine(theme_mode: str, painter_style: str):
    """
    WOW UI: glassmorphism + style tokens + subtle animated background + neon/ink accents.
    """
    tokens = STYLE_TOKENS.get(painter_style, STYLE_TOKENS["Van Gogh"])
    is_dark = theme_mode.lower() == "dark"

    # Base variables
    text_color = "#e5e7eb" if is_dark else "#0f172a"
    subtext = "#cbd5e1" if is_dark else "#334155"
    card_text = "#f8fafc" if is_dark else "#0f172a"
    shadow = "0 18px 50px rgba(0,0,0,0.38)" if is_dark else "0 18px 50px rgba(2,6,23,0.18)"
    glass = "rgba(17,24,39,0.38)" if is_dark else "rgba(255,255,255,0.55)"

    # Pollock splatter overlay (lightweight CSS only)
    splatter = ""
    if painter_style == "Pollock":
        splatter = """
        body:before{
            content:"";
            position:fixed; inset:0;
            background:
              radial-gradient(circle at 10% 20%, rgba(249,115,22,0.18) 0 10%, transparent 12%),
              radial-gradient(circle at 70% 35%, rgba(34,197,94,0.18) 0 9%, transparent 11%),
              radial-gradient(circle at 40% 80%, rgba(96,165,250,0.18) 0 12%, transparent 14%),
              radial-gradient(circle at 85% 75%, rgba(244,114,182,0.16) 0 8%, transparent 10%);
            pointer-events:none;
            filter: blur(0.2px);
            mix-blend-mode: screen;
            opacity:0.85;
        }
        """

    css = f"""
    <style>
    :root {{
        {"".join([f"{k}:{v};" for k,v in tokens.items()])}
        --text: {text_color};
        --subtext: {subtext};
        --glass: {glass};
        --shadow: {shadow};
        --radius: 18px;
        --radius2: 26px;
        --coral: #FF7F50;
    }}

    /* Background */
    body {{
        color: var(--text);
        background: radial-gradient(1200px circle at 12% 8%, var(--bg2) 0%, transparent 55%),
                    radial-gradient(900px circle at 88% 18%, var(--accent2) 0%, transparent 50%),
                    linear-gradient(135deg, var(--bg1), var(--bg2));
        background-attachment: fixed;
    }}
    {splatter}

    /* Streamlit base tweaks */
    .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 3.5rem;
    }}

    /* WOW Top Hero */
    .wow-hero {{
        border-radius: var(--radius2);
        padding: 18px 18px;
        margin: 0 0 14px 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02));
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
    }}
    .wow-title {{
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin: 0;
        color: var(--text);
    }}
    .wow-subtitle {{
        margin: 6px 0 0 0;
        color: var(--subtext);
        font-size: 0.95rem;
    }}
    .wow-chips {{
        margin-top: 10px;
        display:flex;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .wow-chip {{
        display:inline-flex;
        align-items:center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        background: rgba(255,255,255,0.10);
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
        color: var(--text);
    }}

    /* Cards / glass containers */
    .wow-card {{
        border-radius: var(--radius);
        padding: 14px 16px;
        background: var(--glass);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
    }}
    .wow-card h3, .wow-card h4 {{
        margin: 0 0 8px 0;
    }}
    .wow-kpi {{
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 4px;
    }}
    .wow-muted {{
        color: var(--subtext);
        font-size: 0.92rem;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 999px !important;
        border: 1px solid var(--border) !important;
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: {"#0b1020" if not is_dark else "#0b1020"} !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 14px 35px rgba(0,0,0,0.25) !important;
    }}
    .stButton > button:hover {{
        filter: brightness(1.04);
        transform: translateY(-1px);
        transition: 120ms ease;
    }}

    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        background: rgba(255,255,255,{0.06 if is_dark else 0.55}) !important;
        color: var(--text) !important;
    }}

    /* Tabs */
    button[role="tab"] {{
        border-radius: 999px !important;
    }}

    /* Status dot */
    .dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.06);
    }}
    .dot-green {{ background: #22c55e; box-shadow: 0 0 18px rgba(34,197,94,0.55); }}
    .dot-red {{ background: #ef4444; box-shadow: 0 0 18px rgba(239,68,68,0.55); }}
    .dot-amber {{ background: #f59e0b; box-shadow: 0 0 18px rgba(245,158,11,0.55); }}

    /* Small badges */
    .wow-badge {{
        display:inline-flex;
        align-items:center;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.10);
        color: var(--text);
    }}

    /* Coral highlight helper */
    .coral {{
        color: var(--coral);
        font-weight: 800;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# 4) State init
# ============================================================
if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "theme": "Dark",
        "language": "zh-tw",  # "en" or "zh-tw"
        "painter_style": "Van Gogh",
        "model": "gpt-4o-mini",
        "max_tokens": 12000,
        "temperature": 0.2,
        "token_budget_est": 250_000,  # session visual budget
    }

if "history" not in st.session_state:
    st.session_state["history"] = []

if "api_keys" not in st.session_state:
    st.session_state["api_keys"] = {"openai": "", "gemini": "", "anthropic": "", "grok": ""}

if "workflow" not in st.session_state:
    st.session_state["workflow"] = {
        "steps": [],  # list of dicts: {agent_id, model, max_tokens, prompt, system_override?}
        "cursor": 0,
        "input": "",
        "outputs": [],  # per-step output (editable)
        "statuses": [],  # per-step status
    }


# ============================================================
# 5) API key logic (env first; hide if env exists)
# ============================================================
def env_key_present(env_var: str) -> bool:
    v = os.getenv(env_var, "")
    return bool(v and v.strip())


def get_api_key(provider: str, api_keys: Dict[str, str]) -> str:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "grok": "GROK_API_KEY",
    }
    env_var = mapping.get(provider, "")
    return api_keys.get(provider) or os.getenv(env_var) or ""


def api_status(provider: str) -> Tuple[str, str]:
    """
    returns: (status: 'env'|'session'|'missing', label)
    """
    mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "grok": "GROK_API_KEY",
    }
    env_var = mapping[provider]
    if env_key_present(env_var):
        return "env", t("active_env")
    if st.session_state["api_keys"].get(provider, "").strip():
        return "session", t("provided_session")
    return "missing", t("missing")


# ============================================================
# 6) LLM call router
# ============================================================
def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    api_keys: Optional[dict] = None,
) -> str:
    provider = get_provider(model)
    api_keys = api_keys or {}

    key = get_api_key(provider, api_keys)
    if not key:
        raise RuntimeError(f"Missing API key for provider: {provider}")

    if provider == "openai":
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_prompt or ""},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    if provider == "gemini":
        genai.configure(api_key=key)
        llm = genai.GenerativeModel(model)
        resp = llm.generate_content(
            (system_prompt or "").strip() + "\n\n" + (user_prompt or "").strip(),
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return resp.text

    if provider == "anthropic":
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            system=system_prompt or "",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt or ""}],
        )
        return resp.content[0].text

    if provider == "grok":
        with httpx.Client(base_url="https://api.x.ai/v1", timeout=90) as client:
            resp = client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": user_prompt or ""},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"Unsupported provider for model {model}")


# ============================================================
# 7) Generic helpers (pdf/text, logging, status UI)
# ============================================================
def log_event(tab: str, agent: str, model: str, tokens_est: int, meta: Optional[dict] = None):
    st.session_state["history"].append(
        {
            "tab": tab,
            "agent": agent,
            "model": model,
            "tokens_est": int(tokens_est),
            "ts": datetime.utcnow().isoformat(),
            "meta": meta or {},
        }
    )


def est_tokens(text: str) -> int:
    # rough estimate; OK for dashboard meter
    return max(1, int(len(text) / 4))


def extract_pdf_pages_to_text(file, start_page: int, end_page: int) -> str:
    reader = PdfReader(file)
    n = len(reader.pages)
    start = max(0, start_page - 1)
    end = min(n, end_page)
    texts = []
    for i in range(start, end):
        try:
            texts.append(reader.pages[i].extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts).strip()


def create_pdf_from_text(text: str) -> bytes:
    if canvas is None or letter is None:
        raise RuntimeError("Missing 'reportlab'. Add 'reportlab' to requirements.txt to export PDF.")
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 72
    line_height = 14
    y = height - margin
    for line in (text or "").splitlines():
        if y < margin:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line[:2000])
        y -= line_height
    c.save()
    buf.seek(0)
    return buf.getvalue()


def show_pdf(pdf_bytes: bytes, height: int = 600):
    if not pdf_bytes:
        return
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}"></iframe>""",
        unsafe_allow_html=True,
    )


def status_row(label: str, status: str):
    color_class = {
        "pending": "dot-amber",
        "running": "dot-amber",
        "done": "dot-green",
        "error": "dot-red",
        "idle": "dot-amber",
        "thinking": "dot-amber",
    }.get(status, "dot-amber")

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin:2px 0;">
          <span class="dot {color_class}"></span>
          <div style="font-weight:800;">{label}</div>
          <span class="wow-badge">{status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8) Load agents.yaml (fallback included)
# ============================================================
def load_agents_cfg() -> Dict[str, Any]:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "agents" not in cfg:
            cfg["agents"] = {}
        return cfg
    except Exception:
        return {"agents": {}}


def ensure_fallback_agents(cfg: Dict[str, Any]) -> Dict[str, Any]:
    agents = cfg.setdefault("agents", {})

    def put(aid: str, obj: Dict[str, Any]):
        if aid not in agents:
            agents[aid] = obj

    # Keep original default agents (existing app behavior)
    put(
        "fda_510k_intel_agent",
        {
            "name": "510(k) Intelligence Agent",
            "model": "gpt-4o-mini",
            "system_prompt": "You are an FDA 510(k) analyst.",
            "max_tokens": 12000,
            "category": "FDA 510(k)",
            "description_tw": "產出 510(k) 情資/摘要與表格。",
        },
    )
    put(
        "pdf_to_markdown_agent",
        {
            "name": "PDF → Markdown Agent",
            "model": "gemini-2.5-flash",
            "system_prompt": "You convert PDF-extracted text into clean markdown.",
            "max_tokens": 12000,
            "category": "Document",
            "description_tw": "將 PDF 文字轉成乾淨 Markdown。",
        },
    )
    put(
        "tw_screen_review_agent",
        {
            "name": "TFDA 預審形式審查代理",
            "model": "gemini-2.5-flash",
            "system_prompt": "You are a TFDA premarket screen reviewer.",
            "max_tokens": 12000,
            "category": "TFDA Premarket",
            "description_tw": "依申請書與指引做形式審查/缺漏分析。",
        },
    )
    put(
        "tw_app_doc_helper",
        {
            "name": "TFDA 申請書撰寫助手",
            "model": "gpt-4o-mini",
            "system_prompt": "You help improve TFDA application documents.",
            "max_tokens": 12000,
            "category": "TFDA Premarket",
            "description_tw": "優化申請書 Markdown 結構與語句。",
        },
    )

    # New: Note Keeper magics agents
    put(
        "note_organizer",
        {
            "name": "Note Organizer",
            "model": "gpt-4o-mini",
            "system_prompt": "You turn messy notes into structured markdown without adding facts.",
            "max_tokens": 12000,
            "category": "Note Keeper",
            "description_tw": "把雜亂筆記整理成有標題/條列的 Markdown。",
        },
    )
    put(
        "keyword_extractor",
        {
            "name": "Keyword Extractor",
            "model": "gemini-2.5-flash",
            "system_prompt": "You extract high-signal keywords/entities from technical notes.",
            "max_tokens": 4000,
            "category": "Note Keeper",
            "description_tw": "從筆記抽取高訊號關鍵字/實體。",
        },
    )
    put(
        "polisher",
        {
            "name": "Polisher",
            "model": "gpt-4.1-mini",
            "system_prompt": "You rewrite text for clarity and professional tone without changing meaning.",
            "max_tokens": 12000,
            "category": "Note Keeper",
            "description_tw": "在不改變原意下潤稿，提升清晰度與專業性。",
        },
    )
    put(
        "critic",
        {
            "name": "Creative Critic",
            "model": "claude-3-5-sonnet-20241022",
            "system_prompt": "You give constructive, specific critique and improvement suggestions.",
            "max_tokens": 12000,
            "category": "Note Keeper",
            "description_tw": "給出具體、可執行的建議與批判性回饋。",
        },
    )
    put(
        "poet_laureate",
        {
            "name": "Poet Laureate",
            "model": "gemini-3-flash-preview",
            "system_prompt": "You transform content into poetic or artistic prose while preserving core ideas.",
            "max_tokens": 12000,
            "category": "Note Keeper",
            "description_tw": "把內容轉為詩/散文式表達（保留核心意思）。",
        },
    )
    put(
        "translator",
        {
            "name": "Translator",
            "model": "gemini-2.5-flash",
            "system_prompt": "You translate accurately with correct terminology.",
            "max_tokens": 12000,
            "category": "Note Keeper",
            "description_tw": "依 UI 語言自動翻譯（中↔英）。",
        },
    )
    return cfg


if "agents_cfg" not in st.session_state:
    st.session_state["agents_cfg"] = ensure_fallback_agents(load_agents_cfg())
else:
    st.session_state["agents_cfg"] = ensure_fallback_agents(st.session_state["agents_cfg"])


# ============================================================
# 9) WOW header + sidebar
# ============================================================
def render_wow_header():
    openai_s, openai_label = api_status("openai")
    gemini_s, gemini_label = api_status("gemini")
    anth_s, anth_label = api_status("anthropic")
    grok_s, grok_label = api_status("grok")

    def dot(s: str) -> str:
        if s == "env":
            return "dot-green"
        if s == "session":
            return "dot-amber"
        return "dot-red"

    st.markdown(
        f"""
        <div class="wow-hero">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px;">
            <div>
              <div class="wow-title">{t("app_title")}</div>
              <div class="wow-subtitle">{t("top_tagline")}</div>
              <div class="wow-chips">
                <div class="wow-chip"><span class="dot {dot(openai_s)}"></span>OpenAI · {openai_label}</div>
                <div class="wow-chip"><span class="dot {dot(gemini_s)}"></span>Gemini · {gemini_label}</div>
                <div class="wow-chip"><span class="dot {dot(anth_s)}"></span>Anthropic · {anth_label}</div>
                <div class="wow-chip"><span class="dot {dot(grok_s)}"></span>xAI Grok · {grok_label}</div>
              </div>
            </div>
            <div style="text-align:right;">
              <div class="wow-badge">{st.session_state.settings["theme"]} · {st.session_state.settings["painter_style"]}</div><br>
              <div style="height:8px;"></div>
              <div class="wow-badge">Default: {st.session_state.settings["model"]} · max_tokens {st.session_state.settings["max_tokens"]}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(f"## {t('global_settings')}")

        # Theme mode
        theme_choice = st.radio(
            t("theme"),
            [t("light"), t("dark")],
            index=0 if st.session_state.settings["theme"] == "Light" else 1,
            horizontal=True,
        )
        st.session_state.settings["theme"] = "Light" if theme_choice == t("light") else "Dark"

        # Language
        lang_choice = st.radio(
            t("language"),
            [t("english"), t("zh_tw")],
            index=0 if st.session_state.settings["language"] == "en" else 1,
            horizontal=True,
        )
        st.session_state.settings["language"] = "en" if lang_choice == t("english") else "zh-tw"

        st.markdown("---")
        st.markdown(f"## {t('style_engine')}")

        c1, c2 = st.columns([4, 1])
        with c1:
            style = st.selectbox(
                t("painter_style"),
                PAINTER_STYLES_20,
                index=PAINTER_STYLES_20.index(st.session_state.settings["painter_style"])
                if st.session_state.settings["painter_style"] in PAINTER_STYLES_20
                else 0,
            )
        with c2:
            if st.button(f"🎰 {t('jackpot')}"):
                style = random.choice(PAINTER_STYLES_20)
                st.session_state.settings["painter_style"] = style
                st.rerun()

        st.session_state.settings["painter_style"] = style

        st.markdown("---")
        st.session_state.settings["model"] = st.selectbox(
            t("default_model"),
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"])
            if st.session_state.settings["model"] in ALL_MODELS
            else 0,
        )
        st.session_state.settings["max_tokens"] = st.number_input(
            t("default_max_tokens"),
            min_value=1000,
            max_value=120000,
            value=int(st.session_state.settings["max_tokens"]),
            step=1000,
        )
        st.session_state.settings["temperature"] = st.slider(
            t("temperature"),
            0.0,
            1.0,
            float(st.session_state.settings["temperature"]),
            0.05,
        )

        st.markdown("---")
        st.markdown(f"## {t('api_keys')}")

        # Show only inputs if env is missing; do not show env values
        keys = dict(st.session_state["api_keys"])

        # OpenAI
        if env_key_present("OPENAI_API_KEY"):
            st.caption(f"OpenAI: {t('active_env')}")
        else:
            keys["openai"] = st.text_input("OpenAI API Key", type="password", value=keys.get("openai", ""))

        # Gemini
        if env_key_present("GEMINI_API_KEY"):
            st.caption(f"Gemini: {t('active_env')}")
        else:
            keys["gemini"] = st.text_input("Gemini API Key", type="password", value=keys.get("gemini", ""))

        # Anthropic
        if env_key_present("ANTHROPIC_API_KEY"):
            st.caption(f"Anthropic: {t('active_env')}")
        else:
            keys["anthropic"] = st.text_input("Anthropic API Key", type="password", value=keys.get("anthropic", ""))

        # Grok
        if env_key_present("GROK_API_KEY"):
            st.caption(f"xAI Grok: {t('active_env')}")
        else:
            keys["grok"] = st.text_input("Grok (xAI) API Key", type="password", value=keys.get("grok", ""))

        st.session_state["api_keys"] = keys

        st.markdown("---")
        st.markdown(f"## {t('agents_catalog')}")
        uploaded_agents = st.file_uploader(t("upload_agents_yaml"), type=["yaml", "yml"])
        if uploaded_agents is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents.read()) or {}
                if "agents" not in cfg:
                    st.warning("Uploaded YAML missing top-level 'agents'. Kept previous config.")
                else:
                    st.session_state["agents_cfg"] = ensure_fallback_agents(cfg)
                    st.success("Loaded agents.yaml for this session.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to parse YAML: {e}")


# ============================================================
# 10) Dashboard (WOW status indicators + interactive charts)
# ============================================================
def render_dashboard():
    # KPI row (always visible)
    hist = st.session_state["history"]
    df = pd.DataFrame(hist) if hist else pd.DataFrame(columns=["tab", "agent", "model", "tokens_est", "ts"])

    total_runs = int(len(df))
    tokens_total = int(df["tokens_est"].sum()) if total_runs else 0
    unique_models = int(df["model"].nunique()) if total_runs else 0

    token_budget = int(st.session_state.settings.get("token_budget_est", 250_000))
    token_ratio = min(1.0, (tokens_total / token_budget) if token_budget else 0.0)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with c1:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        st.markdown(f"**{t('agent_status')}**")
        status_row("Workspace", "idle" if total_runs == 0 else "active")
        st.markdown(f'<div class="wow-kpi">{total_runs}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wow-muted">Total runs</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        st.markdown(f"**{t('token_meter')}**")
        st.progress(token_ratio)
        st.markdown(
            f'<div class="wow-kpi">{tokens_total:,}</div><div class="wow-muted">Estimated tokens this session / budget {token_budget:,}</div></div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        st.markdown("**Models**")
        st.markdown(f'<div class="wow-kpi">{unique_models}</div><div class="wow-muted">Unique models used</div></div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        st.markdown(f"**{t('api_pulse')}**")
        for p, name in [("openai", "OpenAI"), ("gemini", "Gemini"), ("anthropic", "Anthropic"), ("grok", "xAI Grok")]:
            s, label = api_status(p)
            dot_class = "dot-green" if s == "env" else ("dot-amber" if s == "session" else "dot-red")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                f'<span class="dot {dot_class}"></span><b>{name}</b>'
                f'<span class="wow-badge">{label}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### {t('status_wall')}")

    if total_runs == 0:
        st.info("No runs yet. Start by running an agent, a workflow step, or a note magic.")
        return

    df2 = df.copy()
    df2["ts"] = pd.to_datetime(df2["ts"], errors="coerce")

    last = df2.sort_values("ts", ascending=False).iloc[0]
    severity_grad = "linear-gradient(135deg,#22c55e,#16a34a)"
    if int(last["tokens_est"]) > 40000:
        severity_grad = "linear-gradient(135deg,#f97316,#ea580c)"
    if int(last["tokens_est"]) > 80000:
        severity_grad = "linear-gradient(135deg,#ef4444,#b91c1c)"

    st.markdown(
        f"""
        <div class="wow-card" style="background:{severity_grad}; border: 1px solid rgba(255,255,255,0.22);">
          <div style="font-size:0.85rem; opacity:0.92; font-weight:900; letter-spacing:0.12em;">LATEST RUN</div>
          <div style="font-size:1.35rem; font-weight:900; margin-top:6px;">
            {last["tab"]} · {last["agent"]}
          </div>
          <div style="margin-top:6px; opacity:0.94;">
            Model: <b>{last["model"]}</b> · Tokens ≈ <b>{int(last["tokens_est"]):,}</b><br>
            UTC: {str(last["ts"])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Runs by Tab")
        chart_tab = alt.Chart(df2).mark_bar().encode(
            x=alt.X("tab:N", sort="-y"),
            y="count():Q",
            color="tab:N",
            tooltip=["tab", "count()"],
        )
        st.altair_chart(chart_tab, use_container_width=True)

    with cB:
        st.markdown("#### Runs by Model")
        chart_model = alt.Chart(df2).mark_bar().encode(
            x=alt.X("model:N", sort="-y"),
            y="count():Q",
            color="model:N",
            tooltip=["model", "count()"],
        )
        st.altair_chart(chart_model, use_container_width=True)

    st.markdown("#### Token Usage Over Time")
    chart_time = alt.Chart(df2.dropna(subset=["ts"])).mark_line(point=True).encode(
        x="ts:T",
        y="tokens_est:Q",
        color="tab:N",
        tooltip=["ts", "tab", "agent", "model", "tokens_est"],
    )
    st.altair_chart(chart_time, use_container_width=True)

    st.markdown("#### Recent Activity")
    st.dataframe(df2.sort_values("ts", ascending=False).head(40), use_container_width=True)

    cX, cY = st.columns(2)
    with cX:
        if st.button(t("clear_history")):
            st.session_state["history"] = []
            st.rerun()
    with cY:
        csv_bytes = df2.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("export_history"),
            data=csv_bytes,
            file_name="antigravity_history.csv",
            mime="text/csv",
        )


# ============================================================
# 11) Original Agent runner (single agent UI) - upgraded knobs + editable output
# ============================================================
def agent_run_ui(
    agent_id: str,
    tab_key: str,
    default_prompt: str,
    default_input_text: str = "",
    allow_model_override: bool = True,
    tab_label_for_history: Optional[str] = None,
):
    agents_cfg = st.session_state.get("agents_cfg", {})
    agents_dict = agents_cfg.get("agents", {})
    agent_cfg = agents_dict.get(agent_id, {})

    agent_name = agent_cfg.get("name", agent_id)

    # Defaults
    base_model = agent_cfg.get("model", st.session_state.settings["model"])
    base_max_tokens = int(agent_cfg.get("max_tokens", st.session_state.settings["max_tokens"]))
    system_prompt = agent_cfg.get("system_prompt", "")

    # supported_models (optional schema)
    supported = agent_cfg.get("supported_models", None)
    model_choices = ALL_MODELS
    if isinstance(supported, list) and supported:
        model_choices = [m for m in ALL_MODELS if m in supported] or ALL_MODELS

    status_key = f"{tab_key}_status"
    if status_key not in st.session_state:
        st.session_state[status_key] = "idle"

    # WOW status row
    status_row(agent_name, st.session_state[status_key])

    c1, c2, c3 = st.columns([2.2, 1.0, 1.0])
    with c1:
        user_prompt = st.text_area(
            t("prompt"),
            value=st.session_state.get(f"{tab_key}_prompt", default_prompt),
            height=170,
            key=f"{tab_key}_prompt",
        )
    with c2:
        model_index = model_choices.index(base_model) if base_model in model_choices else 0
        model = st.selectbox(
            t("model"),
            model_choices,
            index=model_index,
            disabled=not allow_model_override,
            key=f"{tab_key}_model",
        )
    with c3:
        max_tokens = st.number_input(
            "max_tokens",
            min_value=1000,
            max_value=120000,
            value=int(st.session_state.get(f"{tab_key}_max_tokens", base_max_tokens)),
            step=1000,
            key=f"{tab_key}_max_tokens",
        )

    input_text = st.text_area(
        t("input_text"),
        value=st.session_state.get(f"{tab_key}_input", default_input_text),
        height=240,
        key=f"{tab_key}_input",
    )

    run = st.button(t("run_agent"), key=f"{tab_key}_run")

    if run:
        st.session_state[status_key] = "thinking"
        status_row(agent_name, "thinking")
        api_keys = st.session_state.get("api_keys", {})
        user_full = f"{user_prompt}\n\n---\n\n{input_text}".strip()

        with st.spinner("Running..."):
            try:
                out = call_llm(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_full,
                    max_tokens=int(max_tokens),
                    temperature=float(st.session_state.settings["temperature"]),
                    api_keys=api_keys,
                )
                st.session_state[f"{tab_key}_output"] = out
                st.session_state[status_key] = "done"

                token_est = est_tokens(user_full + out)
                log_event(
                    tab_label_for_history or tab_key,
                    agent_name,
                    model,
                    token_est,
                    meta={"agent_id": agent_id},
                )
            except Exception as e:
                st.session_state[status_key] = "error"
                st.error(f"Agent error: {e}")

    output = st.session_state.get(f"{tab_key}_output", "")
    view_mode = st.radio(
        t("view_mode"),
        [t("markdown"), t("plain_text")],
        horizontal=True,
        key=f"{tab_key}_viewmode",
    )
    edited = st.text_area(
        f"{t('output')} ({'Markdown' if view_mode == t('markdown') else 'Text'}, editable)",
        value=output,
        height=280,
        key=f"{tab_key}_output_edited",
    )
    st.session_state[f"{tab_key}_output_edited_value"] = edited


# ============================================================
# 12) NEW: Agent Workflow Studio (step-by-step, editable handoff)
# ============================================================
def workflow_default_steps() -> List[Dict[str, Any]]:
    """
    A recommended workflow; user can edit each step before running.
    """
    return [
        {
            "agent_id": "pdf_to_markdown_agent",
            "name": "PDF → Markdown Agent",
            "model": st.session_state.settings["model"],
            "max_tokens": st.session_state.settings["max_tokens"],
            "prompt": "Convert the following content into clean structured Markdown. Preserve headings/lists/tables. Do not add facts.",
        },
        {
            "agent_id": "fda_510k_intel_agent",
            "name": "510(k) Intelligence Agent",
            "model": st.session_state.settings["model"],
            "max_tokens": st.session_state.settings["max_tokens"],
            "prompt": "Analyze the provided context and produce a reviewer-oriented report with tables and risks. Keep it grounded in input.",
        },
    ]


def render_workflow_studio():
    st.markdown(f"## {t('workflow_studio')}")
    st.caption("Run agents step-by-step. Edit prompt/model/max_tokens BEFORE each step. Edit output, then pass to next agent.")

    agents_dict = st.session_state["agents_cfg"].get("agents", {})

    # Initialize workflow steps if empty
    wf = st.session_state["workflow"]
    if not wf["steps"]:
        wf["steps"] = workflow_default_steps()
        wf["outputs"] = [""] * len(wf["steps"])
        wf["statuses"] = ["idle"] * len(wf["steps"])
        wf["cursor"] = 0

    # Controls
    c0, c1, c2, c3 = st.columns([1.2, 1.2, 1.2, 1.6])
    with c0:
        if st.button(t("load_defaults")):
            wf["steps"] = workflow_default_steps()
            wf["outputs"] = [""] * len(wf["steps"])
            wf["statuses"] = ["idle"] * len(wf["steps"])
            wf["cursor"] = 0
            wf["input"] = ""
            st.rerun()

    with c1:
        if st.button(t("add_step")):
            wf["steps"].append(
                {
                    "agent_id": "note_organizer",
                    "name": "Note Organizer",
                    "model": st.session_state.settings["model"],
                    "max_tokens": st.session_state.settings["max_tokens"],
                    "prompt": "Organize into structured Markdown. Do not add new facts.",
                }
            )
            wf["outputs"].append("")
            wf["statuses"].append("idle")
            st.rerun()

    with c2:
        if st.button(t("remove_step")):
            if len(wf["steps"]) > 1:
                wf["steps"].pop()
                wf["outputs"].pop()
                wf["statuses"].pop()
                wf["cursor"] = min(wf["cursor"], len(wf["steps"]) - 1)
                st.rerun()

    with c3:
        wf["cursor"] = st.number_input(
            "Active step index",
            min_value=0,
            max_value=max(0, len(wf["steps"]) - 1),
            value=int(wf["cursor"]),
            step=1,
        )

    st.markdown("---")

    # Workflow Input
    st.markdown(f"### {t('workflow_input')}")
    wf["input"] = st.text_area(
        t("input_text"),
        value=wf.get("input", ""),
        height=200,
        key="wf_input_text",
    )

    st.markdown("---")

    # Step list editor (simple, robust)
    st.markdown(f"### {t('step')}s")
    for idx, step in enumerate(wf["steps"]):
        agent_id = step.get("agent_id", "")
        agent_cfg = agents_dict.get(agent_id, {})
        agent_name = step.get("name") or agent_cfg.get("name") or agent_id

        with st.expander(f"{t('step')} {idx+1}: {agent_name}  ·  ({agent_id})", expanded=(idx == wf["cursor"])):
            wf["statuses"][idx] = wf["statuses"][idx] if idx < len(wf["statuses"]) else "idle"
            status_row(f"{agent_name}", wf["statuses"][idx])

            # Resolve supported models if present
            supported = agent_cfg.get("supported_models", None)
            model_choices = ALL_MODELS
            if isinstance(supported, list) and supported:
                model_choices = [m for m in ALL_MODELS if m in supported] or ALL_MODELS

            cA, cB = st.columns([1.2, 1.2])
            with cA:
                step["agent_id"] = st.selectbox(
                    "agent_id",
                    sorted(list(agents_dict.keys())),
                    index=sorted(list(agents_dict.keys())).index(agent_id) if agent_id in agents_dict else 0,
                    key=f"wf_agent_{idx}",
                )
            with cB:
                # update agent cfg after selection
                agent_id = step["agent_id"]
                agent_cfg = agents_dict.get(agent_id, {})
                supported = agent_cfg.get("supported_models", None)
                model_choices = ALL_MODELS
                if isinstance(supported, list) and supported:
                    model_choices = [m for m in ALL_MODELS if m in supported] or ALL_MODELS

                step["model"] = st.selectbox(
                    t("model"),
                    model_choices,
                    index=model_choices.index(step.get("model")) if step.get("model") in model_choices else 0,
                    key=f"wf_model_{idx}",
                )

            cC, cD = st.columns([1.2, 1.2])
            with cC:
                step["max_tokens"] = st.number_input(
                    "max_tokens",
                    min_value=1000,
                    max_value=120000,
                    value=int(step.get("max_tokens", st.session_state.settings["max_tokens"])),
                    step=1000,
                    key=f"wf_mt_{idx}",
                )
            with cD:
                step["name"] = st.text_input(
                    "Display name",
                    value=str(step.get("name") or agent_cfg.get("name") or agent_id),
                    key=f"wf_name_{idx}",
                )

            step["prompt"] = st.text_area(
                t("prompt"),
                value=step.get("prompt", ""),
                height=150,
                key=f"wf_prompt_{idx}",
            )

            # Determine the input to this step:
            if idx == 0:
                step_input_default = wf.get("input", "")
            else:
                # previous output (editable)
                step_input_default = wf["outputs"][idx - 1] or ""

            step_input = st.text_area(
                f"{t('input_text')} (Step {idx+1})",
                value=step_input_default,
                height=180,
                key=f"wf_input_{idx}",
            )

            # Run buttons
            cR1, cR2 = st.columns([1.0, 1.0])
            run_step = cR1.button(f"▶ {t('run_step')} {idx+1}", key=f"wf_run_{idx}")
            run_next = cR2.button(f"⏭ {t('run_next')} {idx+1}", key=f"wf_run_next_{idx}")

            if run_step or run_next:
                wf["cursor"] = idx
                wf["statuses"][idx] = "thinking"

                agent_cfg = agents_dict.get(agent_id, {})
                system_prompt = agent_cfg.get("system_prompt", "")

                user_prompt = (step.get("prompt") or "").strip()
                user_full = (user_prompt + "\n\n---\n\n" + (step_input or "")).strip()

                try:
                    with st.spinner(f"Running step {idx+1}..."):
                        out = call_llm(
                            model=step["model"],
                            system_prompt=system_prompt,
                            user_prompt=user_full,
                            max_tokens=int(step["max_tokens"]),
                            temperature=float(st.session_state.settings["temperature"]),
                            api_keys=st.session_state.get("api_keys", {}),
                        )
                    wf["outputs"][idx] = out
                    wf["statuses"][idx] = "done"

                    log_event(
                        tab="Workflow Studio",
                        agent=step.get("name", agent_id),
                        model=step["model"],
                        tokens_est=est_tokens(user_full + out),
                        meta={"agent_id": agent_id, "workflow_step": idx + 1},
                    )

                    # If "run_next", jump cursor
                    if run_next and idx < len(wf["steps"]) - 1:
                        wf["cursor"] = idx + 1
                        st.rerun()

                except Exception as e:
                    wf["statuses"][idx] = "error"
                    st.error(f"Workflow step error: {e}")

            # Output (editable, pass to next)
            st.markdown(f"**{t('workflow_output')} (editable; becomes input to next step)**")
            view = st.radio(
                t("view_mode"),
                [t("markdown"), t("plain_text")],
                horizontal=True,
                key=f"wf_view_{idx}",
            )
            wf["outputs"][idx] = st.text_area(
                f"Output (Step {idx+1})",
                value=wf["outputs"][idx] or "",
                height=240,
                key=f"wf_out_{idx}",
            )

    st.markdown("---")
    final_out = wf["outputs"][-1] if wf["outputs"] else ""
    if final_out.strip():
        st.download_button(
            t("download_md"),
            data=final_out.encode("utf-8"),
            file_name="workflow_output.md",
            mime="text/markdown",
        )


# ============================================================
# 13) Original tabs (TW Premarket / 510k / PDF->MD / Pipeline)
#     Kept largely as-is, with minor label tweaks.
# ============================================================

LABELS = {
    "Dashboard": {"en": "Dashboard", "zh-tw": "儀表板"},
    "TW Premarket": {"en": "TW Premarket Application", "zh-tw": "第二、三等級醫療器材查驗登記"},
    "510k_tab": {"en": "510(k) Intelligence", "zh-tw": "510(k) 智能分析"},
    "PDF → Markdown": {"en": "PDF → Markdown", "zh-tw": "PDF → Markdown"},
    "Checklist & Report": {"en": "510(k) Review Pipeline", "zh-tw": "510(k) 審查全流程"},
    "Note Keeper & Magics": {"en": "Note Keeper & Magics", "zh-tw": "筆記助手與魔法"},
    "Agents Config": {"en": "Agents Config Studio", "zh-tw": "代理設定工作室"},
    "Workflow Studio": {"en": "Agent Workflow Studio", "zh-tw": "代理工作流工作室"},
}


def tl(key: str) -> str:
    return LABELS.get(key, {}).get(lang_code(), key)


# -----------------------------
# TW schema helpers (kept from original)
# -----------------------------
TW_APP_FIELDS = [
    "doc_no", "e_no", "apply_date", "case_type", "device_category", "case_kind",
    "origin", "product_class", "similar", "replace_flag", "prior_app_no",
    "name_zh", "name_en", "indications", "spec_comp",
    "main_cat", "item_code", "item_name",
    "uniform_id", "firm_name", "firm_addr",
    "resp_name", "contact_name", "contact_tel", "contact_fax", "contact_email",
    "confirm_match", "cert_raps", "cert_ahwp", "cert_other",
    "manu_type", "manu_name", "manu_country", "manu_addr", "manu_note",
    "auth_applicable", "auth_desc",
    "cfs_applicable", "cfs_desc",
    "qms_applicable", "qms_desc",
    "similar_info", "labeling_info", "tech_file_info",
    "preclinical_info", "preclinical_replace",
    "clinical_just", "clinical_info",
]


def build_tw_app_dict_from_session() -> dict:
    s = st.session_state
    apply_date = s.get("tw_apply_date")
    apply_date_str = apply_date.strftime("%Y-%m-%d") if apply_date else ""
    return {
        "doc_no": s.get("tw_doc_no", ""),
        "e_no": s.get("tw_e_no", ""),
        "apply_date": apply_date_str,
        "case_type": s.get("tw_case_type", ""),
        "device_category": s.get("tw_device_category", ""),
        "case_kind": s.get("tw_case_kind", ""),
        "origin": s.get("tw_origin", ""),
        "product_class": s.get("tw_product_class", ""),
        "similar": s.get("tw_similar", ""),
        "replace_flag": s.get("tw_replace_flag", ""),
        "prior_app_no": s.get("tw_prior_app_no", ""),
        "name_zh": s.get("tw_dev_name_zh", ""),
        "name_en": s.get("tw_dev_name_en", ""),
        "indications": s.get("tw_indications", ""),
        "spec_comp": s.get("tw_spec_comp", ""),
        "main_cat": s.get("tw_main_cat", ""),
        "item_code": s.get("tw_item_code", ""),
        "item_name": s.get("tw_item_name", ""),
        "uniform_id": s.get("tw_uniform_id", ""),
        "firm_name": s.get("tw_firm_name", ""),
        "firm_addr": s.get("tw_firm_addr", ""),
        "resp_name": s.get("tw_resp_name", ""),
        "contact_name": s.get("tw_contact_name", ""),
        "contact_tel": s.get("tw_contact_tel", ""),
        "contact_fax": s.get("tw_contact_fax", ""),
        "contact_email": s.get("tw_contact_email", ""),
        "confirm_match": bool(s.get("tw_confirm_match", False)),
        "cert_raps": bool(s.get("tw_cert_raps", False)),
        "cert_ahwp": bool(s.get("tw_cert_ahwp", False)),
        "cert_other": s.get("tw_cert_other", ""),
        "manu_type": s.get("tw_manu_type", ""),
        "manu_name": s.get("tw_manu_name", ""),
        "manu_country": s.get("tw_manu_country", ""),
        "manu_addr": s.get("tw_manu_addr", ""),
        "manu_note": s.get("tw_manu_note", ""),
        "auth_applicable": s.get("tw_auth_app", ""),
        "auth_desc": s.get("tw_auth_desc", ""),
        "cfs_applicable": s.get("tw_cfs_app", ""),
        "cfs_desc": s.get("tw_cfs_desc", ""),
        "qms_applicable": s.get("tw_qms_app", ""),
        "qms_desc": s.get("tw_qms_desc", ""),
        "similar_info": s.get("tw_similar_info", ""),
        "labeling_info": s.get("tw_labeling_info", ""),
        "tech_file_info": s.get("tw_tech_file_info", ""),
        "preclinical_info": s.get("tw_preclinical_info", ""),
        "preclinical_replace": s.get("tw_preclinical_replace", ""),
        "clinical_just": s.get("tw_clinical_app", ""),
        "clinical_info": s.get("tw_clinical_info", ""),
    }


def apply_tw_app_dict_to_session(data: dict):
    s = st.session_state
    s["tw_doc_no"] = data.get("doc_no", "")
    s["tw_e_no"] = data.get("e_no", "")
    from datetime import date
    try:
        if data.get("apply_date"):
            y, m, d = map(int, str(data["apply_date"]).split("-"))
            s["tw_apply_date"] = date(y, m, d)
    except Exception:
        pass
    s["tw_case_type"] = data.get("case_type", "")
    s["tw_device_category"] = data.get("device_category", "")
    s["tw_case_kind"] = data.get("case_kind", "")
    s["tw_origin"] = data.get("origin", "")
    s["tw_product_class"] = data.get("product_class", "")
    s["tw_similar"] = data.get("similar", "")
    s["tw_replace_flag"] = data.get("replace_flag", "")
    s["tw_prior_app_no"] = data.get("prior_app_no", "")
    s["tw_dev_name_zh"] = data.get("name_zh", "")
    s["tw_dev_name_en"] = data.get("name_en", "")
    s["tw_indications"] = data.get("indications", "")
    s["tw_spec_comp"] = data.get("spec_comp", "")
    s["tw_main_cat"] = data.get("main_cat", "")
    s["tw_item_code"] = data.get("item_code", "")
    s["tw_item_name"] = data.get("item_name", "")
    s["tw_uniform_id"] = data.get("uniform_id", "")
    s["tw_firm_name"] = data.get("firm_name", "")
    s["tw_firm_addr"] = data.get("firm_addr", "")
    s["tw_resp_name"] = data.get("resp_name", "")
    s["tw_contact_name"] = data.get("contact_name", "")
    s["tw_contact_tel"] = data.get("contact_tel", "")
    s["tw_contact_fax"] = data.get("contact_fax", "")
    s["tw_contact_email"] = data.get("contact_email", "")
    s["tw_confirm_match"] = bool(data.get("confirm_match", False))
    s["tw_cert_raps"] = bool(data.get("cert_raps", False))
    s["tw_cert_ahwp"] = bool(data.get("cert_ahwp", False))
    s["tw_cert_other"] = data.get("cert_other", "")
    s["tw_manu_type"] = data.get("manu_type", "")
    s["tw_manu_name"] = data.get("manu_name", "")
    s["tw_manu_country"] = data.get("manu_country", "")
    s["tw_manu_addr"] = data.get("manu_addr", "")
    s["tw_manu_note"] = data.get("manu_note", "")
    s["tw_auth_app"] = data.get("auth_applicable", "")
    s["tw_auth_desc"] = data.get("auth_desc", "")
    s["tw_cfs_app"] = data.get("cfs_applicable", "")
    s["tw_cfs_desc"] = data.get("cfs_desc", "")
    s["tw_qms_app"] = data.get("qms_applicable", "")
    s["tw_qms_desc"] = data.get("qms_desc", "")
    s["tw_similar_info"] = data.get("similar_info", "")
    s["tw_labeling_info"] = data.get("labeling_info", "")
    s["tw_tech_file_info"] = data.get("tech_file_info", "")
    s["tw_preclinical_info"] = data.get("preclinical_info", "")
    s["tw_preclinical_replace"] = data.get("preclinical_replace", "")
    s["tw_clinical_app"] = data.get("clinical_just", "")
    s["tw_clinical_info"] = data.get("clinical_info", "")


def standardize_tw_app_info_with_llm(raw_obj) -> dict:
    api_keys = st.session_state.get("api_keys", {})
    model = "gemini-2.5-flash"
    if not get_api_key("gemini", api_keys):
        raise RuntimeError("No Gemini API key available for standardizing application info.")

    raw_json = json.dumps(raw_obj, ensure_ascii=False, indent=2)
    fields_str = ", ".join(TW_APP_FIELDS)

    system_prompt = f"""
You are a data normalization assistant for a Taiwanese TFDA medical device premarket application.

Map arbitrary JSON/CSV key/value data into a STANDARD JSON object with EXACT keys:
{fields_str}

Rules:
- Output MUST be a single JSON object (no markdown, no comments).
- Every key above MUST appear.
- If missing, use "" or false for boolean-like fields.
- Do NOT invent facts.
- apply_date format: YYYY-MM-DD if inferable, else "".
"""

    out = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=f"Raw data:\n{raw_json}",
        max_tokens=4000,
        temperature=0.1,
        api_keys=api_keys,
    )

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(out[start : end + 1])
        else:
            raise RuntimeError("LLM did not return valid JSON.")
    if not isinstance(data, dict):
        raise RuntimeError("Standardized output is not an object.")
    for k in TW_APP_FIELDS:
        if k not in data:
            data[k] = False if k in ("confirm_match", "cert_raps", "cert_ahwp") else ""
    return data


def compute_tw_app_completeness() -> float:
    s = st.session_state
    required_keys = [
        "tw_e_no", "tw_case_type", "tw_device_category",
        "tw_origin", "tw_product_class",
        "tw_dev_name_zh", "tw_dev_name_en",
        "tw_uniform_id", "tw_firm_name", "tw_firm_addr",
        "tw_resp_name", "tw_contact_name", "tw_contact_tel",
        "tw_contact_email",
        "tw_manu_name", "tw_manu_addr",
    ]
    filled = 0
    for k in required_keys:
        v = s.get(k, "")
        if isinstance(v, str):
            filled += 1 if v.strip() else 0
        else:
            filled += 1 if v else 0
    return filled / len(required_keys) if required_keys else 0.0


def render_tw_premarket_tab():
    st.markdown(f"## {tl('TW Premarket')}")

    st.markdown(
        """
        <div class="wow-card">
          <div style="font-weight:900; font-size:1.05rem;">Quick Guide</div>
          <div class="wow-muted" style="margin-top:6px;">
            Step 1: Fill/import application fields → Step 2: Provide screening guidance → Step 3: Run screening agent → Step 4: Improve doc with helper agent.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Application Info 匯入 / 匯出 (JSON / CSV)")
    col_ie1, col_ie2 = st.columns(2)
    with col_ie1:
        app_file = st.file_uploader("Upload Application Info (JSON / CSV)", type=["json", "csv"], key="tw_app_upload")
        if app_file is not None:
            try:
                if app_file.name.lower().endswith(".json"):
                    raw_data = json.load(app_file)
                else:
                    df = pd.read_csv(app_file)
                    raw_data = df.to_dict(orient="records")[0] if len(df) else None

                if raw_data is not None:
                    if isinstance(raw_data, dict) and all(k in raw_data for k in TW_APP_FIELDS):
                        standardized = raw_data
                    else:
                        with st.spinner("使用 LLM 將欄位轉為標準 TFDA 申請書格式..."):
                            standardized = standardize_tw_app_info_with_llm(raw_data)
                    apply_tw_app_dict_to_session(standardized)
                    st.session_state["tw_app_last_loaded"] = standardized
                    st.success("已套用至申請表單。")
                    st.rerun()
            except Exception as e:
                st.error(f"上傳或標準化失敗：{e}")

    with col_ie2:
        app_dict = build_tw_app_dict_from_session()
        st.download_button(
            "Download JSON",
            data=json.dumps(app_dict, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="tw_premarket_application.json",
            mime="application/json",
        )
        st.download_button(
            "Download CSV",
            data=pd.DataFrame([app_dict]).to_csv(index=False).encode("utf-8"),
            file_name="tw_premarket_application.csv",
            mime="text/csv",
        )

    if "tw_app_last_loaded" in st.session_state:
        st.markdown("**最近載入/標準化之 Application JSON 預覽**")
        st.json(st.session_state["tw_app_last_loaded"], expanded=False)

    st.markdown("---")

    completeness = compute_tw_app_completeness()
    pct = int(completeness * 100)
    if pct >= 80:
        card_grad = "linear-gradient(135deg,#22c55e,#16a34a)"
        txt = "申請基本欄位完成度高，適合進行預審。"
    elif pct >= 50:
        card_grad = "linear-gradient(135deg,#f97316,#ea580c)"
        txt = "部分關鍵欄位仍待補齊，建議補足後再送預審。"
    else:
        card_grad = "linear-gradient(135deg,#ef4444,#b91c1c)"
        txt = "多數基本欄位尚未填寫，請先充實申請資訊。"

    st.markdown(
        f"""
        <div class="wow-card" style="background:{card_grad}; border: 1px solid rgba(255,255,255,0.22);">
          <div style="font-weight:900; letter-spacing:0.12em; opacity:0.95;">APPLICATION COMPLETENESS</div>
          <div style="font-size:1.6rem; font-weight:900; margin-top:6px;">{pct}%</div>
          <div style="margin-top:6px; opacity:0.95;">{txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(completeness)

    st.markdown("### Step 1 – 線上填寫申請書（草稿）")
    # (Original form kept; same as provided, trimmed only minimally for length safety)

    if "tw_app_status" not in st.session_state:
        st.session_state["tw_app_status"] = "pending"
    status_row("申請書填寫", st.session_state["tw_app_status"])

    # --- Basic info
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        doc_no = st.text_input("公文文號", key="tw_doc_no")
        e_no = st.text_input("電子流水號", value=st.session_state.get("tw_e_no", "MDE"), key="tw_e_no")
    with col_a2:
        apply_date = st.date_input("申請日", key="tw_apply_date")
        case_type = st.selectbox(
            "案件類型*",
            ["一般申請案", "同一產品不同品名", "專供外銷", "許可證有效期限屆至後六個月內重新申請"],
            key="tw_case_type",
        )
    with col_a3:
        device_category = st.selectbox("醫療器材類型*", ["一般醫材", "體外診斷器材(IVD)"], key="tw_device_category")
        case_kind = st.selectbox("案件種類*", ["新案", "變更案", "展延案"], index=0, key="tw_case_kind")

    col_a4, col_a5, col_a6 = st.columns(3)
    with col_a4:
        origin = st.selectbox("產地*", ["國產", "輸入", "陸輸"], key="tw_origin")
    with col_a5:
        product_class = st.selectbox("產品等級*", ["第二等級", "第三等級"], key="tw_product_class")
    with col_a6:
        similar = st.selectbox("有無類似品*", ["有", "無", "全球首創"], key="tw_similar")

    col_a7, col_a8 = st.columns(2)
    with col_a7:
        replace_flag = st.radio(
            "是否勾選「替代臨床前測試及原廠品質管制資料」？*",
            ["否", "是"],
            index=0 if st.session_state.get("tw_replace_flag", "否") == "否" else 1,
            key="tw_replace_flag",
        )
    with col_a8:
        prior_app_no = st.text_input("（非首次申請）前次申請案號", key="tw_prior_app_no")

    # --- Device info
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        name_zh = st.text_input("醫療器材中文名稱*", key="tw_dev_name_zh")
        name_en = st.text_input("醫療器材英文名稱*", key="tw_dev_name_en")
    with col_b2:
        indications = st.text_area("效能、用途或適應症說明", value=st.session_state.get("tw_indications", "詳如核定之中文說明書"), key="tw_indications")
        spec_comp = st.text_area("型號、規格或主要成分說明", value=st.session_state.get("tw_spec_comp", "詳如核定之中文說明書"), key="tw_spec_comp")

    col_b3, col_b4, col_b5 = st.columns(3)
    with col_b3:
        main_cat = st.selectbox("主類別", ["", "A.臨床化學及臨床毒理學", "B.血液學及病理學", "C.免疫學及微生物學",
                                      "D.麻醉學", "E.心臟血管醫學", "F.牙科學", "G.耳鼻喉科學", "H.胃腸病科學及泌尿科學",
                                      "I.一般及整形外科手術", "J.一般醫院及個人使用裝置", "K.神經科學", "L.婦產科學", "M.眼科學",
                                      "N.骨科學", "O.物理醫學科學", "P.放射學科學"], key="tw_main_cat")
    with col_b4:
        item_code = st.text_input("分級品項代碼（例：A.1225）", key="tw_item_code")
    with col_b5:
        item_name = st.text_input("分級品項名稱（例：肌氨酸酐試驗系統）", key="tw_item_name")

    # --- Firm info
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        uniform_id = st.text_input("統一編號*", key="tw_uniform_id")
        firm_name = st.text_input("醫療器材商名稱*", key="tw_firm_name")
        firm_addr = st.text_area("醫療器材商地址*", height=80, key="tw_firm_addr")
    with col_c2:
        resp_name = st.text_input("負責人姓名*", key="tw_resp_name")
        contact_name = st.text_input("聯絡人姓名*", key="tw_contact_name")
        contact_tel = st.text_input("電話*", key="tw_contact_tel")
        contact_fax = st.text_input("聯絡人傳真", key="tw_contact_fax")
        contact_email = st.text_input("電子郵件*", key="tw_contact_email")

    confirm_match = st.checkbox(
        "我已確認上述資料與最新版醫療器材商證照資訊(名稱、地址、負責人)相符",
        key="tw_confirm_match",
    )

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        cert_raps = st.checkbox("RAPS", key="tw_cert_raps")
        cert_ahwp = st.checkbox("AHWP", key="tw_cert_ahwp")
    with col_c4:
        cert_other = st.text_input("其它，請敘明", key="tw_cert_other")

    # --- Manufacturer
    manu_type = st.radio(
        "製造方式",
        ["單一製造廠", "全部製程委託製造", "委託非全部製程之製造/包裝/貼標/滅菌及最終驗放"],
        index=0,
        key="tw_manu_type",
    )
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        manu_name = st.text_input("製造廠名稱*", key="tw_manu_name")
        manu_country = st.selectbox(
            "製造國別*",
            ["TAIWAN， ROC", "UNITED STATES", "EU (Member State)", "JAPAN", "CHINA", "KOREA， REPUBLIC OF", "OTHER"],
            key="tw_manu_country",
        )
    with col_d2:
        manu_addr = st.text_area("製造廠地址*", height=80, key="tw_manu_addr")
        manu_note = st.text_area("製造廠相關說明", height=80, key="tw_manu_note")

    with st.expander("附件摘要（可選填）", expanded=False):
        auth_applicable = st.selectbox("原廠授權登記書", ["不適用", "適用"], key="tw_auth_app")
        auth_desc = st.text_area("原廠授權登記書資料說明", height=80, key="tw_auth_desc")

        cfs_applicable = st.selectbox("出產國製售證明", ["不適用", "適用"], key="tw_cfs_app")
        cfs_desc = st.text_area("出產國製售證明資料說明", height=80, key="tw_cfs_desc")

        qms_applicable = st.selectbox("QMS/QSD", ["不適用", "適用"], key="tw_qms_app")
        qms_desc = st.text_area("QMS/QSD 資料說明", height=80, key="tw_qms_desc")

        similar_info = st.text_area("類似品與比較表摘要", height=80, key="tw_similar_info")
        labeling_info = st.text_area("標籤、說明書或包裝擬稿重點", height=100, key="tw_labeling_info")
        tech_file_info = st.text_area("技術檔案摘要", height=120, key="tw_tech_file_info")
        preclinical_info = st.text_area("臨床前測試與品質管制摘要", height=140, key="tw_preclinical_info")
        preclinical_replace = st.text_area("替代臨床前測試資料之說明", height=100, key="tw_preclinical_replace")
        clinical_just = st.selectbox("臨床證據是否適用？", ["不適用", "適用"], key="tw_clinical_app")
        clinical_info = st.text_area("臨床證據摘要", height=140, key="tw_clinical_info")

    if st.button("生成申請書 Markdown 草稿", key="tw_generate_md_btn"):
        missing = []
        for label, val in [
            ("電子流水號", e_no),
            ("案件類型", case_type),
            ("醫療器材類型", device_category),
            ("產地", origin),
            ("產品等級", product_class),
            ("中文名稱", name_zh),
            ("英文名稱", name_en),
            ("統一編號", uniform_id),
            ("醫療器材商名稱", firm_name),
            ("醫療器材商地址", firm_addr),
            ("負責人姓名", resp_name),
            ("聯絡人姓名", contact_name),
            ("電話", contact_tel),
            ("電子郵件", contact_email),
            ("製造廠名稱", manu_name),
            ("製造廠地址", manu_addr),
        ]:
            if isinstance(val, str) and not val.strip():
                missing.append(label)

        st.session_state["tw_app_status"] = "error" if missing else "done"
        if missing:
            st.warning("以下基本欄位尚未完整（形式檢查）：\n- " + "\n- ".join(missing))

        apply_date_str = apply_date.strftime("%Y-%m-%d") if apply_date else ""
        app_md = f"""# 第二、三等級醫療器材查驗登記申請書（線上草稿）

## 一、案件基本資料
- 公文文號：{doc_no or "（未填）"}
- 電子流水號：{e_no or "（未填）"}
- 申請日：{apply_date_str or "（未填）"}
- 案件類型：{case_type}
- 醫療器材類型：{device_category}
- 案件種類：{case_kind}
- 產地：{origin}
- 產品等級：{product_class}
- 有無類似品：{similar}
- 替代臨床前測試及品質管制資料：{replace_flag}
- 前次申請案號：{prior_app_no or "不適用"}

## 二、醫療器材基本資訊
- 中文名稱：{name_zh}
- 英文名稱：{name_en}
- 效能/用途/適應症：{indications}
- 型號/規格/主要成分：{spec_comp}

### 分類分級品項
- 主類別：{main_cat or "（未填）"}
- 代碼：{item_code or "（未填）"}
- 名稱：{item_name or "（未填）"}

## 三、醫療器材商資料
- 統一編號：{uniform_id}
- 名稱：{firm_name}
- 地址：{firm_addr}
- 負責人：{resp_name}
- 聯絡人：{contact_name}
- 電話：{contact_tel}
- 傳真：{contact_fax or "（未填）"}
- 電子郵件：{contact_email}
- 已確認證照相符：{"是" if confirm_match else "否"}

## 四、製造廠資訊
- 製造方式：{manu_type}
- 製造廠名稱：{manu_name}
- 製造國別：{manu_country}
- 製造廠地址：{manu_addr}
- 製造說明：{manu_note or "（未填）"}

## 附件摘要（如適用）
- 原廠授權：{auth_applicable} / {auth_desc or "（未填）"}
- 製售證明：{cfs_applicable} / {cfs_desc or "（未填）"}
- QMS/QSD：{qms_applicable} / {qms_desc or "（未填）"}

### 類似品摘要
{similar_info or "（未填）"}

### 標籤/說明書擬稿重點
{labeling_info or "（未填）"}

### 技術檔案摘要
{tech_file_info or "（未填）"}

### 臨床前測試與品質管制摘要
{preclinical_info or "（未填）"}

### 替代資料說明
{preclinical_replace or "（未填）"}

### 臨床證據
- 適用性：{clinical_just}
- 摘要：{clinical_info or "（未填）"}
"""
        st.session_state["tw_app_markdown"] = app_md

    st.markdown("##### 申請書 Markdown（可編輯）")
    app_md_current = st.session_state.get("tw_app_markdown", "")
    view = st.radio("申請書檢視模式", ["Markdown", "純文字"], horizontal=True, key="tw_app_viewmode")
    app_md_edited = st.text_area("申請書內容", value=app_md_current, height=280, key="tw_app_md_edited")
    st.session_state["tw_app_effective_md"] = app_md_edited

    st.markdown("---")
    st.markdown("### Step 2 – 輸入預審/形式審查指引")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        guidance_file = st.file_uploader("上傳預審指引 (PDF / TXT / MD)", type=["pdf", "txt", "md"], key="tw_guidance_file")
        guidance_text_from_file = ""
        if guidance_file is not None:
            suffix = guidance_file.name.lower().rsplit(".", 1)[-1]
            if suffix == "pdf":
                guidance_text_from_file = extract_pdf_pages_to_text(guidance_file, 1, 9999)
            else:
                guidance_text_from_file = guidance_file.read().decode("utf-8", errors="ignore")
    with col_g2:
        guidance_text_manual = st.text_area("或直接貼上指引文字/Markdown", height=200, key="tw_guidance_manual")

    guidance_text = guidance_text_from_file or guidance_text_manual
    st.session_state["tw_guidance_text"] = guidance_text

    st.markdown("---")
    st.markdown("### Step 3 – 形式審查 / 完整性檢核（Agent）")
    base_app_md = st.session_state.get("tw_app_effective_md", "")
    base_guidance = st.session_state.get("tw_guidance_text", "")
    combined_input = f"""=== 申請書草稿（Markdown） ===
{base_app_md}

=== 預審 / 形式審查指引 ===
{base_guidance or "（尚未提供指引，請依一般法規常規進行形式檢核）"}
"""
    default_screen_prompt = """你是一位熟悉臺灣「第二、三等級醫療器材查驗登記」的形式審查(預審)審查員。

請根據：
1) 申請書草稿（Markdown）
2) 預審/形式審查指引（如有）

以繁體中文 Markdown 輸出：
- 形式完整性檢核表（含：預期應附、是否提及、判定、備註）
- 重要欄位檢核（問題項目/疑慮/建議補充）
- 預審評語摘要（300–600字）
- 無從判斷請明確註記
"""
    agent_run_ui(
        agent_id="tw_screen_review_agent",
        tab_key="tw_screen",
        default_prompt=default_screen_prompt,
        default_input_text=combined_input,
        allow_model_override=True,
        tab_label_for_history="TW Premarket Screen Review",
    )

    st.markdown("---")
    st.markdown("### Step 4 – AI 協助編修申請書內容")
    helper_default_prompt = """你是一位協助臺灣醫療器材查驗登記申請人的文件撰寫助手。

請在不改變實際技術/法規內容的前提下：
- 優化段落結構與標題層級
- 修正文句、提升可讀性
- 資訊不足處以「※待補：...」標註
- 輸出 Markdown
"""
    agent_run_ui(
        agent_id="tw_app_doc_helper",
        tab_key="tw_app_helper",
        default_prompt=helper_default_prompt,
        default_input_text=base_app_md,
        allow_model_override=True,
        tab_label_for_history="TW Application Doc Helper",
    )


def render_510k_tab():
    st.markdown(f"## {tl('510k_tab')}")
    col1, col2 = st.columns(2)
    with col1:
        device_name = st.text_input("Device Name")
        k_number = st.text_input("510(k) Number (e.g., K123456)")
    with col2:
        sponsor = st.text_input("Sponsor / Manufacturer (optional)")
        product_code = st.text_input("Product Code (optional)")
    extra_info = st.text_area("Additional context (indications, technology, etc.)")

    default_prompt = f"""
You are assisting an FDA 510(k) reviewer.

Task:
1) Summarize information for:
   - Device: {device_name}
   - 510(k) number: {k_number}
   - Sponsor: {sponsor}
   - Product code: {product_code}
2) Produce a detailed review-oriented summary (2000–3000 words).
3) Provide markdown tables (overview, indications, testing, risks).

Language: {lang_code()}.
"""
    combined_input = f"""
Device name: {device_name}
510(k) number: {k_number}
Sponsor: {sponsor}
Product code: {product_code}

Additional context:
{extra_info}
"""
    agent_run_ui(
        agent_id="fda_510k_intel_agent",
        tab_key="510k",
        default_prompt=default_prompt,
        default_input_text=combined_input,
        tab_label_for_history="510(k) Intelligence",
    )


def render_pdf_to_md_tab():
    st.markdown(f"## {tl('PDF → Markdown')}")
    uploaded = st.file_uploader("Upload PDF to convert selected pages to Markdown", type=["pdf"], key="pdf_to_md_uploader")
    if uploaded:
        c1, c2 = st.columns(2)
        with c1:
            num_start = st.number_input("From page", min_value=1, value=1, key="pdf_to_md_from")
        with c2:
            num_end = st.number_input("To page", min_value=1, value=5, key="pdf_to_md_to")

        if st.button("Extract Text", key="pdf_to_md_extract_btn"):
            text = extract_pdf_pages_to_text(uploaded, int(num_start), int(num_end))
            st.session_state["pdf_raw_text"] = text

    raw_text = st.session_state.get("pdf_raw_text", "")
    if raw_text:
        default_prompt = f"""
You are converting part of a regulatory PDF into markdown.

- Produce clean, structured markdown preserving headings, lists, tables (as markdown tables if possible).
- Do not hallucinate content.

Language: {lang_code()}.
"""
        agent_run_ui(
            agent_id="pdf_to_markdown_agent",
            tab_key="pdf_to_md",
            default_prompt=default_prompt,
            default_input_text=raw_text,
            tab_label_for_history="PDF → Markdown",
        )
    else:
        st.info("Upload a PDF and click 'Extract Text' to begin.")


def render_510k_review_pipeline_tab():
    st.markdown(f"## {tl('Checklist & Report')}")
    st.markdown("### Step 1 – 提交資料 → 結構化 Markdown")

    raw_subm = st.text_area("Paste 510(k) submission material (text/markdown)", height=200, key="subm_paste")
    default_subm_prompt = """You are a 510(k) submission organizer.

Restructure the following content into organized markdown with sections such as:
- Device & submitter information
- Device description and technology
- Indications for use
- Predicate/comparator information
- Performance testing
- Risks and risk controls

Do not invent facts; reorganize and clarify only.
"""
    if st.button("Structure Submission", key="subm_run_btn"):
        if not raw_subm.strip():
            st.warning("Please paste submission material first.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            try:
                out = call_llm(
                    model=st.session_state.settings["model"],
                    system_prompt="You structure a 510(k) submission.",
                    user_prompt=default_subm_prompt + "\n\n=== SUBMISSION ===\n" + raw_subm,
                    max_tokens=st.session_state.settings["max_tokens"],
                    temperature=0.15,
                    api_keys=api_keys,
                )
                st.session_state["subm_struct_md"] = out
                log_event("510(k) Review Pipeline", "Submission Structurer", st.session_state.settings["model"], est_tokens(raw_subm + out))
            except Exception as e:
                st.error(f"Error: {e}")

    subm_md = st.session_state.get("subm_struct_md", "")
    if subm_md:
        st.text_area("Structured Submission (Markdown, editable)", value=subm_md, height=220, key="subm_struct_md_edited")
    else:
        st.info("Structured submission will appear here.")

    st.markdown("---")
    st.markdown("### Step 2 – Checklist & Step 3 – Review Report")

    chk_md = st.text_area("Paste checklist (markdown or text)", height=200, key="chk_md")
    if st.button("Build Review Report", key="rep_run_btn"):
        subm_md_eff = st.session_state.get("subm_struct_md_edited", subm_md)
        if not subm_md_eff.strip() or not chk_md.strip():
            st.warning("Need both structured submission and checklist.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            rep_prompt = """You are drafting an internal FDA 510(k) review memo.

Using the checklist and structured submission, write a concise review report with:
- Introduction & scope
- Device and submission overview
- Key differences vs predicate(s)
- Checklist-based assessment (headings/tables)
- Conclusion and recommendations
"""
            user_prompt = rep_prompt + "\n\n=== CHECKLIST ===\n" + chk_md + "\n\n=== STRUCTURED SUBMISSION ===\n" + subm_md_eff
            try:
                out = call_llm(
                    model=st.session_state.settings["model"],
                    system_prompt="You are an FDA 510(k) reviewer.",
                    user_prompt=user_prompt,
                    max_tokens=st.session_state.settings["max_tokens"],
                    temperature=0.18,
                    api_keys=api_keys,
                )
                st.session_state["rep_md"] = out
                log_event("510(k) Review Pipeline", "Review Memo Builder", st.session_state.settings["model"], est_tokens(user_prompt + out))
            except Exception as e:
                st.error(f"Error: {e}")

    rep_md = st.session_state.get("rep_md", "")
    if rep_md:
        st.text_area("Review Report (Markdown, editable)", value=rep_md, height=260, key="rep_md_edited")


# ============================================================
# 14) Note Keeper (paste + upload + organize + coral keywords + 6 magics)
# ============================================================
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (s or "").strip())


def read_uploaded_note_file() -> str:
    file = st.session_state.get("note_file")
    if not file:
        return ""
    name = file.name.lower()
    if name.endswith(".pdf"):
        # simple extraction; OCR not included
        return extract_pdf_pages_to_text(file, 1, 9999)
    return file.read().decode("utf-8", errors="ignore")


def highlight_keywords_html(text: str, keywords: List[str], color: str = "#FF7F50") -> str:
    """
    Safer-ish highlighter: word-boundary when possible; preserves markdown-ish text by using HTML spans.
    """
    if not text.strip() or not keywords:
        return text

    # Avoid overlapping replacements by sorting longest first
    kws = sorted({k.strip() for k in keywords if k and k.strip()}, key=len, reverse=True)
    out = text

    for kw in kws:
        # simple heuristic: if keyword has letters/numbers, use boundary; else direct
        if re.search(r"[A-Za-z0-9]", kw):
            pattern = re.compile(rf"(?<![\w-]){re.escape(kw)}(?![\w-])")
            out = pattern.sub(rf'<span style="color:{color};font-weight:800;">{kw}</span>', out)
        else:
            out = out.replace(kw, f'<span style="color:{color};font-weight:800;">{kw}</span>')
    return out


def magic_ai_keywords(base_md: str, color: str, model: str) -> Tuple[List[str], str]:
    """
    Returns (keywords, highlighted_html_markdown)
    """
    api_keys = st.session_state.get("api_keys", {})
    system_prompt = st.session_state["agents_cfg"]["agents"]["keyword_extractor"].get("system_prompt", "")
    user_prompt = f"""
Extract the TOP 10-15 high-signal keywords/entities from the note below.

Rules:
- Prefer: proper nouns, standards (ISO/IEC), guidance names, device names, test names, endpoints, regulatory terms, dates, key metrics.
- Output MUST be JSON only: {{"keywords":["..."]}}

NOTE:
{base_md}
"""
    raw = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1500,
        temperature=0.1,
        api_keys=api_keys,
    )

    # Parse JSON
    try:
        obj = json.loads(raw)
    except Exception:
        s = raw[raw.find("{") : raw.rfind("}") + 1]
        obj = json.loads(s)

    keywords = obj.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    highlighted = highlight_keywords_html(base_md, keywords, color=color)
    return keywords, highlighted


def render_note_keeper_tab():
    st.markdown(f"## {tl('Note Keeper & Magics')}")
    st.caption("Paste or upload notes. Convert into organized Markdown with coral-highlighted keywords, then apply 6 AI Magics.")

    c0, c1 = st.columns([1.2, 1.8])
    with c0:
        st.file_uploader(t("note_upload"), type=["pdf", "txt", "md"], key="note_file")
        if st.button("Load uploaded note into editor"):
            file_text = read_uploaded_note_file()
            st.session_state["notes_raw"] = normalize_whitespace(file_text)
            st.rerun()
    with c1:
        st.text_area(t("note_paste"), height=220, key="notes_raw")

    base_raw = normalize_whitespace(st.session_state.get("notes_raw", ""))

    st.markdown("---")
    st.markdown(f"### {t('note_transform')}")

    cA, cB, cC = st.columns([1.2, 1.2, 1.2])
    with cA:
        note_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]), key="note_model")
    with cB:
        note_max_tokens = st.number_input("max_tokens", 2000, 120000, 12000, 1000, key="note_max_tokens")
    with cC:
        kw_color = st.color_picker(t("note_color"), "#FF7F50", key="note_kw_color")

    organizer_prompt = st.text_area(
        t("prompt"),
        value="""You are an expert note organizer.

Transform the RAW NOTE into clean, organized Markdown with:
- Clear headings and subheadings
- Bullet points where appropriate
- A short "Key Takeaways" section at the top
- A "Questions / Follow-ups" section at the bottom

Hard rules:
- Do not add facts that are not present in the note.
- Keep technical and regulatory terminology accurate.
""",
        height=160,
        key="note_organizer_prompt",
    )

    if st.button(t("run"), key="note_transform_btn"):
        if not base_raw.strip():
            st.warning("No note content.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            agent_cfg = st.session_state["agents_cfg"]["agents"]["note_organizer"]
            system_prompt = agent_cfg.get("system_prompt", "")
            user_prompt = organizer_prompt + "\n\n=== RAW NOTE ===\n" + base_raw

            try:
                with st.spinner("Organizing note..."):
                    out = call_llm(
                        model=note_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=int(note_max_tokens),
                        temperature=0.15,
                        api_keys=api_keys,
                    )
                st.session_state["note_md"] = out
                log_event("Note Keeper", "Note Organizer", note_model, est_tokens(user_prompt + out))
            except Exception as e:
                st.error(f"Error: {e}")

    base_md = normalize_whitespace(st.session_state.get("note_md", base_raw))

    st.markdown("#### Base Note (editable)")
    view = st.radio(t("view_mode"), [t("markdown"), t("plain_text")], horizontal=True, key="note_view_mode")
    base_md = st.text_area("Note", value=base_md, height=260, key="note_md_edited")
    st.session_state["note_effective"] = base_md

    st.markdown("---")
    st.markdown(f"### {t('ai_magics')} (6)")

    base_note = st.session_state.get("note_effective", "")

    # Magic 1: AI Keywords (auto extract) + user editable keyword list + color
    st.markdown(f"#### 1) {t('magic_keywords')}")
    cK1, cK2, cK3 = st.columns([1.1, 1.1, 1.2])
    with cK1:
        kw_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0, key="kw_model")
    with cK2:
        kw_color2 = st.color_picker(t("note_color"), kw_color, key="kw_color2")
    with cK3:
        manual_kw = st.text_input("Manual keywords (comma-separated; optional)", key="manual_kw", value="")

    cK4, cK5 = st.columns([1.0, 1.0])
    with cK4:
        if st.button("Run AI Keywords + Highlight", key="kw_ai_btn"):
            if not base_note.strip():
                st.warning("No base note.")
            else:
                try:
                    kws, highlighted = magic_ai_keywords(base_note, kw_color2, kw_model)
                    st.session_state["kw_ai_list"] = kws
                    st.session_state["kw_highlighted"] = highlighted
                    log_event("Note Keeper", "AI Keywords", kw_model, est_tokens(base_note))
                except Exception as e:
                    st.error(f"AI Keywords failed: {e}")

    with cK5:
        if st.button("Apply Manual Highlight", key="kw_manual_btn"):
            kws = [k.strip() for k in manual_kw.split(",") if k.strip()]
            st.session_state["kw_ai_list"] = kws
            st.session_state["kw_highlighted"] = highlight_keywords_html(base_note, kws, color=kw_color2)

    highlighted = st.session_state.get("kw_highlighted", "")
    if highlighted.strip():
        st.markdown("**Rendered (HTML-capable Markdown):**")
        st.markdown(highlighted, unsafe_allow_html=True)
        st.text_area("Highlighted (editable source)", value=highlighted, height=200, key="kw_highlighted_edit")

    # Magic 2: Summarize
    st.markdown(f"#### 2) {t('magic_summarize')}")
    sum_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0, key="sum_model")
    sum_prompt = st.text_area(
        t("prompt"),
        value="Summarize into 8–12 bullets + a 5-sentence executive summary. Keep terminology accurate. Output Markdown.",
        height=120,
        key="sum_prompt",
    )
    if st.button(t("run"), key="sum_btn"):
        try:
            out = call_llm(
                model=sum_model,
                system_prompt="You write executive summaries for technical/regulatory notes.",
                user_prompt=sum_prompt + "\n\n=== NOTE ===\n" + base_note,
                max_tokens=12000,
                temperature=0.2,
                api_keys=st.session_state.get("api_keys", {}),
            )
            st.session_state["sum_out"] = out
            log_event("Note Keeper", "Summarize", sum_model, est_tokens(base_note + out))
        except Exception as e:
            st.error(f"Summarize failed: {e}")
    if st.session_state.get("sum_out"):
        st.text_area("Summary (editable)", value=st.session_state["sum_out"], height=220, key="sum_out_edit")

    # Magic 3: Polisher
    st.markdown(f"#### 3) {t('magic_polish')}")
    pol_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gpt-4.1-mini") if "gpt-4.1-mini" in ALL_MODELS else 0, key="pol_model")
    pol_prompt = st.text_area(
        t("prompt"),
        value="Polish for clarity, grammar, and professional tone. Do not add facts. Keep Markdown structure.",
        height=120,
        key="pol_prompt",
    )
    if st.button(t("run"), key="pol_btn"):
        try:
            out = call_llm(
                model=pol_model,
                system_prompt=st.session_state["agents_cfg"]["agents"]["polisher"].get("system_prompt", ""),
                user_prompt=pol_prompt + "\n\n=== NOTE ===\n" + base_note,
                max_tokens=12000,
                temperature=0.15,
                api_keys=st.session_state.get("api_keys", {}),
            )
            st.session_state["pol_out"] = out
            log_event("Note Keeper", "Polisher", pol_model, est_tokens(base_note + out))
        except Exception as e:
            st.error(f"Polisher failed: {e}")
    if st.session_state.get("pol_out"):
        st.text_area("Polished (editable)", value=st.session_state["pol_out"], height=220, key="pol_out_edit")

    # Magic 4: Critique
    st.markdown(f"#### 4) {t('magic_critique')}")
    cri_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("claude-3-5-sonnet-20241022") if "claude-3-5-sonnet-20241022" in ALL_MODELS else 0, key="cri_model")
    cri_prompt = st.text_area(
        t("prompt"),
        value="Provide constructive critique: unclear areas, missing evidence, risks, contradictions, and specific improvement suggestions. Output Markdown with sections.",
        height=120,
        key="cri_prompt",
    )
    if st.button(t("run"), key="cri_btn"):
        try:
            out = call_llm(
                model=cri_model,
                system_prompt=st.session_state["agents_cfg"]["agents"]["critic"].get("system_prompt", ""),
                user_prompt=cri_prompt + "\n\n=== NOTE ===\n" + base_note,
                max_tokens=12000,
                temperature=0.35,
                api_keys=st.session_state.get("api_keys", {}),
            )
            st.session_state["cri_out"] = out
            log_event("Note Keeper", "Critique", cri_model, est_tokens(base_note + out))
        except Exception as e:
            st.error(f"Critique failed: {e}")
    if st.session_state.get("cri_out"):
        st.text_area("Critique (editable)", value=st.session_state["cri_out"], height=220, key="cri_out_edit")

    # Magic 5: Poet Mode
    st.markdown(f"#### 5) {t('magic_poet')}")
    poe_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gemini-3-flash-preview") if "gemini-3-flash-preview" in ALL_MODELS else 0, key="poe_model")
    poe_prompt = st.text_area(
        t("prompt"),
        value=f"Transform the note into poetic or artistic prose inspired by the current UI style: {st.session_state.settings['painter_style']}. Preserve core meaning; do not add facts.",
        height=120,
        key="poe_prompt",
    )
    if st.button(t("run"), key="poe_btn"):
        try:
            out = call_llm(
                model=poe_model,
                system_prompt=st.session_state["agents_cfg"]["agents"]["poet_laureate"].get("system_prompt", ""),
                user_prompt=poe_prompt + "\n\n=== NOTE ===\n" + base_note,
                max_tokens=12000,
                temperature=0.75,
                api_keys=st.session_state.get("api_keys", {}),
            )
            st.session_state["poe_out"] = out
            log_event("Note Keeper", "Poet Mode", poe_model, est_tokens(base_note + out))
        except Exception as e:
            st.error(f"Poet failed: {e}")
    if st.session_state.get("poe_out"):
        st.text_area("Poet output (editable)", value=st.session_state["poe_out"], height=220, key="poe_out_edit")

    # Magic 6: Translate to UI language
    st.markdown(f"#### 6) {t('magic_translate')}")
    tr_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0, key="tr_model")
    target = "Traditional Chinese (zh-TW)" if lang_code() == "zh-tw" else "English"
    tr_prompt = st.text_area(
        t("prompt"),
        value=f"Detect the language and translate the note into {target}. Preserve markdown structure and technical terms.",
        height=120,
        key="tr_prompt",
    )
    if st.button(t("run"), key="tr_btn"):
        try:
            out = call_llm(
                model=tr_model,
                system_prompt=st.session_state["agents_cfg"]["agents"]["translator"].get("system_prompt", ""),
                user_prompt=tr_prompt + "\n\n=== NOTE ===\n" + base_note,
                max_tokens=12000,
                temperature=0.2,
                api_keys=st.session_state.get("api_keys", {}),
            )
            st.session_state["tr_out"] = out
            log_event("Note Keeper", "Translate", tr_model, est_tokens(base_note + out))
        except Exception as e:
            st.error(f"Translate failed: {e}")
    if st.session_state.get("tr_out"):
        st.text_area("Translated (editable)", value=st.session_state["tr_out"], height=220, key="tr_out_edit")

    # download base note
    if base_note.strip():
        st.download_button(
            t("download_md"),
            data=base_note.encode("utf-8"),
            file_name="note.md",
            mime="text/markdown",
        )


# ============================================================
# 15) Agents Config tab (kept; improved resilience)
# ============================================================
def render_agents_config_tab():
    st.markdown(f"## {tl('Agents Config')}")
    agents_cfg = st.session_state["agents_cfg"]
    agents_dict = agents_cfg.get("agents", {})

    st.subheader("1) Agents Overview")
    if not agents_dict:
        st.warning("No agents found in current agents.yaml.")
    else:
        df = pd.DataFrame(
            [
                {
                    "agent_id": aid,
                    "name": acfg.get("name", ""),
                    "model": acfg.get("model", ""),
                    "max_tokens": acfg.get("max_tokens", ""),
                    "category": acfg.get("category", ""),
                    "supported_models": ", ".join(acfg.get("supported_models", [])) if isinstance(acfg.get("supported_models"), list) else "",
                    "description_tw": acfg.get("description_tw", ""),
                }
                for aid, acfg in agents_dict.items()
            ]
        )
        st.dataframe(df, use_container_width=True, height=320)

    st.markdown("---")
    st.subheader("2) Edit Full agents.yaml (raw text)")
    yaml_str_current = yaml.dump(st.session_state["agents_cfg"], allow_unicode=True, sort_keys=False)

    edited_yaml_text = st.text_area("agents.yaml (editable)", value=yaml_str_current, height=360, key="agents_yaml_text_editor")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Apply edited YAML to session", key="apply_edited_yaml"):
            try:
                cfg = yaml.safe_load(edited_yaml_text) or {}
                if not isinstance(cfg, dict) or "agents" not in cfg:
                    st.error("Parsed YAML missing top-level key 'agents'. No changes applied.")
                else:
                    st.session_state["agents_cfg"] = ensure_fallback_agents(cfg)
                    st.success("Updated agents.yaml in current session.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to parse edited YAML: {e}")

    with c2:
        uploaded_agents_tab = st.file_uploader("Upload agents.yaml file", type=["yaml", "yml"], key="agents_yaml_tab_uploader")
        if uploaded_agents_tab is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents_tab.read()) or {}
                if "agents" in cfg:
                    st.session_state["agents_cfg"] = ensure_fallback_agents(cfg)
                    st.success("Uploaded agents.yaml applied to this session.")
                    st.rerun()
                else:
                    st.warning("Uploaded file has no top-level 'agents' key. Ignoring.")
            except Exception as e:
                st.error(f"Failed to parse uploaded YAML: {e}")

    with c3:
        st.download_button(
            "Download current agents.yaml",
            data=yaml_str_current.encode("utf-8"),
            file_name="agents.yaml",
            mime="text/yaml",
            key="download_agents_yaml_current",
        )


# ============================================================
# 16) Main render
# ============================================================
render_sidebar()
apply_style_engine(st.session_state.settings["theme"], st.session_state.settings["painter_style"])
render_wow_header()

tab_labels = [
    tl("Dashboard"),
    tl("Workflow Studio"),
    tl("TW Premarket"),
    tl("510k_tab"),
    tl("PDF → Markdown"),
    tl("Checklist & Report"),
    tl("Note Keeper & Magics"),
    tl("Agents Config"),
]
tabs = st.tabs(tab_labels)

with tabs[0]:
    render_dashboard()
with tabs[1]:
    render_workflow_studio()
with tabs[2]:
    render_tw_premarket_tab()
with tabs[3]:
    render_510k_tab()
with tabs[4]:
    render_pdf_to_md_tab()
with tabs[5]:
    render_510k_review_pipeline_tab()
with tabs[6]:
    render_note_keeper_tab()
with tabs[7]:
    render_agents_config_tab()
