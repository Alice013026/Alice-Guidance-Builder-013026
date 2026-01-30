import os
import json
import base64
import random
import re
import difflib
from datetime import datetime, date
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

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
except ImportError:
    pytesseract = None
    convert_from_bytes = None
    Image = None

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
    # Anthropic
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
        # ... (Previous translations kept, truncated for brevity) ...
        "run_agent": "Run Agent",
        "reload_defaults": "Reload Default Datasets",
    },
    "zh-tw": {
        "app_title": "Antigravity AI 工作空間",
        "top_tagline": "WOW 級：代理工作流、互動儀表板、筆記魔法、藝術主題",
        "theme": "主題",
        # ... (Previous translations kept, truncated for brevity) ...
        "run_agent": "執行代理",
        "reload_defaults": "重新載入預設資料集",
    },
}


def lang_code() -> str:
    return st.session_state.settings.get("language", "zh-tw")


def t(key: str) -> str:
    return I18N.get(lang_code(), I18N["en"]).get(key, key)


# ============================================================
# 3) Style Engine
# ============================================================
PAINTER_STYLES_20 = [
    "Van Gogh", "Picasso", "Monet", "Da Vinci", "Dali", "Mondrian", "Warhol", "Rembrandt", "Klimt", "Hokusai",
    "Munch", "O'Keeffe", "Basquiat", "Matisse", "Pollock", "Kahlo", "Hopper", "Magritte", "Cyberpunk", "Bauhaus",
]

STYLE_TOKENS: Dict[str, Dict[str, str]] = {
    "Van Gogh": {"--bg1": "#0b1020", "--bg2": "#1f3b73", "--accent": "#f7c948", "--accent2": "#60a5fa", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    # ... (Other styles omitted for brevity, assume they exist) ...
    "Cyberpunk": {"--bg1": "#050816", "--bg2": "#1b0033", "--accent": "#22d3ee", "--accent2": "#a78bfa", "--card": "rgba(255,255,255,0.08)", "--border": "rgba(34,211,238,0.25)"},
}

def apply_style_engine(theme_mode: str, painter_style: str):
    tokens = STYLE_TOKENS.get(painter_style, STYLE_TOKENS["Van Gogh"])
    is_dark = theme_mode.lower() == "dark"
    text_color = "#e5e7eb" if is_dark else "#0f172a"
    subtext = "#cbd5e1" if is_dark else "#334155"
    shadow = "0 18px 50px rgba(0,0,0,0.38)" if is_dark else "0 18px 50px rgba(2,6,23,0.18)"
    glass = "rgba(17,24,39,0.38)" if is_dark else "rgba(255,255,255,0.55)"

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
    body {{
        color: var(--text);
        background: radial-gradient(1200px circle at 12% 8%, var(--bg2) 0%, transparent 55%),
                    radial-gradient(900px circle at 88% 18%, var(--accent2) 0%, transparent 50%),
                    linear-gradient(135deg, var(--bg1), var(--bg2));
        background-attachment: fixed;
    }}
    .block-container {{ padding-top: 1.0rem; padding-bottom: 3.5rem; }}
    .wow-hero {{
        border-radius: var(--radius2); padding: 18px 18px; margin: 0 0 14px 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02));
        border: 1px solid var(--border); box-shadow: var(--shadow); backdrop-filter: blur(12px);
    }}
    .wow-title {{ font-size: 1.35rem; font-weight: 800; letter-spacing: 0.02em; margin: 0; color: var(--text); }}
    .wow-subtitle {{ margin: 6px 0 0 0; color: var(--subtext); font-size: 0.95rem; }}
    .wow-card {{
        border-radius: var(--radius); padding: 14px 16px; background: var(--glass);
        border: 1px solid var(--border); box-shadow: var(--shadow); backdrop-filter: blur(12px);
    }}
    .wow-kpi {{ font-size: 1.55rem; font-weight: 800; margin-top: 4px; }}
    .wow-muted {{ color: var(--subtext); font-size: 0.92rem; }}
    .stButton > button {{
        border-radius: 999px !important; border: 1px solid var(--border) !important;
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #0b1020 !important; font-weight: 800 !important;
    }}
    .wow-badge {{
        display:inline-flex; align-items:center; padding: 3px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 700; border: 1px solid var(--border);
        background: rgba(255,255,255,0.10); color: var(--text);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# 4) Data Loading Logic (Datasets & Guidance)
# ============================================================
def load_default_datasets_from_file() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads defaultdataset.json containing tw_cases and k510_checklists."""
    tw_cases = {}
    k510_checklists = {}
    try:
        with open("defaultdataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            tw_cases = data.get("tw_cases", {})
            k510_checklists = data.get("k510_checklists", {})
    except FileNotFoundError:
        st.error("Default dataset file 'defaultdataset.json' not found.")
    except Exception as e:
        st.error(f"Error loading defaultdataset.json: {e}")
    return tw_cases, k510_checklists

def load_default_guidance_from_file() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Loads defaultguide.md and parses sections for TW and 510k guidances."""
    tw_guides = {}
    k510_guides = {}
    
    try:
        with open("defaultguide.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Regex to find sections: <!-- BEGIN_SECTION: id | TITLE: title --> ... <!-- END_SECTION -->
        pattern = re.compile(
            r"<!--\s*BEGIN_SECTION:\s*(.*?)\s*\|\s*TITLE:\s*(.*?)\s*-->(.*?)<!--\s*END_SECTION\s*-->", 
            re.DOTALL
        )
        matches = pattern.findall(content)
        
        for key, title, body in matches:
            key = key.strip()
            title = title.strip()
            body = body.strip()
            entry = {"title": title, "md": body}
            
            if key.startswith("tw_"):
                tw_guides[key] = entry
            elif key.startswith("k510_"):
                k510_guides[key] = entry
                
    except FileNotFoundError:
        st.error("Default guidance file 'defaultguide.md' not found.")
    except Exception as e:
        st.error(f"Error loading defaultguide.md: {e}")
        
    return tw_guides, k510_guides

def refresh_defaults():
    tw_c, k510_c = load_default_datasets_from_file()
    tw_g, k510_g = load_default_guidance_from_file()
    st.session_state["DEFAULT_TW_CASESETS"] = tw_c
    st.session_state["DEFAULT_510K_CHECKLIST_SETS"] = k510_c
    st.session_state["DEFAULT_TW_GUIDANCES"] = tw_g
    st.session_state["DEFAULT_510K_GUIDANCES"] = k510_g
    st.toast("Defaults reloaded successfully!")


# ============================================================
# 5) State init
# ============================================================
if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "theme": "Dark", "language": "zh-tw", "painter_style": "Van Gogh",
        "model": "gpt-4o-mini", "max_tokens": 12000, "temperature": 0.2, "token_budget_est": 250_000,
    }

if "history" not in st.session_state:
    st.session_state["history"] = []

if "api_keys" not in st.session_state:
    st.session_state["api_keys"] = {"openai": "", "gemini": "", "anthropic": "", "grok": ""}

if "workflow" not in st.session_state:
    st.session_state["workflow"] = {"steps": [], "cursor": 0, "input": "", "outputs": [], "statuses": []}

# Initialize defaults if not present
if "DEFAULT_TW_CASESETS" not in st.session_state:
    refresh_defaults()

# Rule-mapping dictionary (editable)
if "tw_field_mapping" not in st.session_state:
    st.session_state["tw_field_mapping"] = {
        "公文文號": "doc_no", "docNo": "doc_no", "電子流水號": "e_no", "eNo": "e_no",
        "申請日": "apply_date", "applyDate": "apply_date", "案件類型": "case_type",
        "醫療器材類型": "device_category", "案件種類": "case_kind", "產地": "origin",
        "產品等級": "product_class", "有無類似品": "similar", "替代條款": "replace_flag",
        "前次申請案號": "prior_app_no", "中文名稱": "name_zh", "品名(中)": "name_zh",
        "英文名稱": "name_en", "品名(英)": "name_en", "適應症": "indications",
        "規格": "spec_comp", "主類別": "main_cat", "品項代碼": "item_code",
        "品項名稱": "item_name", "統一編號": "uniform_id", "醫療器材商名稱": "firm_name",
        "醫療器材商地址": "firm_addr", "負責人": "resp_name", "聯絡人": "contact_name",
        "電話": "contact_tel", "傳真": "contact_fax", "電子郵件": "contact_email",
        "製造方式": "manu_type", "製造廠名稱": "manu_name", "製造國別": "manu_country",
        "製造廠地址": "manu_addr", "已確認證照相符": "confirm_match"
    }

TW_APP_FIELDS = [
    "doc_no", "e_no", "apply_date", "case_type", "device_category", "case_kind",
    "origin", "product_class", "similar", "replace_flag", "prior_app_no",
    "name_zh", "name_en", "indications", "spec_comp",
    "main_cat", "item_code", "item_name",
    "uniform_id", "firm_name", "firm_addr",
    "resp_name", "contact_name", "contact_tel", "contact_fax", "contact_email",
    "confirm_match", "cert_raps", "cert_ahwp", "cert_other",
    "manu_type", "manu_name", "manu_country", "manu_addr", "manu_note",
    "auth_applicable", "auth_desc", "cfs_applicable", "cfs_desc",
    "qms_applicable", "qms_desc", "similar_info", "labeling_info", "tech_file_info",
    "preclinical_info", "preclinical_replace", "clinical_just", "clinical_info",
]
BOOL_FIELDS = {"confirm_match", "cert_raps", "cert_ahwp"}


# ============================================================
# 6) API key logic & LLM Router (Standard)
# ============================================================
def env_key_present(env_var: str) -> bool:
    v = os.getenv(env_var, "")
    return bool(v and v.strip())

def get_api_key(provider: str, api_keys: Dict[str, str]) -> str:
    mapping = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "grok": "GROK_API_KEY"}
    return api_keys.get(provider) or os.getenv(mapping.get(provider, "")) or ""

def api_status(provider: str) -> Tuple[str, str]:
    mapping = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "grok": "GROK_API_KEY"}
    if env_key_present(mapping[provider]): return "env", t("active_env")
    if st.session_state["api_keys"].get(provider, "").strip(): return "session", t("provided_session")
    return "missing", t("missing")

def call_llm(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 12000, temperature: float = 0.2, api_keys: Optional[dict] = None) -> str:
    provider = get_provider(model)
    key = get_api_key(provider, api_keys or {})
    if not key: raise RuntimeError(f"Missing API key for provider: {provider}")

    if provider == "openai":
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content
    if provider == "gemini":
        genai.configure(api_key=key)
        llm = genai.GenerativeModel(model)
        resp = llm.generate_content(system_prompt + "\n\n" + user_prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
        return resp.text
    if provider == "anthropic":
        client = Anthropic(api_key=key)
        resp = client.messages.create(model=model, system=system_prompt, max_tokens=max_tokens, temperature=temperature, messages=[{"role": "user", "content": user_prompt}])
        return resp.content[0].text
    if provider == "grok":
        with httpx.Client(base_url="https://api.x.ai/v1", timeout=90) as client:
            resp = client.post("/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "max_tokens": max_tokens, "temperature": temperature})
            return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"Unsupported provider")


# ============================================================
# 7) Utilities (PDF, Markdown, etc)
# ============================================================
def est_tokens(text: str) -> int: return max(1, int(len(text or "") / 4))
def log_event(tab: str, agent: str, model: str, tokens_est: int, meta: Optional[dict] = None):
    st.session_state["history"].append({"tab": tab, "agent": agent, "model": model, "tokens_est": int(tokens_est), "ts": datetime.utcnow().isoformat(), "meta": meta or {}})

def extract_pdf_pages_to_text(file, start_page: int, end_page: int, use_ocr: bool = False) -> str:
    pdf_text = ""
    try:
        reader = PdfReader(file)
        n = len(reader.pages)
        raw_texts = []
        for i in range(max(0, start_page - 1), min(n, end_page)):
            raw_texts.append(reader.pages[i].extract_text() or "")
        pdf_text = "\n\n".join(raw_texts).strip()
    except Exception as e: print(f"pypdf extraction error: {e}")
    if use_ocr and pytesseract and convert_from_bytes:
        try:
            file.seek(0)
            images = convert_from_bytes(file.read(), first_page=start_page, last_page=end_page)
            ocr_text = [pytesseract.image_to_string(img, lang="eng+chi_tra") for img in images]
            return "\n\n".join(ocr_text).strip()
        except Exception as e: return pdf_text + f"\n\n[OCR failed: {e}]"
    return pdf_text

def status_row(label: str, status: str):
    color = {"pending": "dot-amber", "running": "dot-amber", "done": "dot-green", "error": "dot-red", "idle": "dot-amber", "thinking": "dot-amber"}.get(status, "dot-amber")
    st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin:2px 0;"><span class="dot {color}"></span><div style="font-weight:800;">{label}</div><span class="wow-badge">{status}</span></div>', unsafe_allow_html=True)

def normalize_md(md: str) -> str: return re.sub(r"\n{3,}", "\n\n", (md or "").strip())

def diff_markdown(a: str, b: str) -> str:
    diff = difflib.unified_diff((a or "").splitlines(keepends=True), (b or "").splitlines(keepends=True), fromfile="A.md", tofile="B.md")
    return "".join(diff).strip()

def merge_guidance_markdowns(mds: List[str], extra_rules_md: str = "") -> str:
    parts = [normalize_md(x) for x in (mds or []) if normalize_md(x)]
    if extra_rules_md.strip(): parts.append("## 自訂追加規則\n" + normalize_md(extra_rules_md))
    return normalize_md("\n\n---\n\n".join(parts))

def _to_bool(v: Any) -> bool:
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y", "是", "有", "checked"}

def standardize_tw_record_rule_mapping(raw: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (raw or {}).items():
        if k in TW_APP_FIELDS: out[k] = v
        elif k in mapping: out[mapping[k]] = v
        elif str(k).strip() in mapping: out[mapping[str(k).strip()]] = v
    for f in TW_APP_FIELDS:
        if f not in out: out[f] = False if f in BOOL_FIELDS else ""
    for bf in BOOL_FIELDS: out[bf] = _to_bool(out.get(bf, False))
    return out


# ============================================================
# 8) Agents YAML
# ============================================================
def load_agents_cfg() -> Dict[str, Any]:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f: return yaml.safe_load(f) or {"agents": {}}
    except Exception: return {"agents": {}}

def ensure_fallback_agents(cfg: Dict[str, Any]) -> Dict[str, Any]:
    agents = cfg.setdefault("agents", {})
    def put(aid, obj):
        if aid not in agents: agents[aid] = obj
    put("fda_510k_intel_agent", {"name": "510(k) Intelligence Agent", "model": "gpt-4o-mini", "system_prompt": "You are an FDA 510(k) analyst."})
    put("pdf_to_markdown_agent", {"name": "PDF → Markdown Agent", "model": "gemini-2.5-flash", "system_prompt": "You convert PDF-extracted text into clean markdown."})
    put("tw_screen_review_agent", {"name": "TFDA 預審形式審查代理", "model": "gemini-2.5-flash", "system_prompt": "You are a TFDA premarket screen reviewer."})
    put("tw_app_doc_helper", {"name": "TFDA 申請書撰寫助手", "model": "gpt-4o-mini", "system_prompt": "You help improve TFDA application documents."})
    put("note_organizer", {"name": "Note Organizer", "model": "gpt-4o-mini", "system_prompt": "You turn messy notes into structured markdown."})
    put("keyword_extractor", {"name": "Keyword Extractor", "model": "gemini-2.5-flash", "system_prompt": "You extract keywords."})
    put("polisher", {"name": "Polisher", "model": "gpt-4.1-mini", "system_prompt": "You rewrite text for clarity."})
    put("critic", {"name": "Creative Critic", "model": "claude-3-5-sonnet-20241022", "system_prompt": "You give constructive critique."})
    put("poet_laureate", {"name": "Poet Laureate", "model": "gemini-3-flash-preview", "system_prompt": "Transform content into poetic prose."})
    put("translator", {"name": "Translator", "model": "gemini-2.5-flash", "system_prompt": "You translate accurately."})
    return cfg

if "agents_cfg" not in st.session_state: st.session_state["agents_cfg"] = ensure_fallback_agents(load_agents_cfg())

def agent_run_ui(agent_id, tab_key, default_prompt, default_input_text="", allow_model_override=True, tab_label_for_history=None):
    agent_cfg = st.session_state["agents_cfg"].get("agents", {}).get(agent_id, {})
    name = agent_cfg.get("name", agent_id)
    base_model = agent_cfg.get("model", st.session_state.settings["model"])
    
    st.markdown(f"**Agent:** {name}")
    c1, c2, c3 = st.columns([2.2, 1.0, 1.0])
    with c1: prompt = st.text_area("System/User Prompt", value=st.session_state.get(f"{tab_key}_prompt", default_prompt), height=100, key=f"{tab_key}_prompt")
    with c2: model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(base_model) if base_model in ALL_MODELS else 0, key=f"{tab_key}_model")
    with c3: max_tokens = st.number_input("Max Tokens", 1000, 100000, 12000, key=f"{tab_key}_max_tokens")
    
    inp = st.text_area("Input Text", value=st.session_state.get(f"{tab_key}_input", default_input_text), height=150, key=f"{tab_key}_input")
    
    if st.button(f"Run {name}", key=f"{tab_key}_run"):
        with st.spinner("Thinking..."):
            try:
                out = call_llm(model, agent_cfg.get("system_prompt", ""), prompt + "\n\n" + inp, max_tokens, 0.2, st.session_state.get("api_keys"))
                st.session_state[f"{tab_key}_output"] = out
                log_event(tab_label_for_history or tab_key, name, model, est_tokens(inp+out))
            except Exception as e: st.error(f"Error: {e}")
            
    if st.session_state.get(f"{tab_key}_output"):
        st.text_area("Output", value=st.session_state[f"{tab_key}_output"], height=250)


# ============================================================
# 9) UI Sections
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(f"## {t('global_settings')}")
        if st.button(t("reload_defaults")):
            refresh_defaults()
            st.rerun()
            
        theme_choice = st.radio(t("theme"), [t("light"), t("dark")], index=0 if st.session_state.settings["theme"] == "Light" else 1, horizontal=True)
        st.session_state.settings["theme"] = "Light" if theme_choice == t("light") else "Dark"
        
        style = st.selectbox(t("painter_style"), PAINTER_STYLES_20, index=PAINTER_STYLES_20.index(st.session_state.settings["painter_style"]))
        st.session_state.settings["painter_style"] = style
        
        st.session_state.settings["model"] = st.selectbox(t("default_model"), ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]))
        
        st.markdown("---")
        st.markdown(f"## {t('api_keys')}")
        keys = dict(st.session_state["api_keys"])
        keys["openai"] = st.text_input("OpenAI API Key", type="password", value=keys.get("openai", ""))
        keys["gemini"] = st.text_input("Gemini API Key", type="password", value=keys.get("gemini", ""))
        st.session_state["api_keys"] = keys


def render_tw_premarket_tab():
    st.markdown("## 第二、三等級醫療器材查驗登記（TW Premarket）")
    
    DEFAULT_CASES = st.session_state.get("DEFAULT_TW_CASESETS", {})
    DEFAULT_GUIDES = st.session_state.get("DEFAULT_TW_GUIDANCES", {})

    st.markdown("### 0) Case Dataset")
    if not DEFAULT_CASES:
        st.warning("No default cases loaded. Check `defaultdataset.json`.")
        
    ds_keys = list(DEFAULT_CASES.keys())
    if ds_keys:
        ds_labels = [DEFAULT_CASES[k]["title"] for k in ds_keys]
        sel = st.selectbox("Select default dataset", ds_labels)
        if st.button("Load Dataset"):
             key = ds_keys[ds_labels.index(sel)]
             st.session_state["tw_cases_dataset"] = DEFAULT_CASES[key]["cases"]
             st.success("Loaded.")
             
    st.markdown("### 1) Guidance")
    if not DEFAULT_GUIDES:
        st.warning("No default guidance loaded. Check `defaultguide.md`.")
        
    g_keys = list(DEFAULT_GUIDES.keys())
    if g_keys:
        g_labels = [DEFAULT_GUIDES[k]["title"] for k in g_keys]
        sel_g = st.selectbox("Select Guidance", g_labels)
        if st.button("Load Guidance"):
            key = g_keys[g_labels.index(sel_g)]
            st.session_state["tw_guidance_effective_md"] = DEFAULT_GUIDES[key]["md"]
            st.success("Loaded.")
            st.rerun()
            
    st.text_area("Guidance Editor", value=st.session_state.get("tw_guidance_effective_md", ""), height=300)

    # (Rest of TW logic - simplified for brevity, assume full logic from original here)
    st.markdown("---")
    st.info("Form filling and AI Agent logic would appear here (omitted for file size).")


def render_510k_review_tab():
    st.markdown("## 510(k) Review Pipeline")
    
    DEFAULT_CHK = st.session_state.get("DEFAULT_510K_CHECKLIST_SETS", {})
    DEFAULT_G_510K = st.session_state.get("DEFAULT_510K_GUIDANCES", {})

    c1, c2 = st.columns(2)
    with c1:
        if DEFAULT_CHK:
            k = st.selectbox("Checklist", list(DEFAULT_CHK.keys()), format_func=lambda x: DEFAULT_CHK[x]["title"])
            if st.button("Load Checklist"):
                st.session_state["k510_checklist_dataset"] = DEFAULT_CHK[k]["items"]
    with c2:
        if DEFAULT_G_510K:
            k = st.selectbox("Guidance", list(DEFAULT_G_510K.keys()), format_func=lambda x: DEFAULT_G_510K[x]["title"])
            if st.button("Load 510k Guidance"):
                st.session_state["k510_guidance_effective_md"] = DEFAULT_G_510K[k]["md"]
                
    st.write("Checklist items loaded:", len(st.session_state.get("k510_checklist_dataset", [])))


# ============================================================
# Main
# ============================================================
render_sidebar()
apply_style_engine(st.session_state.settings["theme"], st.session_state.settings["painter_style"])

tabs = st.tabs(["Dashboard", "TW Premarket", "510(k) Pipeline", "Agents Config"])

with tabs[0]:
    st.title("Antigravity AI Workspace")
    st.write("Welcome. Use the sidebar to reload default datasets.")
    
with tabs[1]:
    render_tw_premarket_tab()

with tabs[2]:
    render_510k_review_tab()

with tabs[3]:
    st.write(st.session_state["agents_cfg"])
147.0s
Model
ThinkingThoughts
Refactoring the Application
I'm now zeroing in on the core goal: refactoring app.py, a hefty Streamlit application for AI agents dealing with medical device regulations like TW Premarket and FDA 510(k). I'm focusing on breaking down the monolithic structure into more manageable, testable components, targeting a design where each part can be easily updated and doesn't rely on the parts around it.
Isolating Hardcoded Data
I've made progress in isolating the hardcoded data. I've pinpointed the exact sections within app.py that hold the default datasets and guidance text. My next step will be to create and populate external files (defaultdataset.json, defaultguide.md) with this content. My ultimate objective will be to modify app.py to dynamically load these files, rather than embedding the data directly within the code.
Defining the Core Modules
I'm now identifying the key modules I'll need. This means looking at helper functions, UI rendering components (sidebar, tabs, etc.), and implementations for each tab (Dashboard, Workflow, TW Premarket, 510k). This is a top-level breakdown before I dive into details.
Implementing Data Loading
I'm now implementing the dynamic loading of data. I'm writing the functions load_default_datasets_from_file() and load_default_guidance_from_file() to read from external files and load them into session state. I'm also adding a "Reload" button to the sidebar for easy updates. The UI components are being updated to reference the session state. I'm currently constructing the JSON file from existing constants like DEFAULT_TW_CASESETS and DEFAULT_510K_CHECKLIST_SETS and formatting markdown for parsing.
Updating Data Loading
I'm now focusing on the specific file loading mechanics. The JSON loading uses json.load(f), while the markdown parsing employs re.findall with re.DOTALL to extract the relevant text. I'm verifying that data loading from these external sources accurately populates st.session_state variables like DEFAULT_TW_CASESETS.
Defining UI Components
My current focus is on defining and organizing the user interface components. I am planning to keep the tw_field_mapping in code because it functions as a schema definition for standardization, but I will move the case data. I'm focusing on UI components like run_ui, render_dashboard, and render_workflow_studio. I am also working on how to render note_keeper_tab, render_agents_config_tab, and handle PDF processing logic.
Collapse to hide model thoughts

chevron_right
Here is the complete solution.

I have separated the hardcoded data into `defaultdataset.json` and `defaultguide.md`, and refactored `app.py` to preserve all original functionality (Dashboard, Workflow, PDF tools, Agents, etc.) while loading this data dynamically.

### 1. `defaultdataset.json`
Save this file in the same directory as `app.py`. It contains the Case Datasets (TW) and Checklists (510k).

```json
{
  "tw_cases": {
    "tw_default_syringe_single": {
      "title": "Default Case (Syringe, single)",
      "cases": [
        {
          "doc_no": "衛授醫器字第1130001234號",
          "e_no": "MDE-2026-000198",
          "apply_date": "2026-01-15",
          "case_type": "一般申請案",
          "device_category": "一般醫材",
          "case_kind": "新案",
          "origin": "輸入",
          "product_class": "第二等級",
          "similar": "有",
          "replace_flag": "否",
          "prior_app_no": "",
          "name_zh": "一次性使用無菌注射器（含針）",
          "name_en": "Single-use Sterile Syringe with Needle",
          "indications": "供醫療專業人員於臨床注射與抽取液體使用。詳如核定之中文說明書。",
          "spec_comp": "容量 1 mL / 3 mL / 5 mL；針規 23G / 25G；主要材質：PP（筒身）、不鏽鋼（針管）。",
          "main_cat": "J.一般醫院及個人使用裝置",
          "item_code": "J.4830",
          "item_name": "注射器及其附件",
          "uniform_id": "24567890",
          "firm_name": "安澤醫材股份有限公司",
          "firm_addr": "臺北市內湖區瑞光路321號8樓",
          "resp_name": "陳志豪",
          "contact_name": "林怡君",
          "contact_tel": "02-2799-1234",
          "contact_fax": "02-2799-5678",
          "contact_email": "regulatory@anzenmed.com.tw",
          "confirm_match": true,
          "cert_raps": false,
          "cert_ahwp": true,
          "cert_other": "ISO 13485 內部稽核員訓練證明（2025）",
          "manu_type": "單一製造廠",
          "manu_name": "GLOBAL MEDICAL DEVICES CO., LTD.",
          "manu_country": "UNITED STATES",
          "manu_addr": "1200 Innovation Drive, Irvine, CA 92618, USA",
          "manu_note": "原廠負責製造、包裝與最終放行；台灣醫療器材商負責進口與上市後監督。",
          "auth_applicable": "適用",
          "auth_desc": "附原廠授權登記書正本與中譯本；授權範圍包含本產品全規格與型號。",
          "cfs_applicable": "適用",
          "cfs_desc": "提供美國出產國製售證明／同等效力文件影本，並附中譯本。",
          "qms_applicable": "適用",
          "qms_desc": "原廠 ISO 13485:2016 證書有效期至 2027-06；驗證範圍涵蓋本產品。",
          "similar_info": "同類產品已於多國上市；提供與既有同類品比較：材質、滅菌方式、容量與針規差異。類似品許可證資訊另附表。",
          "labeling_info": "中文標籤與說明書擬稿含：產品名稱、規格、滅菌方式（EO）、有效期限、注意事項、單次使用警語。",
          "tech_file_info": "提供產品結構圖、材料清單、尺寸規格、製程概要、滅菌確效摘要、包裝完整性測試摘要。",
          "preclinical_info": "臨床前測試摘要：生物相容性（依 ISO 10993 系列）、滅菌確效（ISO 11135）、包裝完整性與運輸模擬、針尖穿刺力與滑動性測試。",
          "preclinical_replace": "",
          "clinical_just": "不適用",
          "clinical_info": "依現行規範與產品風險評估，本品屬成熟技術與同類產品，提供臨床前測試與上市後監測計畫，無臨床試驗需求。"
        }
      ]
    },
    "tw_mock_suture_import": {
        "title": "Mock Dataset: Surgical Suture (Import, Class II)",
        "cases": [
            {
                "doc_no": "衛授醫器字第1150000456號",
                "e_no": "MDE-2026-001102",
                "apply_date": "2026-03-12",
                "case_type": "一般申請案",
                "device_category": "一般醫材",
                "case_kind": "新案",
                "origin": "輸入",
                "product_class": "第二等級",
                "similar": "有",
                "replace_flag": "否",
                "prior_app_no": "",
                "name_zh": "一次性使用無菌外科縫合針線組",
                "name_en": "Single-use Sterile Surgical Suture with Needle",
                "indications": "供醫療專業人員於外科手術中進行軟組織縫合使用。詳如核定之中文說明書。",
                "spec_comp": "縫線材質：PGA/PP（依型號）；縫線規格：USP 2-0/3-0/4-0；針型：圓針/三角針；線長：45/75 cm。",
                "main_cat": "I.一般及整形外科手術",
                "item_code": "I.3670",
                "item_name": "外科縫合材料",
                "uniform_id": "24813579",
                "firm_name": "曜澄醫療器材股份有限公司",
                "firm_addr": "新北市板橋區文化路二段88號12樓",
                "resp_name": "吳冠廷",
                "contact_name": "黃筱雯",
                "contact_tel": "02-2258-6600",
                "contact_fax": "02-2258-6611",
                "contact_email": "ra@yaochengmed.com.tw",
                "confirm_match": true,
                "cert_raps": true,
                "cert_ahwp": false,
                "cert_other": "UDI/Labeling training certificate (2025)",
                "manu_type": "單一製造廠",
                "manu_name": "SURGITECH MEDICAL INC.",
                "manu_country": "UNITED STATES",
                "manu_addr": "5000 Medical Parkway, San Diego, CA 92121, USA",
                "manu_note": "原廠負責製造、滅菌（EO）與最終放行；臺灣醫療器材商負責進口、分銷與上市後監督。",
                "auth_applicable": "適用",
                "auth_desc": "附原廠授權登記書正本與中譯本；授權範圍涵蓋所有縫線規格與針型。",
                "cfs_applicable": "適用",
                "cfs_desc": "提供美國出產國製售證明影本，含簽發機關資訊與有效日期。",
                "qms_applicable": "適用",
                "qms_desc": "原廠 ISO 13485:2016 證書有效期至 2028-01；驗證範圍涵蓋設計與製造及本產品類別。",
                "similar_info": "提供與已上市同類縫合線之比較表：材質、吸收性、拉伸強度、針型、滅菌方式等差異；另附同類品許可證資訊。",
                "labeling_info": "中文標籤/說明書擬稿含：產品規格、針型、材質、無菌、單次使用、滅菌方式（EO）、效期、批號、禁忌與注意事項。",
                "tech_file_info": "提供產品結構與規格表、材料清單、製程概述、滅菌確效摘要、包裝完整性測試摘要與性能測試摘要。",
                "preclinical_info": "臨床前測試摘要：生物相容性（ISO 10993 系列）、滅菌確效（ISO 11135）、包裝完整性與運輸模擬、拉伸強度、針-線連接強度等性能測試。",
                "preclinical_replace": "",
                "clinical_just": "不適用",
                "clinical_info": "本品屬成熟技術之縫合材料，風險評估與法規要求以臨床前性能/生物相容性及滅菌確效為主，無臨床試驗需求。"
            },
            {
                "doc_no": "衛授醫器字第1150000457號",
                "e_no": "MDE-2026-001103",
                "apply_date": "2026-03-12",
                "case_type": "一般申請案",
                "device_category": "一般醫材",
                "case_kind": "新案",
                "origin": "輸入",
                "product_class": "第二等級",
                "similar": "有",
                "replace_flag": "否",
                "prior_app_no": "",
                "name_zh": "一次性使用無菌外科縫合線（無針）",
                "name_en": "Single-use Sterile Surgical Suture (Needleless)",
                "indications": "供醫療專業人員於外科手術或傷口處置中進行軟組織縫合使用。詳如核定之中文說明書。",
                "spec_comp": "縫線材質：Nylon/PP（依型號）；縫線規格：USP 2-0/3-0/4-0；線長：45/90 cm；無針版本。",
                "main_cat": "I.一般及整形外科手術",
                "item_code": "I.3670",
                "item_name": "外科縫合材料",
                "uniform_id": "24813579",
                "firm_name": "曜澄醫療器材股份有限公司",
                "firm_addr": "新北市板橋區文化路二段88號12樓",
                "resp_name": "吳冠廷",
                "contact_name": "黃筱雯",
                "contact_tel": "02-2258-6600",
                "contact_fax": "02-2258-6611",
                "contact_email": "ra@yaochengmed.com.tw",
                "confirm_match": true,
                "cert_raps": true,
                "cert_ahwp": false,
                "cert_other": "UDI/Labeling training certificate (2025)",
                "manu_type": "單一製造廠",
                "manu_name": "SURGITECH MEDICAL INC.",
                "manu_country": "UNITED STATES",
                "manu_addr": "5000 Medical Parkway, San Diego, CA 92121, USA",
                "manu_note": "與同系列含針產品同廠製造與滅菌（EO）；臺灣商負責上市後監督。",
                "auth_applicable": "適用",
                "auth_desc": "授權書範圍包含本無針版本型號與規格；附中譯本。",
                "cfs_applicable": "適用",
                "cfs_desc": "同系列產品出產國製售證明文件；附簽發機關資訊。",
                "qms_applicable": "適用",
                "qms_desc": "ISO 13485:2016 證書有效期至 2028-01。",
                "similar_info": "提供無針縫合線與市售同類品比較：材質、線徑、結節強度等。",
                "labeling_info": "中文標示含：無菌、單次使用、滅菌方式（EO）、效期、批號及禁忌注意事項。",
                "tech_file_info": "提供規格表、材料清單、製程簡述、滅菌確效摘要、包裝完整性摘要及性能測試摘要。",
                "preclinical_info": "生物相容性、滅菌確效、包裝完整性與性能（拉伸/結節強度）測試摘要。",
                "preclinical_replace": "",
                "clinical_just": "不適用",
                "clinical_info": "成熟技術產品，以臨床前資料支持安全有效。"
            }
        ]
    }
  },
  "k510_checklists": {
    "k510_default_general_v1": {
      "title": "Default 510(k) Checklist (General v1)",
      "items": [
        {"section": "Administrative", "item": "Cover letter and contact info", "expected": "Present and complete", "notes": ""},
        {"section": "Device Description", "item": "Device description and technological characteristics", "expected": "Clear and complete", "notes": ""},
        {"section": "Indications", "item": "Indications for Use statement", "expected": "Consistent across submission", "notes": ""},
        {"section": "Predicate Comparison", "item": "Substantial equivalence rationale", "expected": "Clear comparison table", "notes": ""},
        {"section": "Performance", "item": "Bench testing summary", "expected": "Adequate to support SE", "notes": ""},
        {"section": "Performance", "item": "Software documentation (if applicable)", "expected": "Appropriate level of concern", "notes": ""},
        {"section": "Performance", "item": "Biocompatibility (if patient-contacting)", "expected": "ISO 10993 rationale/testing", "notes": ""},
        {"section": "Labeling", "item": "Labeling/IFU review", "expected": "Consistent with indications and risks", "notes": ""},
        {"section": "Risk Management", "item": "Risk analysis summary", "expected": "Hazards and mitigations addressed", "notes": ""}
      ]
    }
  }
}
```

### 2. `defaultguide.md`
Save this file in the same directory. Note the delimiters (`<!-- BEGIN...`) used to separate different guidance texts.

```markdown
<!-- BEGIN_SECTION: tw_guidance_syringe_original | TITLE: Guidance: Syringe (Import, Class II) -->
# 預審/形式審查指引（Mock）—第二等級輸入一般醫材：一次性使用無菌注射器（含針）

## 0. 審查目的
本指引用於形式審查（預審）階段，確認申請書與主要附件是否齊備、資訊是否一致、文件可追溯性是否足以進入技術審查。

---

## 1. 必要文件清單（預期應附）
> 審查時請逐一確認「是否提及」「是否檢附」「是否有效/在效期內」。

1. 申請書（第二、三等級醫療器材查驗登記申請書）
2. 醫療器材商許可執照（名稱/地址/負責人須與申請書一致）
3. 原廠授權登記書（輸入案通常必附）
4. 出產國製售證明（CFS 或同等效力文件，含效期/簽發機關）
5. QMS/QSD 或 ISO 13485 證明（涵蓋產品範圍、有效期）
6. 產品中文標籤/說明書/外盒標示擬稿
7. 產品技術檔案摘要（結構、材料、規格、製程簡述、圖樣）
8. 滅菌確效摘要（如 EO：ISO 11135；需說明 SAL、循環、放行方式）
9. 包裝完整性/運輸模擬摘要（如適用）
10. 生物相容性摘要（接觸性質/時間與 ISO 10993 對應）
11. 性能/功能測試摘要（例如針尖穿刺力、滑動性、漏液、尺寸）
12. 風險管理摘要（ISO 14971，至少包含主要危害與控制）
13. 上市後監督/抱怨處理機制簡述（可用摘要）

---

## 2. 申請書關鍵欄位檢核
- 案件基本資料：案件類型、案件種類、產地、產品等級、有無類似品、是否勾選替代條款
- 名稱一致性：中文品名/英文品名與標籤、授權書、CFS 是否一致
- 分類分級：主類別/品項代碼/品項名稱是否填寫且合理
- 醫療器材商資料：統編、地址、負責人、聯絡資訊是否完整
- 製造廠資訊：製造廠名稱/地址/國別是否完整，與 QMS/CFS 是否一致

---

## 3. 文件一致性與效期檢核
- 原廠授權範圍：是否涵蓋所有規格（容量、針規）
- CFS 文件：是否明確涵蓋產品、是否仍在效期
- QMS/ISO 13485：範圍是否包含「設計/製造」及本產品類別；是否有效
- 標示擬稿：是否包含「無菌、單次使用、滅菌方式、效期、批號、警語」

---

## 4. 常見缺失（請在報告中明確列為缺漏/補件）
- 缺原廠授權或授權未涵蓋全部規格
- 缺 CFS 或文件過期/無簽發機關資訊
- 缺滅菌確效摘要或未說明 EO 參數/放行條件
- 缺生物相容性對應（ISO 10993 測項與接觸分類不匹配）
- 中文標示未含批號/效期/保存條件/一次性使用警語

---

## 5. 建議輸出格式（提供給審查代理）
- 表格：文件項目｜預期應附？｜申請書是否提及？｜是否檢附？｜判定｜備註/補件
- 條列：關鍵欄位缺失、文件一致性疑慮、下一步建議（必補/建議補充）
<!-- END_SECTION -->

<!-- BEGIN_SECTION: tw_guidance_suture | TITLE: Guidance: Suture (Import, Class II) -->
# 預審/形式審查指引（Mock）—第二等級輸入一般醫材：外科縫合針線組
## 0. 審查目的
本指引用於形式審查（預審）階段，確認申請書關鍵欄位、必備附件、文件一致性與追溯性是否足以進入技術審查。
---
## 1. 必要文件清單（預期應附）
1. 第二、三等級醫療器材查驗登記申請書
2. 醫療器材商許可執照（名稱/地址/負責人需一致）
3. 原廠授權登記書（輸入案通常必附；需涵蓋所有規格與型號）
4. 出產國製售證明（CFS 或同等效力文件；需可辨識產品並在效期內）
5. QMS/QSD 或 ISO 13485 證明（涵蓋設計/製造與產品範圍；在效期內）
6. 中文標籤/中文說明書/外盒標示擬稿（需含：無菌、單次使用、滅菌方式、效期、批號）
7. 技術檔案摘要（產品規格、材料、結構、製程簡述、關鍵圖樣）
8. 滅菌確效摘要（EO：ISO 11135；需說明 SAL、循環、放行方式）
9. 包裝完整性/運輸模擬摘要（如適用）
10. 生物相容性摘要（ISO 10993；需符合接觸分類）
11. 性能測試摘要（拉伸強度、結節強度、針-線連接強度等）
12. 風險管理摘要（ISO 14971）
---
## 3. 文件一致性與效期檢核
- 授權範圍是否涵蓋含針/無針版本、所有 USP 線徑與針型
- CFS 是否仍在效期且可辨識產品
- ISO 13485 範圍是否包含設計/製造
---
## 4. 常見缺失
- 授權書未涵蓋全部規格/針型
- 滅菌確效摘要未說明 SAL 或放行方式
- 性能測試缺少結節強度/針-線連接強度
---
## 5. 建議輸出格式
- 表格：文件項目｜預期應附？｜是否提及？｜是否檢附？｜判定｜備註
<!-- END_SECTION -->

<!-- BEGIN_SECTION: k510_guidance_review_memo | TITLE: Guidance: Internal 510(k) Review Memo (Mock) -->
# 510(k) Review Pipeline Guidance (Mock) — Internal Use

## 0. Purpose
Use this guidance to structure checklist-based review notes and draft an internal review memo grounded in the submission.

## 1. Required Inputs
- Structured submission (organized markdown)
- Checklist items (section/item/expected)
- Any device-specific guidance (optional)

## 2. Consistency Checks
- Indications consistent across cover letter, IFU, and summary
- Predicate comparison matches technological characteristics and testing
- Testing supports claims and intended use

## 3. Common Gaps
- Missing software documentation level-of-concern alignment
- Labeling inconsistent with risk controls
- Incomplete biocompatibility rationale for contacting components

## 4. Output Format
- Executive summary (5–10 bullets)
- Checklist assessment table (section → item → finding → evidence → gap → recommendation)
- Conclusion and next steps
<!-- END_SECTION -->
```

### 3. `app.py` (Refactored)
This version of `app.py` has no hardcoded case/guidance data. It loads them at startup and includes a "Reload Default Datasets" button in the Sidebar.

```python
import os
import json
import base64
import random
import re
import difflib
from datetime import datetime, date
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

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
except ImportError:
    pytesseract = None
    convert_from_bytes = None
    Image = None

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
    # Anthropic
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
        "reload_defaults": "Reload Default Datasets",
        "reload_success": "Datasets and Guidance reloaded successfully.",
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
        "reload_defaults": "重新載入預設資料集",
        "reload_success": "預設資料集與指引已重新載入。",
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

STYLE_TOKENS: Dict[str, Dict[str, str]] = {
    "Van Gogh": {"--bg1": "#0b1020", "--bg2": "#1f3b73", "--accent": "#f7c948", "--accent2": "#60a5fa", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    "Picasso": {"--bg1": "#2b2b2b", "--bg2": "#7c2d12", "--accent": "#f59e0b", "--accent2": "#a3e635", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    "Monet": {"--bg1": "#a1c4fd", "--bg2": "#c2e9fb", "--accent": "#2563eb", "--accent2": "#0ea5e9", "--card": "rgba(255,255,255,0.35)", "--border": "rgba(255,255,255,0.45)"},
    "Da Vinci": {"--bg1": "#f6f0d9", "--bg2": "#cbb38b", "--accent": "#7c2d12", "--accent2": "#1f2937", "--card": "rgba(255,255,255,0.35)", "--border": "rgba(17,24,39,0.18)"},
    "Dali": {"--bg1": "#0f172a", "--bg2": "#b91c1c", "--accent": "#fbbf24", "--accent2": "#38bdf8", "--card": "rgba(255,255,255,0.12)", "--border": "rgba(255,255,255,0.22)"},
    "Mondrian": {"--bg1": "#f8fafc", "--bg2": "#e2e8f0", "--accent": "#ef4444", "--accent2": "#2563eb", "--card": "rgba(255,255,255,0.60)", "--border": "rgba(0,0,0,0.18)"},
    "Warhol": {"--bg1": "#0b1020", "--bg2": "#6d28d9", "--accent": "#22c55e", "--accent2": "#f472b6", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    "Rembrandt": {"--bg1": "#07050a", "--bg2": "#2c1810", "--accent": "#f59e0b", "--accent2": "#fbbf24", "--card": "rgba(255,255,255,0.08)", "--border": "rgba(245,158,11,0.20)"},
    "Klimt": {"--bg1": "#0b1020", "--bg2": "#3b2f0b", "--accent": "#fbbf24", "--accent2": "#fde68a", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(251,191,36,0.25)"},
    "Hokusai": {"--bg1": "#061a2b", "--bg2": "#1e3a8a", "--accent": "#60a5fa", "--accent2": "#93c5fd", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(147,197,253,0.25)"},
    "Munch": {"--bg1": "#1f2937", "--bg2": "#7f1d1d", "--accent": "#fb7185", "--accent2": "#fde047", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    "O'Keeffe": {"--bg1": "#fff7ed", "--bg2": "#fecdd3", "--accent": "#db2777", "--accent2": "#f97316", "--card": "rgba(255,255,255,0.55)", "--border": "rgba(219,39,119,0.18)"},
    "Basquiat": {"--bg1": "#111827", "--bg2": "#f59e0b", "--accent": "#22c55e", "--accent2": "#60a5fa", "--card": "rgba(255,255,255,0.12)", "--border": "rgba(255,255,255,0.22)"},
    "Matisse": {"--bg1": "#ffedd5", "--bg2": "#fde68a", "--accent": "#ea580c", "--accent2": "#2563eb", "--card": "rgba(255,255,255,0.60)", "--border": "rgba(234,88,12,0.20)"},
    "Pollock": {"--bg1": "#0b1020", "--bg2": "#111827", "--accent": "#f97316", "--accent2": "#22c55e", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.20)"},
    "Kahlo": {"--bg1": "#064e3b", "--bg2": "#7f1d1d", "--accent": "#fbbf24", "--accent2": "#22c55e", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    "Hopper": {"--bg1": "#0b1020", "--bg2": "#0f766e", "--accent": "#60a5fa", "--accent2": "#fbbf24", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    "Magritte": {"--bg1": "#0b1020", "--bg2": "#1d4ed8", "--accent": "#e2e8f0", "--accent2": "#fbbf24", "--card": "rgba(255,255,255,0.10)", "--border": "rgba(255,255,255,0.22)"},
    "Cyberpunk": {"--bg1": "#050816", "--bg2": "#1b0033", "--accent": "#22d3ee", "--accent2": "#a78bfa", "--card": "rgba(255,255,255,0.08)", "--border": "rgba(34,211,238,0.25)"},
    "Bauhaus": {"--bg1": "#f8fafc", "--bg2": "#e2e8f0", "--accent": "#111827", "--accent2": "#ef4444", "--card": "rgba(255,255,255,0.70)", "--border": "rgba(17,24,39,0.15)"},
}


def apply_style_engine(theme_mode: str, painter_style: str):
    tokens = STYLE_TOKENS.get(painter_style, STYLE_TOKENS["Van Gogh"])
    is_dark = theme_mode.lower() == "dark"
    text_color = "#e5e7eb" if is_dark else "#0f172a"
    subtext = "#cbd5e1" if is_dark else "#334155"
    shadow = "0 18px 50px rgba(0,0,0,0.38)" if is_dark else "0 18px 50px rgba(2,6,23,0.18)"
    glass = "rgba(17,24,39,0.38)" if is_dark else "rgba(255,255,255,0.55)"

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

    body {{
        color: var(--text);
        background: radial-gradient(1200px circle at 12% 8%, var(--bg2) 0%, transparent 55%),
                    radial-gradient(900px circle at 88% 18%, var(--accent2) 0%, transparent 50%),
                    linear-gradient(135deg, var(--bg1), var(--bg2));
        background-attachment: fixed;
    }}
    {splatter}

    .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 3.5rem;
    }}

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

    .wow-card {{
        border-radius: var(--radius);
        padding: 14px 16px;
        background: var(--glass);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
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

    .stButton > button {{
        border-radius: 999px !important;
        border: 1px solid var(--border) !important;
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: {"#0b1020"} !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 14px 35px rgba(0,0,0,0.25) !important;
    }}
    .stButton > button:hover {{
        filter: brightness(1.04);
        transform: translateY(-1px);
        transition: 120ms ease;
    }}

    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        background: rgba(255,255,255,{0.06 if is_dark else 0.55}) !important;
        color: var(--text) !important;
    }}

    button[role="tab"] {{
        border-radius: 999px !important;
    }}

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

    .coral {{
        color: var(--coral);
        font-weight: 800;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# 4) Data Loading Logic (Datasets & Guidance)
# ============================================================
def load_default_datasets_from_file() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads defaultdataset.json containing tw_cases and k510_checklists."""
    tw_cases = {}
    k510_checklists = {}
    try:
        with open("defaultdataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            tw_cases = data.get("tw_cases", {})
            k510_checklists = data.get("k510_checklists", {})
    except FileNotFoundError:
        st.error("Default dataset file 'defaultdataset.json' not found.")
    except Exception as e:
        st.error(f"Error loading defaultdataset.json: {e}")
    return tw_cases, k510_checklists

def load_default_guidance_from_file() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Loads defaultguide.md and parses sections for TW and 510k guidances."""
    tw_guides = {}
    k510_guides = {}
    
    try:
        with open("defaultguide.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Regex to find sections: <!-- BEGIN_SECTION: id | TITLE: title --> ... <!-- END_SECTION -->
        pattern = re.compile(
            r"<!--\s*BEGIN_SECTION:\s*(.*?)\s*\|\s*TITLE:\s*(.*?)\s*-->(.*?)<!--\s*END_SECTION\s*-->", 
            re.DOTALL
        )
        matches = pattern.findall(content)
        
        for key, title, body in matches:
            key = key.strip()
            title = title.strip()
            body = body.strip()
            entry = {"title": title, "md": body}
            
            if key.startswith("tw_"):
                tw_guides[key] = entry
            elif key.startswith("k510_"):
                k510_guides[key] = entry
                
    except FileNotFoundError:
        st.error("Default guidance file 'defaultguide.md' not found.")
    except Exception as e:
        st.error(f"Error loading defaultguide.md: {e}")
        
    return tw_guides, k510_guides

def refresh_defaults():
    tw_c, k510_c = load_default_datasets_from_file()
    tw_g, k510_g = load_default_guidance_from_file()
    st.session_state["DEFAULT_TW_CASESETS"] = tw_c
    st.session_state["DEFAULT_510K_CHECKLIST_SETS"] = k510_c
    st.session_state["DEFAULT_TW_GUIDANCES"] = tw_g
    st.session_state["DEFAULT_510K_GUIDANCES"] = k510_g
    
    # Also log/toast to user
    # st.toast(t("reload_success")) # Standard streamit toast if version allows, or just pass


# ============================================================
# 5) State init
# ============================================================
if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "theme": "Dark",
        "language": "zh-tw",
        "painter_style": "Van Gogh",
        "model": "gpt-4o-mini",
        "max_tokens": 12000,
        "temperature": 0.2,
        "token_budget_est": 250_000,
    }

if "history" not in st.session_state:
    st.session_state["history"] = []

if "api_keys" not in st.session_state:
    st.session_state["api_keys"] = {"openai": "", "gemini": "", "anthropic": "", "grok": ""}

if "workflow" not in st.session_state:
    st.session_state["workflow"] = {
        "steps": [],
        "cursor": 0,
        "input": "",
        "outputs": [],
        "statuses": [],
    }

# New state for TW datasets/guidance/templates/mapping
if "tw_cases_dataset" not in st.session_state:
    st.session_state["tw_cases_dataset"] = []
if "tw_active_case_index" not in st.session_state:
    st.session_state["tw_active_case_index"] = 0
if "tw_guidance_effective_md" not in st.session_state:
    st.session_state["tw_guidance_effective_md"] = ""
if "tw_guidance_struct" not in st.session_state:
    st.session_state["tw_guidance_struct"] = {}
if "company_templates" not in st.session_state:
    st.session_state["company_templates"] = []  # list of dicts {name, ...firm/contact fields...}

# Rule-mapping dictionary (editable)
if "tw_field_mapping" not in st.session_state:
    st.session_state["tw_field_mapping"] = {
        # Core IDs
        "公文文號": "doc_no",
        "docNo": "doc_no",
        "DocNo": "doc_no",
        "電子流水號": "e_no",
        "eNo": "e_no",
        "E_NO": "e_no",
        "申請日": "apply_date",
        "applyDate": "apply_date",
        "案件類型": "case_type",
        "醫療器材類型": "device_category",
        "案件種類": "case_kind",
        "產地": "origin",
        "產品等級": "product_class",
        "有無類似品": "similar",
        "替代條款": "replace_flag",
        "前次申請案號": "prior_app_no",
        # Names
        "中文名稱": "name_zh",
        "品名(中)": "name_zh",
        "英文名稱": "name_en",
        "品名(英)": "name_en",
        "適應症": "indications",
        "用途": "indications",
        "規格": "spec_comp",
        "型號規格": "spec_comp",
        # Classification
        "主類別": "main_cat",
        "品項代碼": "item_code",
        "品項名稱": "item_name",
        # Firm fields
        "統一編號": "uniform_id",
        "公司統編": "uniform_id",
        "醫療器材商名稱": "firm_name",
        "公司名稱": "firm_name",
        "醫療器材商地址": "firm_addr",
        "公司地址": "firm_addr",
        "負責人": "resp_name",
        "負責人姓名": "resp_name",
        "聯絡人": "contact_name",
        "聯絡人姓名": "contact_name",
        "電話": "contact_tel",
        "聯絡電話": "contact_tel",
        "傳真": "contact_fax",
        "電子郵件": "contact_email",
        "email": "contact_email",
        # Manufacturer
        "製造方式": "manu_type",
        "製造廠名稱": "manu_name",
        "製造國別": "manu_country",
        "製造廠地址": "manu_addr",
        "製造說明": "manu_note",
        # Attachments summaries
        "原廠授權": "auth_applicable",
        "授權": "auth_applicable",
        "授權說明": "auth_desc",
        "製售證明": "cfs_applicable",
        "CFS": "cfs_applicable",
        "製售證明說明": "cfs_desc",
        "QMS": "qms_applicable",
        "ISO13485": "qms_applicable",
        "QMS說明": "qms_desc",
        "類似品摘要": "similar_info",
        "標示摘要": "labeling_info",
        "技術檔案摘要": "tech_file_info",
        "臨床前摘要": "preclinical_info",
        "替代資料說明": "preclinical_replace",
        "臨床證據適用性": "clinical_just",
        "臨床證據摘要": "clinical_info",
        # Booleans
        "已確認證照相符": "confirm_match",
        "RAPS": "cert_raps",
        "AHWP": "cert_ahwp",
    }

# New state for 510k checklist/guidance manager
if "k510_checklist_dataset" not in st.session_state:
    st.session_state["k510_checklist_dataset"] = []  # list of dicts
if "k510_guidance_effective_md" not in st.session_state:
    st.session_state["k510_guidance_effective_md"] = ""

# ** Initialize Defaults from External Files **
if "DEFAULT_TW_CASESETS" not in st.session_state:
    refresh_defaults()


# ============================================================
# 5) API key logic
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
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
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
# 7) Generic helpers
# ============================================================
def est_tokens(text: str) -> int:
    # Simple estimate: 4 chars per token
    return max(1, int(len(text or "") / 4))


def log_event(tab: str, agent: str, model: str, tokens_est: int, meta: Optional[dict] = None):
    st.session_state["history"].append(
        {"tab": tab, "agent": agent, "model": model, "tokens_est": int(tokens_est), "ts": datetime.utcnow().isoformat(), "meta": meta or {}}
    )


def extract_pdf_pages_to_text(file, start_page: int, end_page: int, use_ocr: bool = False) -> str:
    pdf_text = ""
    try:
        reader = PdfReader(file)
        n = len(reader.pages)
        start = max(0, start_page - 1)
        end = min(n, end_page)
        raw_texts = []
        for i in range(start, end):
            try:
                page_text = reader.pages[i].extract_text() or ""
                raw_texts.append(page_text)
            except Exception:
                raw_texts.append("")
        pdf_text = "\n\n".join(raw_texts).strip()
    except Exception as e:
        print(f"pypdf extraction error: {e}")

    if use_ocr:
        if pytesseract is None or convert_from_bytes is None:
            return pdf_text + "\n\n[System: OCR requested but libraries (pytesseract/pdf2image) are missing.]"
        ocr_text = []
        file.seek(0)
        try:
            images = convert_from_bytes(file.read(), first_page=start_page, last_page=end_page)
            for img in images:
                text = pytesseract.image_to_string(img, lang="eng+chi_tra")
                ocr_text.append(text)
            return "\n\n".join(ocr_text).strip()
        except Exception as e:
            return pdf_text + f"\n\n[System: OCR failed: {e}]"

    return pdf_text


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
    st.markdown(f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}"></iframe>""", unsafe_allow_html=True)


def status_row(label: str, status: str):
    color_class = {
        "pending": "dot-amber",
        "running": "dot-amber",
        "done": "dot-green",
        "error": "dot-red",
        "idle": "dot-amber",
        "thinking": "dot-amber",
        "active": "dot-green",
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
# 8) Guidance struct conversion + merge/diff utilities
# ============================================================
def normalize_md(md: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (md or "").strip())


def _find_section(md: str, patterns: List[str]) -> str:
    """
    Extract section content for the first matched heading pattern.
    patterns: list of regex patterns to match headings like '## 0. 審查目的'
    """
    text = normalize_md(md)
    lines = text.splitlines()
    idx = None
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.search(pat, line.strip()):
                idx = i
                break
        if idx is not None:
            break
    if idx is None:
        return ""

    # capture until next heading starting with ##
    out = []
    for j in range(idx + 1, len(lines)):
        if lines[j].strip().startswith("## "):
            break
        out.append(lines[j])
    return normalize_md("\n".join(out)).strip()


def _extract_list_items(section_text: str) -> List[str]:
    items = []
    for line in (section_text or "").splitlines():
        s = line.strip()
        # numbered list "1." or "- "
        if re.match(r"^\d+\.", s):
            items.append(re.sub(r"^\d+\.\s*", "", s).strip())
        elif s.startswith("- "):
            items.append(s[2:].strip())
    return [x for x in items if x]


def guidance_markdown_to_struct(md: str) -> Dict[str, Any]:
    """
    Deterministic conversion of guidance markdown to structured fields.
    Output fields:
      - purpose (str)
      - required_documents (list[str])
      - consistency_checks (list[str])   (includes key-field checks + consistency)
      - common_defects (list[str])
      - output_format (str)
      - raw_markdown (str)
    """
    md = normalize_md(md)

    purpose = _find_section(md, [r"^##\s*0\.", r"審查目的"])
    req = _find_section(md, [r"必要文件清單", r"^##\s*1\."])
    checks = _find_section(md, [r"關鍵欄位檢核", r"一致性", r"^##\s*2\.", r"^##\s*3\."])
    defects = _find_section(md, [r"常見缺失", r"^##\s*4\."])
    outfmt = _find_section(md, [r"建議輸出格式", r"^##\s*5\."])

    required_documents = _extract_list_items(req)
    # checks may include bullet list
    consistency_checks = _extract_list_items(checks)
    common_defects = _extract_list_items(defects)

    # If out format is bullet list, keep as text too
    output_format = outfmt.strip()

    return {
        "purpose": purpose,
        "required_documents": required_documents,
        "consistency_checks": consistency_checks,
        "common_defects": common_defects,
        "output_format": output_format,
        "raw_markdown": md,
    }


def guidance_struct_to_one_row_csv(struct: Dict[str, Any]) -> bytes:
    row = {
        "purpose": struct.get("purpose", ""),
        "required_documents": " | ".join(struct.get("required_documents", []) or []),
        "consistency_checks": " | ".join(struct.get("consistency_checks", []) or []),
        "common_defects": " | ".join(struct.get("common_defects", []) or []),
        "output_format": struct.get("output_format", ""),
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def guidance_required_docs_csv(struct: Dict[str, Any]) -> bytes:
    docs = struct.get("required_documents", []) or []
    df = pd.DataFrame([{"doc_item": d, "required": True} for d in docs])
    return df.to_csv(index=False).encode("utf-8")


def merge_guidance_markdowns(mds: List[str], extra_rules_md: str = "") -> str:
    parts = [normalize_md(x) for x in (mds or []) if normalize_md(x)]
    if extra_rules_md and extra_rules_md.strip():
        parts.append("## 自訂追加規則\n" + normalize_md(extra_rules_md))
    return normalize_md("\n\n---\n\n".join(parts))


def diff_markdown(a: str, b: str) -> str:
    a_lines = (a or "").splitlines(keepends=True)
    b_lines = (b or "").splitlines(keepends=True)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="A.md", tofile="B.md")
    return "".join(diff).strip()


# ============================================================
# 9) TW dataset standardization (rule-mapping prioritized, skip failed rows)
# ============================================================
BOOL_FIELDS = {"confirm_match", "cert_raps", "cert_ahwp"}
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

def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "有", "勾選", "checked"}:
        return True
    return False


def _normalize_apply_date(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    if not s:
        return ""
    # Accept YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    m = re.match(r"^\s*(\d{4})[\/\.-](\d{1,2})[\/\.-](\d{1,2})\s*$", s)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        try:
            return date(int(y), mo, d).strftime("%Y-%m-%d")
        except Exception:
            return ""
    # If already YYYY-MM-DD but with extra text, best effort:
    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m2:
        try:
            return date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3))).strftime("%Y-%m-%d")
        except Exception:
            return ""
    return ""


def standardize_tw_record_rule_mapping(raw: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Rule-mapping only (dictionary alias -> standard field). Extra keys ignored.
    """
    out: Dict[str, Any] = {}
    # Map keys
    for k, v in (raw or {}).items():
        if k in TW_APP_FIELDS:
            out[k] = v
        elif k in mapping:
            out[mapping[k]] = v
        else:
            # try case-insensitive mapping for common variants
            kk = str(k).strip()
            if kk in mapping:
                out[mapping[kk]] = v

    # Ensure all fields exist
    for f in TW_APP_FIELDS:
        if f not in out:
            out[f] = False if f in BOOL_FIELDS else ""

    # Normalize booleans
    for bf in BOOL_FIELDS:
        out[bf] = _to_bool(out.get(bf, False))

    # Normalize apply_date
    out["apply_date"] = _normalize_apply_date(out.get("apply_date"))

    # Stringify non-string (except booleans)
    for f in TW_APP_FIELDS:
        if f in BOOL_FIELDS:
            continue
        v = out.get(f)
        if v is None:
            out[f] = ""
        elif isinstance(v, (dict, list)):
            out[f] = json.dumps(v, ensure_ascii=False)
        else:
            out[f] = str(v)

    return out


def _validate_tw_record(std: Dict[str, Any]) -> List[str]:
    """
    Return missing/weak fields list (for reporting). This is not strict validation; used for failure heuristics.
    """
    missing = []
    critical = ["e_no", "case_type", "device_category", "origin", "product_class", "name_zh", "firm_name", "firm_addr", "contact_name", "contact_tel", "contact_email", "manu_name", "manu_addr"]
    for k in critical:
        v = std.get(k, "")
        if isinstance(v, str) and not v.strip():
            missing.append(k)
    return missing


def standardize_tw_dataset_records(records: List[Dict[str, Any]], mapping: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Standardize a list of records. Skip "failed" rows and return:
      (success_records, failures)
    failures: list of {row_index, reason, missing_fields, raw_keys}
    """
    ok = []
    failures = []
    for i, rec in enumerate(records or []):
        try:
            std = standardize_tw_record_rule_mapping(rec, mapping)
            missing = _validate_tw_record(std)

            # Failure heuristic: if too many criticals missing AND the record had data
            # (prevents skipping legitimate partially-filled drafts too aggressively)
            nonempty_raw = sum(1 for v in (rec or {}).values() if str(v).strip())
            if nonempty_raw >= 3 and len(missing) >= 7:
                failures.append(
                    {
                        "row_index": i,
                        "reason": "Too many critical fields missing after rule mapping; skipped.",
                        "missing_fields": missing,
                        "raw_keys": list((rec or {}).keys()),
                    }
                )
                continue

            ok.append(std)
        except Exception as e:
            failures.append(
                {
                    "row_index": i,
                    "reason": f"Exception during standardization: {e}",
                    "missing_fields": [],
                    "raw_keys": list((rec or {}).keys()) if isinstance(rec, dict) else [],
                }
            )
    return ok, failures


def parse_uploaded_cases_file(file) -> List[Dict[str, Any]]:
    """
    Accept JSON (object or list) or CSV (rows). Return list of dict records.
    """
    name = (file.name or "").lower()
    if name.endswith(".json"):
        obj = json.load(file)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []
    if name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.fillna("").to_dict(orient="records")
    raise ValueError("Unsupported file type. Please upload JSON or CSV.")


# ============================================================
# 10) Agents YAML loading (+ standardization as in original)
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


def standardize_agents_yaml(raw_yaml_text: str) -> Dict[str, Any]:
    model = st.session_state.settings["model"]
    api_keys = st.session_state.get("api_keys", {})
    system_prompt = """
You are a configuration Standardization Agent.
Convert the user's uploaded agent configuration (which might be in any format) into the STANDARD format used by this system.

STANDARD FORMAT (YAML):
agents:
  unique_agent_id_snake_case:
    name: "Human Readable Name"
    description: "Short description"
    category: "Category Name"
    model: "gpt-4o-mini"
    temperature: 0.2
    max_tokens: 12000
    system_prompt: |
      The system prompt text...
    user_prompt_template: |
      Optional template...

RULES:
1. Extract as many agents as possible.
2. Map fields as best as you can.
3. Ensure valid YAML output.
4. Output ONLY the YAML, no markdown code blocks.
"""
    try:
        out = call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=f"Raw Content:\n{raw_yaml_text}",
            max_tokens=8000,
            temperature=0.0,
            api_keys=api_keys
        )
        clean_out = out.replace("```yaml", "").replace("```", "").strip()
        data = yaml.safe_load(clean_out)
        return data
    except Exception as e:
        print(f"Standardization error: {e}")
        return {}


# ============================================================
# 11) WOW header + sidebar
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
        
        # New Reload Button
        if st.button(t("reload_defaults")):
            refresh_defaults()
            st.success(t("reload_success"))
            st.rerun()

        theme_choice = st.radio(
            t("theme"),
            [t("light"), t("dark")],
            index=0 if st.session_state.settings["theme"] == "Light" else 1,
            horizontal=True,
        )
        st.session_state.settings["theme"] = "Light" if theme_choice == t("light") else "Dark"

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
                if st.session_state.settings["painter_style"] in PAINTER_STYLES_20 else 0,
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
            if st.session_state.settings["model"] in ALL_MODELS else 0,
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
            0.0, 1.0, float(st.session_state.settings["temperature"]), 0.05
        )

        st.markdown("---")
        st.markdown(f"## {t('api_keys')}")
        keys = dict(st.session_state["api_keys"])

        if env_key_present("OPENAI_API_KEY"):
            st.caption(f"OpenAI: {t('active_env')}")
        else:
            keys["openai"] = st.text_input("OpenAI API Key", type="password", value=keys.get("openai", ""))

        if env_key_present("GEMINI_API_KEY"):
            st.caption(f"Gemini: {t('active_env')}")
        else:
            keys["gemini"] = st.text_input("Gemini API Key", type="password", value=keys.get("gemini", ""))

        if env_key_present("ANTHROPIC_API_KEY"):
            st.caption(f"Anthropic: {t('active_env')}")
        else:
            keys["anthropic"] = st.text_input("Anthropic API Key", type="password", value=keys.get("anthropic", ""))

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
                raw_content = uploaded_agents.read().decode("utf-8", errors="ignore")
                cfg = yaml.safe_load(raw_content) or {}
                if "agents" in cfg and isinstance(cfg["agents"], dict) and len(cfg["agents"]) > 0:
                    st.session_state["agents_cfg"] = ensure_fallback_agents(cfg)
                    st.success("Loaded valid agents.yaml.")
                    st.rerun()
                else:
                    st.info("Uploaded YAML does not match standard schema. Attempting to standardize with AI...")
                    std_cfg = standardize_agents_yaml(raw_content)
                    if std_cfg and "agents" in std_cfg:
                        st.session_state["agents_cfg"] = ensure_fallback_agents(std_cfg)
                        st.success("Standardized and loaded agent configuration!")
                        st.rerun()
                    else:
                        st.error("Could not standardize the YAML file.")
            except Exception as e:
                st.error(f"Failed to process YAML: {e}")


# ============================================================
# 12) Dashboard
# ============================================================
def render_dashboard():
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
        st.download_button(t("export_history"), data=csv_bytes, file_name="antigravity_history.csv", mime="text/csv")


# ============================================================
# 13) Agent runner UI (kept)
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

    base_model = agent_cfg.get("model", st.session_state.settings["model"])
    base_max_tokens = int(agent_cfg.get("max_tokens", st.session_state.settings["max_tokens"]))
    system_prompt = agent_cfg.get("system_prompt", "")

    supported = agent_cfg.get("supported_models", None)
    model_choices = ALL_MODELS
    if isinstance(supported, list) and supported:
        model_choices = [m for m in ALL_MODELS if m in supported] or ALL_MODELS

    status_key = f"{tab_key}_status"
    if status_key not in st.session_state:
        st.session_state[status_key] = "idle"

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
        model = st.selectbox(t("model"), model_choices, index=model_index, disabled=not allow_model_override, key=f"{tab_key}_model")
    with c3:
        max_tokens = st.number_input(
            "max_tokens",
            min_value=1000,
            max_value=120000,
            value=int(st.session_state.get(f"{tab_key}_max_tokens", base_max_tokens)),
            step=1000,
            key=f"{tab_key}_max_tokens",
        )

    input_text = st.text_area(t("input_text"), value=st.session_state.get(f"{tab_key}_input", default_input_text), height=240, key=f"{tab_key}_input")
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
                st.session_state[f"{tab_key}_output_edited"] = out
                st.rerun()  # 建議
                token_est = est_tokens(user_full + out)
                log_event(tab_label_for_history or tab_key, agent_name, model, token_est, meta={"agent_id": agent_id})
            except Exception as e:
                st.session_state[status_key] = "error"
                st.error(f"Agent error: {e}")

    output = st.session_state.get(f"{tab_key}_output", "")
    view_mode = st.radio(t("view_mode"), [t("markdown"), t("plain_text")], horizontal=True, key=f"{tab_key}_viewmode")
    edited = st.text_area(
        f"{t('output')} ({'Markdown' if view_mode == t('markdown') else 'Text'}, editable)",
        value=output,
        height=280,
        key=f"{tab_key}_output_edited",
    )
    st.session_state[f"{tab_key}_output_edited_value"] = edited


# ============================================================
# 14) Workflow Studio (kept)
# ============================================================
def workflow_default_steps() -> List[Dict[str, Any]]:
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
    wf = st.session_state["workflow"]

    if not wf["steps"]:
        wf["steps"] = workflow_default_steps()
        wf["outputs"] = [""] * len(wf["steps"])
        wf["statuses"] = ["idle"] * len(wf["steps"])
        wf["cursor"] = 0

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
                {"agent_id": "note_organizer", "name": "Note Organizer", "model": st.session_state.settings["model"], "max_tokens": st.session_state.settings["max_tokens"], "prompt": "Organize into structured Markdown. Do not add new facts."}
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
        wf["cursor"] = st.number_input("Active step index", min_value=0, max_value=max(0, len(wf["steps"]) - 1), value=int(wf["cursor"]), step=1)

    st.markdown("---")
    st.markdown(f"### {t('workflow_input')}")
    wf["input"] = st.text_area(t("input_text"), value=wf.get("input", ""), height=200, key="wf_input_text")

    st.markdown("---")
    st.markdown(f"### {t('step')}s")
    for idx, step in enumerate(wf["steps"]):
        agent_id = step.get("agent_id", "")
        agent_cfg = agents_dict.get(agent_id, {})
        agent_name = step.get("name") or agent_cfg.get("name") or agent_id

        with st.expander(f"{t('step')} {idx+1}: {agent_name}  ·  ({agent_id})", expanded=(idx == wf["cursor"])):
            wf["statuses"][idx] = wf["statuses"][idx] if idx < len(wf["statuses"]) else "idle"
            status_row(f"{agent_name}", wf["statuses"][idx])

            supported = agent_cfg.get("supported_models", None)
            model_choices = ALL_MODELS
            if isinstance(supported, list) and supported:
                model_choices = [m for m in ALL_MODELS if m in supported] or ALL_MODELS

            cA, cB = st.columns([1.2, 1.2])
            with cA:
                step["agent_id"] = st.selectbox("agent_id", sorted(list(agents_dict.keys())), index=sorted(list(agents_dict.keys())).index(agent_id) if agent_id in agents_dict else 0, key=f"wf_agent_{idx}")
            with cB:
                agent_id = step["agent_id"]
                agent_cfg = agents_dict.get(agent_id, {})
                supported = agent_cfg.get("supported_models", None)
                model_choices = ALL_MODELS
                if isinstance(supported, list) and supported:
                    model_choices = [m for m in ALL_MODELS if m in supported] or ALL_MODELS
                step["model"] = st.selectbox(t("model"), model_choices, index=model_choices.index(step.get("model")) if step.get("model") in model_choices else 0, key=f"wf_model_{idx}")

            cC, cD = st.columns([1.2, 1.2])
            with cC:
                step["max_tokens"] = st.number_input("max_tokens", min_value=1000, max_value=120000, value=int(step.get("max_tokens", st.session_state.settings["max_tokens"])), step=1000, key=f"wf_mt_{idx}")
            with cD:
                step["name"] = st.text_input("Display name", value=str(step.get("name") or agent_cfg.get("name") or agent_id), key=f"wf_name_{idx}")

            step["prompt"] = st.text_area(t("prompt"), value=step.get("prompt", ""), height=150, key=f"wf_prompt_{idx}")

            if idx == 0:
                step_input_default = wf.get("input", "")
            else:
                step_input_default = wf["outputs"][idx - 1] or ""

            step_input = st.text_area(f"{t('input_text')} (Step {idx+1})", value=step_input_default, height=180, key=f"wf_input_{idx}")

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
                    log_event("Workflow Studio", step.get("name", agent_id), step["model"], est_tokens(user_full + out), meta={"agent_id": agent_id, "workflow_step": idx + 1})
                    if run_next and idx < len(wf["steps"]) - 1:
                        wf["cursor"] = idx + 1
                        st.rerun()
                except Exception as e:
                    wf["statuses"][idx] = "error"
                    st.error(f"Workflow step error: {e}")

            st.markdown(f"**{t('workflow_output')} (editable; becomes input to next step)**")
            view = st.radio(t("view_mode"), [t("markdown"), t("plain_text")], horizontal=True, key=f"wf_view_{idx}")
            wf["outputs"][idx] = st.text_area(f"Output (Step {idx+1})", value=wf["outputs"][idx] or "", height=240, key=f"wf_out_{idx}")

    st.markdown("---")
    final_out = wf["outputs"][-1] if wf["outputs"] else ""
    if final_out.strip():
        st.download_button(t("download_md"), data=final_out.encode("utf-8"), file_name="workflow_output.md", mime="text/markdown")


# ============================================================
# 15) TW Premarket helpers: session <-> dict
# ============================================================
def build_tw_app_dict_from_session() -> dict:
    s = st.session_state
    apply_date_val = s.get("tw_apply_date")
    apply_date_str = apply_date_val.strftime("%Y-%m-%d") if isinstance(apply_date_val, (datetime, date)) else ""
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


def compute_tw_missing_items_report() -> Dict[str, Any]:
    s = st.session_state
    missing_required = []
    required_keys = [
        ("電子流水號", "tw_e_no"),
        ("案件類型", "tw_case_type"),
        ("醫療器材類型", "tw_device_category"),
        ("產地", "tw_origin"),
        ("產品等級", "tw_product_class"),
        ("中文名稱", "tw_dev_name_zh"),
        ("英文名稱", "tw_dev_name_en"),
        ("統一編號", "tw_uniform_id"),
        ("醫療器材商名稱", "tw_firm_name"),
        ("醫療器材商地址", "tw_firm_addr"),
        ("負責人姓名", "tw_resp_name"),
        ("聯絡人姓名", "tw_contact_name"),
        ("電話", "tw_contact_tel"),
        ("電子郵件", "tw_contact_email"),
        ("製造廠名稱", "tw_manu_name"),
        ("製造廠地址", "tw_manu_addr"),
    ]
    for label, key in required_keys:
        v = s.get(key, "")
        if isinstance(v, str) and not v.strip():
            missing_required.append(label)

    guidance_md = (s.get("tw_guidance_effective_md") or "").strip()
    guidance_notes = []
    if not guidance_md:
        guidance_notes.append("尚未提供預審/形式審查指引（建議提供）")

    # Optional attachment summary checks
    optional_missing = []
    if s.get("tw_origin") == "輸入":
        if (s.get("tw_auth_app") or "").strip() != "適用":
            optional_missing.append("輸入案通常需原廠授權（auth_applicable 建議為「適用」或說明例外）")
        if (s.get("tw_cfs_app") or "").strip() != "適用":
            optional_missing.append("輸入案通常需 CFS（cfs_applicable 建議為「適用」或說明例外）")
    if (s.get("tw_qms_app") or "").strip() != "適用":
        optional_missing.append("QMS/ISO 13485 證明建議標註適用（qms_applicable）")

    return {
        "missing_required_fields": missing_required,
        "guidance_readiness": guidance_notes,
        "optional_recommendations": optional_missing,
        "generated_at": datetime.utcnow().isoformat(),
    }


# ============================================================
# 16) Company template feature
# ============================================================
def current_company_from_session() -> Dict[str, Any]:
    app = build_tw_app_dict_from_session()
    return {k: app.get(k, "") for k in COMPANY_FIELDS}


def apply_company_to_case(case: Dict[str, Any], company: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(case)
    for k in COMPANY_FIELDS:
        if k in company:
            out[k] = company[k]
    return out


# ============================================================
# 17) TW Field Mapping Dictionary Editor (upload/download/edit)
# ============================================================
def mapping_dict_to_df(mapping: Dict[str, str]) -> pd.DataFrame:
    items = [{"alias": k, "standard_key": v} for k, v in (mapping or {}).items()]
    return pd.DataFrame(items).sort_values(["standard_key", "alias"], kind="stable")


def df_to_mapping_dict(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        a = str(row.get("alias", "")).strip()
        s = str(row.get("standard_key", "")).strip()
        if a and s:
            out[a] = s
    return out


def parse_mapping_upload(file) -> Dict[str, str]:
    name = (file.name or "").lower()
    if name.endswith(".json"):
        obj = json.load(file)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
        if isinstance(obj, list):
            # list of {alias, standard_key}
            out = {}
            for x in obj:
                if isinstance(x, dict) and x.get("alias") and x.get("standard_key"):
                    out[str(x["alias"])] = str(x["standard_key"])
            return out
        return {}
    if name.endswith(".csv"):
        df = pd.read_csv(file).fillna("")
        if "alias" in df.columns and "standard_key" in df.columns:
            return df_to_mapping_dict(df)
        # accept two-column CSV
        if len(df.columns) >= 2:
            out = {}
            for _, r in df.iterrows():
                out[str(r[df.columns[0]]).strip()] = str(r[df.columns[1]]).strip()
            return out
        return {}
    raise ValueError("Unsupported mapping file type (JSON/CSV).")


# ============================================================
# 18) TW Premarket Tab (updated with dataset + guidance manager)
# ============================================================
def render_tw_premarket_tab():
    st.markdown("## 第二、三等級醫療器材查驗登記（TW Premarket）")

    st.markdown(
        """
        <div class="wow-card">
          <div style="font-weight:900; font-size:1.05rem;">Quick Guide</div>
          <div class="wow-muted" style="margin-top:6px;">
            Step 0: 選擇/匯入 Case Dataset + Guidance → Step 1: 編輯申請書 → Step 2: 編輯指引（可合併/差異比較/結構化） → Step 3: Run screening agent → Step 4: Improve doc.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 0) Case Dataset Library / Upload / Standardize (JSON/CSV)")

    DEFAULT_TW_CASESETS = st.session_state.get("DEFAULT_TW_CASESETS", {})
    DEFAULT_TW_GUIDANCES = st.session_state.get("DEFAULT_TW_GUIDANCES", {})

    # Load default dataset or upload
    c0a, c0b = st.columns([1.2, 1.8])
    with c0a:
        ds_mode = st.radio("Cases Source", ["Default library", "Upload file"], horizontal=True, key="tw_cases_source_mode")
    with c0b:
        if ds_mode == "Default library":
            if not DEFAULT_TW_CASESETS:
                st.warning("No default datasets available. Check 'defaultdataset.json'.")
            else:
                ds_keys = list(DEFAULT_TW_CASESETS.keys())
                ds_labels = [DEFAULT_TW_CASESETS[k]["title"] for k in ds_keys]
                ds_sel = st.selectbox("Select default case dataset", ds_labels, index=0, key="tw_cases_default_select")
                
                if st.button("Load selected default dataset", key="tw_load_default_cases_btn"):
                    ds_id = ds_keys[ds_labels.index(ds_sel)]
                    st.session_state["tw_cases_dataset"] = [standardize_tw_record_rule_mapping(x, st.session_state["tw_field_mapping"]) for x in DEFAULT_TW_CASESETS[ds_id]["cases"]]
                    st.session_state["tw_active_case_index"] = 0
                    # auto-apply first case to form
                    if st.session_state["tw_cases_dataset"]:
                        apply_tw_app_dict_to_session(st.session_state["tw_cases_dataset"][0])
                    st.success("Loaded default dataset into session.")
                    st.rerun()
        else:
            up = st.file_uploader("Upload cases dataset (JSON/CSV)", type=["json", "csv"], key="tw_cases_upload_file")
            if up is not None and st.button("Import + Standardize (rule mapping)", key="tw_import_cases_btn"):
                try:
                    records = parse_uploaded_cases_file(up)
                    ok, failures = standardize_tw_dataset_records(records, st.session_state["tw_field_mapping"])
                    st.session_state["tw_cases_dataset"] = ok
                    st.session_state["tw_active_case_index"] = 0
                    st.session_state["tw_cases_failures"] = failures
                    if ok:
                        apply_tw_app_dict_to_session(ok[0])
                    if failures:
                        st.warning(f"Standardization skipped {len(failures)} row(s). See details below.")
                    st.success(f"Imported standardized cases: {len(ok)} row(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    failures = st.session_state.get("tw_cases_failures", [])
    if failures:
        with st.expander("Standardization failures (skipped rows)", expanded=False):
            st.json(failures)

    cases = st.session_state.get("tw_cases_dataset", []) or []
    if cases:
        # Active case selector
        labels = []
        for i, c in enumerate(cases):
            labels.append(f"[{i}] {c.get('e_no','(no e_no)')} · {c.get('name_zh','(no name)')}")
        st.session_state["tw_active_case_index"] = st.selectbox(
            "Active case (used to populate the form & screening input)",
            options=list(range(len(cases))),
            format_func=lambda i: labels[i],
            index=min(st.session_state.get("tw_active_case_index", 0), len(cases)-1),
            key="tw_active_case_selectbox",
        )

        c_apply, c_update, c_dl = st.columns([1.0, 1.2, 1.8])
        with c_apply:
            if st.button("Apply active case → form", key="tw_apply_active_case_btn"):
                apply_tw_app_dict_to_session(cases[st.session_state["tw_active_case_index"]])
                st.success("Applied active case to form.")
                st.rerun()
        with c_update:
            if st.button("Update active case ← form", key="tw_update_active_case_btn"):
                idx = st.session_state["tw_active_case_index"]
                cur = build_tw_app_dict_from_session()
                cur_std = standardize_tw_record_rule_mapping(cur, st.session_state["tw_field_mapping"])
                cases[idx] = cur_std
                st.session_state["tw_cases_dataset"] = cases
                st.success("Updated active case in dataset from current form fields.")
                st.rerun()
        with c_dl:
            # Downloads
            ds_json = json.dumps(cases, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("Download cases.json", data=ds_json, file_name="cases.json", mime="application/json", key="tw_cases_dl_json")
            st.download_button("Download cases.csv", data=pd.DataFrame(cases).to_csv(index=False).encode("utf-8"), file_name="cases.csv", mime="text/csv", key="tw_cases_dl_csv")

        with st.expander("Edit cases dataset (table)", expanded=False):
            df = pd.DataFrame(cases).copy()
            df = df[TW_APP_FIELDS] if all(c in df.columns for c in TW_APP_FIELDS) else df
            edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="tw_cases_data_editor")
            if st.button("Apply table edits to dataset", key="tw_apply_cases_table_btn"):
                st.session_state["tw_cases_dataset"] = edited_df.fillna("").to_dict(orient="records")
                st.success("Applied table edits.")
                st.rerun()

        with st.expander("Edit cases dataset (raw JSON)", expanded=False):
            raw = st.text_area("cases.json (editable)", value=json.dumps(cases, ensure_ascii=False, indent=2), height=260, key="tw_cases_raw_json_editor")
            if st.button("Apply JSON edits to dataset", key="tw_apply_cases_json_btn"):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        obj = [obj]
                    if not isinstance(obj, list):
                        raise ValueError("JSON must be an object or a list of objects.")
                    # Standardize each row using rule mapping; skip failed rows
                    ok, failures2 = standardize_tw_dataset_records([x for x in obj if isinstance(x, dict)], st.session_state["tw_field_mapping"])
                    st.session_state["tw_cases_dataset"] = ok
                    st.session_state["tw_cases_failures"] = failures2
                    if ok:
                        st.session_state["tw_active_case_index"] = 0
                        apply_tw_app_dict_to_session(ok[0])
                    st.success(f"Applied JSON edits. OK={len(ok)}, skipped={len(failures2)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
    else:
        st.info("No cases dataset loaded yet. Load a default dataset or upload a JSON/CSV file.")

    st.markdown("---")
    st.markdown("### 0.1) Company Info Templates（公司資訊模板：共用 firm/contact）")

    ctpl1, ctpl2 = st.columns([1.2, 1.8])
    with ctpl1:
        tpl_name = st.text_input("Template name", value="My Company Template", key="company_tpl_name")
        if st.button("Save current form firm/contact as template", key="save_company_tpl_btn"):
            tpl = current_company_from_session()
            tpl["template_name"] = tpl_name.strip() or f"Template {len(st.session_state['company_templates'])+1}"
            tpl["created_at"] = datetime.utcnow().isoformat()
            st.session_state["company_templates"].append(tpl)
            st.success("Saved template.")
            st.rerun()

        # upload templates
        tpl_up = st.file_uploader("Upload company templates (JSON/CSV)", type=["json", "csv"], key="company_tpl_upload")
        if tpl_up is not None and st.button("Import templates", key="company_tpl_import_btn"):
            try:
                if tpl_up.name.lower().endswith(".json"):
                    obj = json.load(tpl_up)
                    if isinstance(obj, dict):
                        obj = [obj]
                    if not isinstance(obj, list):
                        raise ValueError("JSON must be a list of template objects.")
                    st.session_state["company_templates"].extend([x for x in obj if isinstance(x, dict)])
                else:
                    df = pd.read_csv(tpl_up).fillna("")
                    st.session_state["company_templates"].extend(df.to_dict(orient="records"))
                st.success("Imported templates.")
                st.rerun()
            except Exception as e:
                st.error(f"Import templates failed: {e}")

    with ctpl2:
        tpls = st.session_state.get("company_templates", []) or []
        if tpls:
            idx = st.selectbox("Select template", options=list(range(len(tpls))), format_func=lambda i: tpls[i].get("template_name", f"Template {i}"), key="company_tpl_select")
            apply_scope = st.radio("Apply scope", ["Active case only", "All cases in dataset"], horizontal=True, key="company_tpl_scope")
            if st.button("Apply selected template", key="company_tpl_apply_btn"):
                tpl = tpls[idx]
                if not cases:
                    # apply to current form only
                    cur = build_tw_app_dict_from_session()
                    new_case = apply_company_to_case(cur, tpl)
                    apply_tw_app_dict_to_session(new_case)
                else:
                    if apply_scope == "Active case only":
                        ci = st.session_state["tw_active_case_index"]
                        cases[ci] = apply_company_to_case(cases[ci], tpl)
                        st.session_state["tw_cases_dataset"] = cases
                        apply_tw_app_dict_to_session(cases[ci])
                    else:
                        st.session_state["tw_cases_dataset"] = [apply_company_to_case(c, tpl) for c in cases]
                        # keep active index; apply active to form
                        ai = st.session_state["tw_active_case_index"]
                        apply_tw_app_dict_to_session(st.session_state["tw_cases_dataset"][ai])
                st.success("Applied company template.")
                st.rerun()

            st.markdown("**Template preview**")
            st.json({k: tpl.get(k) for k in (["template_name"] + COMPANY_FIELDS)}, expanded=False)

            # download templates
            tpl_json = json.dumps(tpls, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("Download templates.json", data=tpl_json, file_name="company_templates.json", mime="application/json", key="company_tpl_dl_json")
            st.download_button("Download templates.csv", data=pd.DataFrame(tpls).to_csv(index=False).encode("utf-8"), file_name="company_templates.csv", mime="text/csv", key="company_tpl_dl_csv")
        else:
            st.info("No templates yet. Save one from current form, or upload templates.")

    st.markdown("---")
    st.markdown("### 0.2) Field Mapping Dictionary（匯入欄位字典對照：可編輯/下載/上傳）")

    with st.expander("Edit mapping dictionary (alias → standard_key)", expanded=False):
        md_df = mapping_dict_to_df(st.session_state["tw_field_mapping"])
        edited = st.data_editor(md_df, use_container_width=True, num_rows="dynamic", key="tw_mapping_editor")
        cmm1, cmm2, cmm3 = st.columns([1.0, 1.0, 1.0])
        with cmm1:
            if st.button("Apply mapping edits", key="tw_apply_mapping_btn"):
                st.session_state["tw_field_mapping"] = df_to_mapping_dict(edited)
                st.success("Updated mapping dictionary.")
                st.rerun()
        with cmm2:
            upm = st.file_uploader("Upload mapping (JSON/CSV)", type=["json", "csv"], key="tw_mapping_upload")
            if upm is not None and st.button("Import mapping", key="tw_mapping_import_btn"):
                try:
                    incoming = parse_mapping_upload(upm)
                    # merge: uploaded overrides existing
                    st.session_state["tw_field_mapping"] = {**st.session_state["tw_field_mapping"], **incoming}
                    st.success("Imported mapping (merged; uploaded overrides).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import mapping failed: {e}")
        with cmm3:
            st.download_button("Download mapping.json", data=json.dumps(st.session_state["tw_field_mapping"], ensure_ascii=False, indent=2).encode("utf-8"), file_name="tw_field_mapping.json", mime="application/json", key="tw_mapping_dl_json")
            st.download_button("Download mapping.csv", data=mapping_dict_to_df(st.session_state["tw_field_mapping"]).to_csv(index=False).encode("utf-8"), file_name="tw_field_mapping.csv", mime="text/csv", key="tw_mapping_dl_csv")

    st.markdown("---")
    st.markdown("### Step 1 – 線上填寫申請書（草稿）")

    # Keep original form (fields are identical; we reuse the same keys)
    if "tw_app_status" not in st.session_state:
        st.session_state["tw_app_status"] = "pending"
    status_row("申請書填寫", st.session_state["tw_app_status"])

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        doc_no = st.text_input("公文文號", key="tw_doc_no")
        e_no = st.text_input("電子流水號", value=st.session_state.get("tw_e_no", "MDE"), key="tw_e_no")
    with col_a2:
        apply_date_val = st.date_input("申請日", key="tw_apply_date")
        case_type = st.selectbox("案件類型*", ["一般申請案", "同一產品不同品名", "專供外銷", "許可證有效期限屆至後六個月內重新申請"], key="tw_case_type")
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
        replace_flag = st.radio("是否勾選「替代臨床前測試及原廠品質管制資料」？*", ["否", "是"], index=0 if st.session_state.get("tw_replace_flag", "否") == "否" else 1, key="tw_replace_flag")
    with col_a8:
        prior_app_no = st.text_input("（非首次申請）前次申請案號", key="tw_prior_app_no")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        name_zh = st.text_input("醫療器材中文名稱*", key="tw_dev_name_zh")
        name_en = st.text_input("醫療器材英文名稱*", key="tw_dev_name_en")
    with col_b2:
        indications = st.text_area("效能、用途或適應症說明", value=st.session_state.get("tw_indications", "詳如核定之中文說明書"), key="tw_indications")
        spec_comp = st.text_area("型號、規格或主要成分說明", value=st.session_state.get("tw_spec_comp", "詳如核定之中文說明書"), key="tw_spec_comp")

    col_b3, col_b4, col_b5 = st.columns(3)
    with col_b3:
        main_cat = st.selectbox(
            "主類別",
            ["", "A.臨床化學及臨床毒理學", "B.血液學及病理學", "C.免疫學及微生物學",
             "D.麻醉學", "E.心臟血管醫學", "F.牙科學", "G.耳鼻喉科學", "H.胃腸病科學及泌尿科學",
             "I.一般及整形外科手術", "J.一般醫院及個人使用裝置", "K.神經科學", "L.婦產科學", "M.眼科學",
             "N.骨科學", "O.物理醫學科學", "P.放射學科學"],
            key="tw_main_cat",
        )
    with col_b4:
        item_code = st.text_input("分級品項代碼（例：A.1225）", key="tw_item_code")
    with col_b5:
        item_name = st.text_input("分級品項名稱（例：肌氨酸酐試驗系統）", key="tw_item_name")

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

    confirm_match = st.checkbox("我已確認上述資料與最新版醫療器材商證照資訊(名稱、地址、負責人)相符", key="tw_confirm_match")
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        cert_raps = st.checkbox("RAPS", key="tw_cert_raps")
        cert_ahwp = st.checkbox("AHWP", key="tw_cert_ahwp")
    with col_c4:
        cert_other = st.text_input("其它，請敘明", key="tw_cert_other")

    manu_type = st.radio("製造方式", ["單一製造廠", "全部製程委託製造", "委託非全部製程之製造/包裝/貼標/滅菌及最終驗放"], index=0, key="tw_manu_type")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        manu_name = st.text_input("製造廠名稱*", key="tw_manu_name")
        manu_country = st.selectbox("製造國別*", ["TAIWAN， ROC", "UNITED STATES", "EU (Member State)", "JAPAN", "CHINA", "KOREA， REPUBLIC OF", "OTHER"], key="tw_manu_country")
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

    # Completeness + refresh button + missing items report
    st.markdown("---")
    st.markdown("### Application Completeness（含刷新與缺漏清單）")
    ccomp1, ccomp2 = st.columns([1.0, 2.0])
    with ccomp1:
        if st.button("🔄 Refresh Application Completeness", key="tw_refresh_completeness_btn"):
            st.session_state["tw_missing_items_report"] = compute_tw_missing_items_report()
            st.session_state["tw_completeness_last"] = compute_tw_app_completeness()
            st.success("Refreshed completeness + missing items.")
            st.rerun()

    completeness = float(st.session_state.get("tw_completeness_last", compute_tw_app_completeness()))
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

    with ccomp2:
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

    report = st.session_state.get("tw_missing_items_report", {})
    if report:
        with st.expander("What items are not finished?（缺漏項目）", expanded=True):
            st.markdown("**A) Required fields missing**")
            miss = report.get("missing_required_fields", []) or []
            if miss:
                st.write("\n".join([f"- {x}" for x in miss]))
            else:
                st.write("- None")

            st.markdown("**B) Guidance readiness**")
            gr = report.get("guidance_readiness", []) or []
            if gr:
                st.write("\n".join([f"- {x}" for x in gr]))
            else:
                st.write("- OK")

            st.markdown("**C) Optional recommendations**")
            opt = report.get("optional_recommendations", []) or []
            if opt:
                st.write("\n".join([f"- {x}" for x in opt]))
            else:
                st.write("- None")

    # Generate application markdown draft
    st.markdown("---")
    st.markdown("### Generate / Edit Application Markdown")
    if st.button("生成申請書 Markdown 草稿", key="tw_generate_md_btn"):
        missing_labels = []
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
                missing_labels.append(label)

        st.session_state["tw_app_status"] = "error" if missing_labels else "done"
        if missing_labels:
            st.warning("以下基本欄位尚未完整（形式檢查）：\n- " + "\n- ".join(missing_labels))

        apply_date_str = apply_date_val.strftime("%Y-%m-%d") if apply_date_val else ""
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
        # ⭐關鍵：同步更新 editor widget 的 key，讓 UI 立即顯示新內容
        st.session_state["tw_app_md_edited"] = app_md
        st.session_state["tw_app_effective_md"] = app_md  # 可選：讓後續 screening 直接吃到最新
        st.rerun()  # 可選但建議，確保 UI 即刻刷新

    st.markdown("##### 申請書 Markdown（可編輯）")
    app_md_current = st.session_state.get("tw_app_markdown", "")
    view = st.radio("申請書檢視模式", ["Markdown", "純文字"], horizontal=True, key="tw_app_viewmode")
    app_md_edited = st.text_area("申請書內容", value=app_md_current, height=280, key="tw_app_md_edited")
    st.session_state["tw_app_effective_md"] = app_md_edited

    # Guidance manager (new)
    st.markdown("---")
    st.markdown("### Step 2 – Guidance Library / Merge / Diff / Structured Export")

    gcol1, gcol2 = st.columns([1.2, 1.8])
    with gcol1:
        g_mode = st.radio("Guidance Source", ["Default library", "Upload file", "Manual paste"], horizontal=True, key="tw_guidance_source_mode")
    with gcol2:
        if g_mode == "Default library":
            if not DEFAULT_TW_GUIDANCES:
                st.warning("No default guidance available. Check 'defaultguide.md'.")
            else:
                gids = list(DEFAULT_TW_GUIDANCES.keys())
                glabels = [DEFAULT_TW_GUIDANCES[g]["title"] for g in gids]
                selected = st.multiselect("Select guidance(s) to use/merge", glabels, default=[glabels[0]], key="tw_guidance_multi_select")
                extra_rules = st.text_area("Custom additional rules (Markdown; optional)", height=120, key="tw_guidance_extra_rules")
                if st.button("Merge selected guidance(s)", key="tw_guidance_merge_btn"):
                    mds = [DEFAULT_TW_GUIDANCES[gids[glabels.index(x)]]["md"] for x in selected]
                    merged = merge_guidance_markdowns(mds, extra_rules_md=extra_rules)
                    st.session_state["tw_guidance_effective_md"] = merged
                    st.success("Merged guidance into editable Markdown.")
                    st.rerun()

                # Diff compare: choose two
                st.markdown("**Diff compare (A vs B)**")
                a = st.selectbox("A", glabels, index=0, key="tw_guidance_diff_a")
                b = st.selectbox("B", glabels, index=min(1, len(glabels)-1), key="tw_guidance_diff_b")
                if st.button("Show diff", key="tw_guidance_diff_btn"):
                    a_md = DEFAULT_TW_GUIDANCES[gids[glabels.index(a)]]["md"]
                    b_md = DEFAULT_TW_GUIDANCES[gids[glabels.index(b)]]["md"]
                    st.session_state["tw_guidance_diff_text"] = diff_markdown(a_md, b_md)

                diff_text = st.session_state.get("tw_guidance_diff_text", "")
                if diff_text:
                    st.code(diff_text, language="diff")

        elif g_mode == "Upload file":
            gfile = st.file_uploader("Upload guidance (PDF/TXT/MD)", type=["pdf", "txt", "md"], key="tw_guidance_upload_file_new")
            use_ocr = st.checkbox("Use OCR (PDF only; slower)", value=False, key="tw_guidance_upload_ocr")
            if gfile is not None and st.button("Load uploaded guidance", key="tw_load_uploaded_guidance_btn"):
                suffix = gfile.name.lower().rsplit(".", 1)[-1]
                if suffix == "pdf":
                    txt = extract_pdf_pages_to_text(gfile, 1, 9999, use_ocr=use_ocr)
                    # store as markdown text (raw if not converted)
                    st.session_state["tw_guidance_effective_md"] = normalize_md(txt)
                else:
                    st.session_state["tw_guidance_effective_md"] = normalize_md(gfile.read().decode("utf-8", errors="ignore"))
                st.success("Loaded uploaded guidance into editor.")
                st.rerun()
        else:
            pasted = st.text_area("Paste guidance Markdown/Text", height=180, key="tw_guidance_manual_paste_new")
            if st.button("Use pasted guidance", key="tw_guidance_use_paste_btn"):
                st.session_state["tw_guidance_effective_md"] = normalize_md(pasted)
                st.success("Applied pasted guidance.")
                st.rerun()

    st.markdown("#### Guidance Editor (Markdown, stored as Markdown)")
    guidance_md = st.text_area("Guidance Markdown", value=st.session_state.get("tw_guidance_effective_md", ""), height=260, key="tw_guidance_editor_main")
    st.session_state["tw_guidance_effective_md"] = guidance_md

    gdl1, gdl2, gdl3, gdl4 = st.columns([1.0, 1.0, 1.0, 1.0])
    with gdl1:
        st.download_button("Download guidance.md", data=(guidance_md or "").encode("utf-8"), file_name="guidance.md", mime="text/markdown", key="tw_guidance_dl_md")
    with gdl2:
        st.download_button("Download guidance.txt", data=(guidance_md or "").encode("utf-8"), file_name="guidance.txt", mime="text/plain", key="tw_guidance_dl_txt")
    with gdl3:
        if st.button("Convert → Structured fields", key="tw_guidance_to_struct_btn"):
            st.session_state["tw_guidance_struct"] = guidance_markdown_to_struct(guidance_md)
            st.success("Converted guidance to structured fields (deterministic).")
            st.rerun()
    with gdl4:
        if st.button("Clear structured fields", key="tw_guidance_clear_struct_btn"):
            st.session_state["tw_guidance_struct"] = {}
            st.rerun()

    gstruct = st.session_state.get("tw_guidance_struct", {})
    if gstruct:
        with st.expander("Structured guidance (editable + export JSON/CSV)", expanded=True):
            st.json(gstruct, expanded=False)
            edited_struct_raw = st.text_area("Structured JSON (editable)", value=json.dumps(gstruct, ensure_ascii=False, indent=2), height=240, key="tw_guidance_struct_json_editor")
            if st.button("Apply structured JSON edits", key="tw_guidance_apply_struct_json_btn"):
                try:
                    obj = json.loads(edited_struct_raw)
                    if not isinstance(obj, dict):
                        raise ValueError("Structured JSON must be an object.")
                    st.session_state["tw_guidance_struct"] = obj
                    st.success("Applied structured edits.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

            st.download_button(
                "Download structured_guidance.json",
                data=json.dumps(st.session_state["tw_guidance_struct"], ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="structured_guidance.json",
                mime="application/json",
                key="tw_guidance_struct_dl_json",
            )
            st.download_button(
                "Download structured_guidance.csv (1 row)",
                data=guidance_struct_to_one_row_csv(st.session_state["tw_guidance_struct"]),
                file_name="structured_guidance.csv",
                mime="text/csv",
                key="tw_guidance_struct_dl_csv",
            )
            st.download_button(
                "Download required_documents.csv",
                data=guidance_required_docs_csv(st.session_state["tw_guidance_struct"]),
                file_name="guidance_required_documents.csv",
                mime="text/csv",
                key="tw_guidance_docs_dl_csv",
            )

    # Screening input
    st.markdown("---")
    st.markdown("### Step 3 – 形式審查 / 完整性檢核（Agent）")
    base_app_md = st.session_state.get("tw_app_effective_md", "")
    base_guidance = st.session_state.get("tw_guidance_effective_md", "") or "（尚未提供指引，請依一般法規常規進行形式檢核）"

    combined_input = f"""=== 申請書草稿（Markdown） ===
{base_app_md}

=== 預審 / 形式審查指引（Markdown） ===
{base_guidance}
"""

    default_screen_prompt = """你是一位熟悉臺灣「第二、三等級醫療器材查驗登記」的形式審查(預審)審查員。

請根據：
1) 申請書草稿（Markdown）
2) 預審/形式審查指引（如有）

以繁體中文 Markdown 輸出：
- 形式完整性檢核表（含：文件項目｜預期應附？｜是否提及？｜是否檢附？｜判定｜備註/補件）
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


# ============================================================
# 19) 510(k) Intelligence tab (kept)
# ============================================================
def render_510k_tab():
    st.markdown("## 510(k) Intelligence")
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


# ============================================================
# 20) PDF → Markdown tab (kept)
# ============================================================
def render_pdf_to_md_tab():
    st.markdown("## PDF → Markdown")
    uploaded = st.file_uploader("Upload PDF to convert selected pages to Markdown", type=["pdf"], key="pdf_to_md_uploader")
    if uploaded:
        c1, c2 = st.columns(2)
        with c1:
            num_start = st.number_input("From page", min_value=1, value=1, key="pdf_to_md_from")
        with c2:
            num_end = st.number_input("To page", min_value=1, value=5, key="pdf_to_md_to")

            text = extract_pdf_pages_to_text(uploaded, int(num_start), int(num_end), use_ocr=False)
            st.session_state["pdf_raw_text"] = text
            st.success("Text extracted (Raw). Run agent below to clean up.")

        with c2:
            if st.button("Extract with OCR (Slower)", key="pdf_ocr_btn"):
                with st.spinner("Running OCR..."):
                    text = extract_pdf_pages_to_text(uploaded, int(num_start), int(num_end), use_ocr=True)
                    st.session_state["pdf_raw_text"] = text
                    st.success("OCR Complete. Run agent below to clean up.")

    raw_text = st.session_state.get("pdf_raw_text", "")
    if raw_text:
        st.markdown("---")
        all_agents = list(st.session_state["agents_cfg"]["agents"].keys())
        default_idx = all_agents.index("pdf_to_markdown_agent") if "pdf_to_markdown_agent" in all_agents else 0
        selected_agent = st.selectbox("Select Agent for Processing", all_agents, index=default_idx, key="pdf_agent_select")
        sel_cfg = st.session_state["agents_cfg"]["agents"][selected_agent]
        base_def_prompt = sel_cfg.get("user_prompt_template", "") or sel_cfg.get("system_prompt", "")

        if selected_agent == "pdf_to_markdown_agent":
            base_def_prompt = f"""
You are converting part of a regulatory PDF into markdown.

- Produce clean, structured markdown preserving headings, lists, tables (as markdown tables if possible).
- Do not hallucinate content.

Language: {lang_code()}.
"""

        agent_run_ui(
            agent_id=selected_agent,
            tab_key="pdf_to_md",
            default_prompt=base_def_prompt,
            default_input_text=raw_text,
            tab_label_for_history="PDF Processing",
        )
    else:
        st.info("Upload a PDF and click 'Extract Text' to begin.")


# ============================================================
# 21) 510(k) Review Pipeline tab (updated with default checklist+guidance management)
# ============================================================

def parse_510k_checklist_upload(file) -> List[Dict[str, Any]]:
    """
    Accept JSON list of dicts or CSV with at least columns: section, item, expected, notes(optional).
    Also accept "non-standard" column names via simple alias mapping.
    """
    name = (file.name or "").lower()
    alias = {
        "Section": "section",
        "SECTION": "section",
        "Item": "item",
        "ITEM": "item",
        "Requirement": "item",
        "Expected": "expected",
        "EXPECTED": "expected",
        "Notes": "notes",
        "NOTE": "notes",
        "Finding": "notes",
    }

    if name.endswith(".json"):
        obj = json.load(file)
        if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
            obj = obj["items"]
        if isinstance(obj, list):
            out = []
            for x in obj:
                if not isinstance(x, dict):
                    continue
                y = {}
                for k, v in x.items():
                    kk = alias.get(k, k)
                    y[kk] = v
                out.append(
                    {
                        "section": str(y.get("section", "")).strip(),
                        "item": str(y.get("item", "")).strip(),
                        "expected": str(y.get("expected", "")).strip(),
                        "notes": str(y.get("notes", "")).strip(),
                    }
                )
            return out
        return []

    if name.endswith(".csv"):
        df = pd.read_csv(file).fillna("")
        # rename columns
        cols = {}
        for c in df.columns:
            cols[c] = alias.get(c, c)
        df = df.rename(columns=cols)
        # ensure required
        for col in ["section", "item", "expected", "notes"]:
            if col not in df.columns:
                df[col] = ""
        out = []
        for _, r in df.iterrows():
            out.append(
                {
                    "section": str(r.get("section", "")).strip(),
                    "item": str(r.get("item", "")).strip(),
                    "expected": str(r.get("expected", "")).strip(),
                    "notes": str(r.get("notes", "")).strip(),
                }
            )
        # filter empty rows
        out = [x for x in out if (x["section"] or x["item"] or x["expected"] or x["notes"])]
        return out

    raise ValueError("Unsupported checklist upload (JSON/CSV).")


def checklist_items_to_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    df = pd.DataFrame(items)
    for col in ["section", "item", "expected", "notes"]:
        if col not in df.columns:
            df[col] = ""
    df = df[["section", "item", "expected", "notes"]]
    lines = ["| Section | Item | Expected | Notes |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['section']} | {r['item']} | {r['expected']} | {r['notes']} |")
    return "\n".join(lines)


def render_510k_review_pipeline_tab():
    st.markdown("## 510(k) Review Pipeline")
    st.caption("Updated: includes Checklist + Guidance manager (default/upload/edit/download), similar to TW Premarket.")

    st.markdown("---")
    st.markdown("### 0) Checklist + Guidance Manager (Default / Upload / Edit / Download)")

    DEFAULT_510K_CHECKLIST_SETS = st.session_state.get("DEFAULT_510K_CHECKLIST_SETS", {})
    DEFAULT_510K_GUIDANCES = st.session_state.get("DEFAULT_510K_GUIDANCES", {})

    cm1, cm2 = st.columns([1.2, 1.8])
    with cm1:
        chk_mode = st.radio("Checklist source", ["Default library", "Upload file"], horizontal=True, key="k510_chk_source_mode")
    with cm2:
        if chk_mode == "Default library":
            if not DEFAULT_510K_CHECKLIST_SETS:
                st.warning("No default checklists available. Check 'defaultdataset.json'.")
            else:
                keys = list(DEFAULT_510K_CHECKLIST_SETS.keys())
                labels = [DEFAULT_510K_CHECKLIST_SETS[k]["title"] for k in keys]
                sel = st.selectbox("Select default checklist dataset", labels, index=0, key="k510_chk_default_sel")
                if st.button("Load default checklist", key="k510_chk_load_default_btn"):
                    dsid = keys[labels.index(sel)]
                    st.session_state["k510_checklist_dataset"] = DEFAULT_510K_CHECKLIST_SETS[dsid]["items"]
                    st.success("Loaded default checklist dataset.")
                    st.rerun()
        else:
            up = st.file_uploader("Upload checklist dataset (JSON/CSV)", type=["json", "csv"], key="k510_chk_upload")
            if up is not None and st.button("Import checklist", key="k510_chk_import_btn"):
                try:
                    items = parse_510k_checklist_upload(up)
                    st.session_state["k510_checklist_dataset"] = items
                    st.success(f"Imported checklist items: {len(items)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    items = st.session_state.get("k510_checklist_dataset", []) or []
    if items:
        with st.expander("Edit checklist dataset (table)", expanded=False):
            df = pd.DataFrame(items).fillna("")
            for col in ["section", "item", "expected", "notes"]:
                if col not in df.columns:
                    df[col] = ""
            df = df[["section", "item", "expected", "notes"]]
            edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="k510_chk_editor")
            if st.button("Apply checklist table edits", key="k510_chk_apply_table_btn"):
                st.session_state["k510_checklist_dataset"] = edited_df.fillna("").to_dict(orient="records")
                st.success("Applied edits.")
                st.rerun()

        with st.expander("Edit checklist dataset (raw JSON)", expanded=False):
            raw = st.text_area("checklist.json (editable)", value=json.dumps(items, ensure_ascii=False, indent=2), height=220, key="k510_chk_raw_editor")
            if st.button("Apply checklist JSON edits", key="k510_chk_apply_json_btn"):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and "items" in obj:
                        obj = obj["items"]
                    if not isinstance(obj, list):
                        raise ValueError("Must be a list of items.")
                    cleaned = []
                    for x in obj:
                        if not isinstance(x, dict):
                            continue
                        cleaned.append(
                            {
                                "section": str(x.get("section", "")).strip(),
                                "item": str(x.get("item", "")).strip(),
                                "expected": str(x.get("expected", "")).strip(),
                                "notes": str(x.get("notes", "")).strip(),
                            }
                        )
                    st.session_state["k510_checklist_dataset"] = [x for x in cleaned if (x["section"] or x["item"] or x["expected"] or x["notes"])]
                    st.success("Applied JSON edits.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        dl1, dl2, dl3 = st.columns([1.0, 1.0, 1.0])
        with dl1:
            st.download_button("Download checklist.json", data=json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"), file_name="510k_checklist.json", mime="application/json", key="k510_chk_dl_json")
        with dl2:
            st.download_button("Download checklist.csv", data=pd.DataFrame(items).to_csv(index=False).encode("utf-8"), file_name="510k_checklist.csv", mime="text/csv", key="k510_chk_dl_csv")
        with dl3:
            md_tbl = checklist_items_to_markdown(items)
            st.download_button("Download checklist.md (table)", data=md_tbl.encode("utf-8"), file_name="510k_checklist.md", mime="text/markdown", key="k510_chk_dl_md")

    st.markdown("#### 510(k) Guidance Library (merge/diff basic)")
    gmode = st.radio("510(k) guidance source", ["Default library", "Upload file", "Manual paste"], horizontal=True, key="k510_guid_source")
    if gmode == "Default library":
        if not DEFAULT_510K_GUIDANCES:
            st.warning("No default guidances available. Check 'defaultguide.md'.")
        else:
            gids = list(DEFAULT_510K_GUIDANCES.keys())
            glabels = [DEFAULT_510K_GUIDANCES[g]["title"] for g in gids]
            sel = st.multiselect("Select guidance(s) to merge", glabels, default=[glabels[0]], key="k510_guid_multi")
            extra = st.text_area("Custom additional rules (optional)", height=120, key="k510_guid_extra")
            if st.button("Merge 510(k) guidance(s)", key="k510_guid_merge_btn"):
                mds = [DEFAULT_510K_GUIDANCES[gids[glabels.index(x)]]["md"] for x in sel]
                st.session_state["k510_guidance_effective_md"] = merge_guidance_markdowns(mds, extra_rules_md=extra)
                st.success("Merged guidance.")
                st.rerun()
            st.markdown("**Diff compare (A vs B)**")
            a = st.selectbox("A", glabels, index=0, key="k510_guid_diff_a")
            b = st.selectbox("B", glabels, index=0, key="k510_guid_diff_b")
            if st.button("Show diff", key="k510_guid_diff_btn"):
                a_md = DEFAULT_510K_GUIDANCES[gids[glabels.index(a)]]["md"]
                b_md = DEFAULT_510K_GUIDANCES[gids[glabels.index(b)]]["md"]
                st.session_state["k510_guid_diff_text"] = diff_markdown(a_md, b_md)
            if st.session_state.get("k510_guid_diff_text"):
                st.code(st.session_state["k510_guid_diff_text"], language="diff")

    elif gmode == "Upload file":
        gfile = st.file_uploader("Upload 510(k) guidance (PDF/TXT/MD)", type=["pdf", "txt", "md"], key="k510_guid_upload")
        use_ocr = st.checkbox("Use OCR (PDF only)", value=False, key="k510_guid_ocr")
        if gfile is not None and st.button("Load uploaded 510(k) guidance", key="k510_guid_load_btn"):
            suffix = gfile.name.lower().rsplit(".", 1)[-1]
            if suffix == "pdf":
                txt = extract_pdf_pages_to_text(gfile, 1, 9999, use_ocr=use_ocr)
                st.session_state["k510_guidance_effective_md"] = normalize_md(txt)
            else:
                st.session_state["k510_guidance_effective_md"] = normalize_md(gfile.read().decode("utf-8", errors="ignore"))
            st.success("Loaded guidance.")
            st.rerun()
    else:
        pasted = st.text_area("Paste 510(k) guidance", height=160, key="k510_guid_paste")
        if st.button("Use pasted 510(k) guidance", key="k510_guid_use_paste_btn"):
            st.session_state["k510_guidance_effective_md"] = normalize_md(pasted)
            st.rerun()

    k510_guid = st.text_area("510(k) guidance (Markdown)", value=st.session_state.get("k510_guidance_effective_md", ""), height=220, key="k510_guid_editor")
    st.session_state["k510_guidance_effective_md"] = k510_guid
    st.download_button("Download 510k_guidance.md", data=(k510_guid or "").encode("utf-8"), file_name="510k_guidance.md", mime="text/markdown", key="k510_guid_dl_md")

    st.markdown("---")
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

    # Use managed checklist if available; allow manual override
    managed_chk_md = checklist_items_to_markdown(st.session_state.get("k510_checklist_dataset", []) or [])
    chk_md_default = managed_chk_md if managed_chk_md.strip() else ""
    chk_md = st.text_area("Checklist (markdown or text). Tip: load/edit in manager above.", height=200, value=chk_md_default, key="chk_md")

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

Keep it grounded in evidence. If evidence is missing, say so explicitly.
"""
            # include pipeline guidance if provided
            guidance = st.session_state.get("k510_guidance_effective_md", "")
            user_prompt = rep_prompt + "\n\n=== PIPELINE GUIDANCE (optional) ===\n" + guidance + "\n\n=== CHECKLIST ===\n" + chk_md + "\n\n=== STRUCTURED SUBMISSION ===\n" + subm_md_eff
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
# 22) Note Keeper (kept; trimmed for brevity of this updated file)
# ============================================================
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (s or "").strip())


def read_uploaded_note_file() -> str:
    file = st.session_state.get("note_file")
    if not file:
        return ""
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf_pages_to_text(file, 1, 9999)
    return file.read().decode("utf-8", errors="ignore")


def highlight_keywords_html(text: str, keywords: List[str], color: str = "#FF7F50") -> str:
    if not text.strip() or not keywords:
        return text
    kws = sorted({k.strip() for k in keywords if k and k.strip()}, key=len, reverse=True)
    out = text
    for kw in kws:
        if re.search(r"[A-Za-z0-9]", kw):
            pattern = re.compile(rf"(?<![\w-]){re.escape(kw)}(?![\w-])")
            out = pattern.sub(rf'<span style="color:{color};font-weight:800;">{kw}</span>', out)
        else:
            out = out.replace(kw, f'<span style="color:{color};font-weight:800;">{kw}</span>')
    return out


def magic_ai_keywords(base_md: str, color: str, model: str) -> Tuple[List[str], str]:
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
    raw = call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=1500, temperature=0.1, api_keys=api_keys)
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
    st.markdown("## Note Keeper & Magics")
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
                    out = call_llm(model=note_model, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=int(note_max_tokens), temperature=0.15, api_keys=api_keys)
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

    # Keep remaining 5 magics as in original (compact)
    def _magic_block(title: str, key_prefix: str, default_model: str, system_prompt: str, default_prompt: str, temperature: float):
        st.markdown(f"#### {title}")
        m = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(default_model) if default_model in ALL_MODELS else 0, key=f"{key_prefix}_model")
        p = st.text_area(t("prompt"), value=default_prompt, height=120, key=f"{key_prefix}_prompt")
        if st.button(t("run"), key=f"{key_prefix}_btn"):
            try:
                out = call_llm(
                    model=m,
                    system_prompt=system_prompt,
                    user_prompt=p + "\n\n=== NOTE ===\n" + base_note,
                    max_tokens=12000,
                    temperature=temperature,
                    api_keys=st.session_state.get("api_keys", {}),
                )
                st.session_state[f"{key_prefix}_out"] = out
                log_event("Note Keeper", title, m, est_tokens(base_note + out))
            except Exception as e:
                st.error(f"{title} failed: {e}")
        if st.session_state.get(f"{key_prefix}_out"):
            st.text_area(f"{title} (editable)", value=st.session_state[f"{key_prefix}_out"], height=220, key=f"{key_prefix}_out_edit")

    _magic_block(
        title="2) Summarize",
        key_prefix="sum",
        default_model="gpt-4o-mini",
        system_prompt="You write executive summaries for technical/regulatory notes.",
        default_prompt="Summarize into 8–12 bullets + a 5-sentence executive summary. Keep terminology accurate. Output Markdown.",
        temperature=0.2,
    )
    _magic_block(
        title="3) Polisher",
        key_prefix="pol",
        default_model="gpt-4.1-mini",
        system_prompt=st.session_state["agents_cfg"]["agents"]["polisher"].get("system_prompt", ""),
        default_prompt="Polish for clarity, grammar, and professional tone. Do not add facts. Keep Markdown structure.",
        temperature=0.15,
    )
    _magic_block(
        title="4) Critique",
        key_prefix="cri",
        default_model="claude-3-5-sonnet-20241022",
        system_prompt=st.session_state["agents_cfg"]["agents"]["critic"].get("system_prompt", ""),
        default_prompt="Provide constructive critique: unclear areas, missing evidence, risks, contradictions, and specific improvement suggestions. Output Markdown with sections.",
        temperature=0.35,
    )
    _magic_block(
        title="5) Poet Mode",
        key_prefix="poe",
        default_model="gemini-3-flash-preview",
        system_prompt=st.session_state["agents_cfg"]["agents"]["poet_laureate"].get("system_prompt", ""),
        default_prompt=f"Transform the note into poetic or artistic prose inspired by the current UI style: {st.session_state.settings['painter_style']}. Preserve core meaning; do not add facts.",
        temperature=0.75,
    )
    target = "Traditional Chinese (zh-TW)" if lang_code() == "zh-tw" else "English"
    _magic_block(
        title="6) Translate",
        key_prefix="tr",
        default_model="gemini-2.5-flash",
        system_prompt=st.session_state["agents_cfg"]["agents"]["translator"].get("system_prompt", ""),
        default_prompt=f"Detect the language and translate the note into {target}. Preserve markdown structure and technical terms.",
        temperature=0.2,
    )

    st.markdown("---")
    st.markdown("### Run Any Agent")
    agent_options = list(st.session_state["agents_cfg"]["agents"].keys())
    if agent_options:
        selected_agent = st.selectbox("Select Agent", agent_options, key="note_generic_agent_select")
        sel_cfg = st.session_state["agents_cfg"]["agents"][selected_agent]
        def_prompt = sel_cfg.get("user_prompt_template", "") or "Analyze the input note."
        agent_run_ui(agent_id=selected_agent, tab_key="note_generic", default_prompt=def_prompt, default_input_text=base_note, allow_model_override=True, tab_label_for_history="Note Keeper Generic")

    if base_note.strip():
        st.download_button(t("download_md"), data=base_note.encode("utf-8"), file_name="note.md", mime="text/markdown")


# ============================================================
# 23) Agents Config tab (kept)
# ============================================================
def render_agents_config_tab():
    st.markdown("## Agents Config Studio")
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
        st.download_button("Download current agents.yaml", data=yaml_str_current.encode("utf-8"), file_name="agents.yaml", mime="text/yaml", key="download_agents_yaml_current")


# ============================================================
# 24) Tabs / Main render
# ============================================================
LABELS = {
    "Dashboard": {"en": "Dashboard", "zh-tw": "儀表板"},
    "Workflow Studio": {"en": "Agent Workflow Studio", "zh-tw": "代理工作流工作室"},
    "TW Premarket": {"en": "TW Premarket Application", "zh-tw": "第二、三等級醫療器材查驗登記"},
    "510k_tab": {"en": "510(k) Intelligence", "zh-tw": "510(k) 智能分析"},
    "PDF → Markdown": {"en": "PDF → Markdown", "zh-tw": "PDF → Markdown"},
    "Checklist & Report": {"en": "510(k) Review Pipeline", "zh-tw": "510(k) 審查全流程"},
    "Note Keeper & Magics": {"en": "Note Keeper & Magics", "zh-tw": "筆記助手與魔法"},
    "Agents Config": {"en": "Agents Config Studio", "zh-tw": "代理設定工作室"},
}


def tl(key: str) -> str:
    return LABELS.get(key, {}).get(lang_code(), key)


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
