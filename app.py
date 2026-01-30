import os
import re
import json
import base64
import difflib
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import yaml
import pandas as pd
import altair as alt
from pypdf import PdfReader

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx

# Optional OCR deps (graceful fallback on HF if system binaries missing)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
except Exception:
    pytesseract = None
    convert_from_bytes = None
    Image = None


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Antigravity Agentic Data+Guidance Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Models & Providers
# ============================================================
ALL_MODELS = [
    "gpt-4o-mini", "gpt-4.1-mini",
    "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-3-pro-preview",
    "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
    "grok-4-fast-reasoning", "grok-4-1-fast-non-reasoning",
]
OPENAI_MODELS = {"gpt-4o-mini", "gpt-4.1-mini"}
GEMINI_MODELS = {"gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-3-pro-preview"}
ANTHROPIC_MODELS = {"claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"}
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
    raise ValueError(f"Unknown model: {model}")


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# ============================================================
# Session State Init
# ============================================================
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {"openai": "", "gemini": "", "anthropic": "", "grok": ""}

if "settings" not in st.session_state:
    st.session_state.settings = {
        "model": "gpt-4o-mini",
        "max_tokens": 12000,
        "temperature": 0.2,
        # NEW: SKILL injection
        "inject_skill": True,
        "inject_skill_max_chars": 6000,  # guardrail
    }

if "history" not in st.session_state:
    st.session_state.history = []

if "agents_cfg" not in st.session_state:
    st.session_state.agents_cfg = {"agents": {}}

if "skill_md" not in st.session_state:
    try:
        with open("SKILL.md", "r", encoding="utf-8") as f:
            st.session_state.skill_md = f.read()
    except Exception:
        st.session_state.skill_md = ""

if "bundle" not in st.session_state:
    st.session_state.bundle = {
        "defaultdataset": {"tw_cases": {}, "k510_checklists": {}, "meta": {"generated_at": now_iso(), "generated_by": {}}},
        "defaultguide": "",
        "bundle_meta": {"last_updated": now_iso()},
        "saved_prompts": [],  # for "keep prompt on results"
    }


# ============================================================
# API Key Helpers
# ============================================================
def env_key_present(env_var: str) -> bool:
    v = os.getenv(env_var, "")
    return bool(v and v.strip())


def get_api_key(provider: str) -> str:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "grok": "GROK_API_KEY",
    }
    env_var = mapping.get(provider, "")
    return (st.session_state.api_keys.get(provider) or os.getenv(env_var) or "").strip()


# ============================================================
# LLM Router + SKILL Injection (NEW)
# ============================================================
def build_system_prompt(agent_system_prompt: str) -> str:
    """
    NEW: auto-inject SKILL.md as shared knowledge into ALL agent calls (configurable).
    """
    base = (agent_system_prompt or "").strip()
    if not st.session_state.settings.get("inject_skill", True):
        return base

    skill = (st.session_state.skill_md or "").strip()
    if not skill:
        return base

    max_chars = int(st.session_state.settings.get("inject_skill_max_chars", 6000))
    skill = skill[:max_chars]

    injected = f"""
[SHARED KNOWLEDGE: SKILL.md]
{skill}
[/SHARED KNOWLEDGE]

[AGENT SYSTEM PROMPT]
{base}
""".strip()
    return injected


def call_llm(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
    provider = get_provider(model)
    key = get_api_key(provider)
    if not key:
        raise RuntimeError(f"Missing API key for provider: {provider}")

    system_prompt = build_system_prompt(system_prompt)

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
        with httpx.Client(base_url="https://api.x.ai/v1", timeout=120) as client:
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

    raise RuntimeError("Unsupported provider")


# ============================================================
# File / Format Utilities
# ============================================================
SECTION_RE = re.compile(
    r"<!--\s*BEGIN_SECTION:\s*(.*?)\s*\|\s*TITLE:\s*(.*?)\s*-->(.*?)<!--\s*END_SECTION\s*-->",
    re.DOTALL
)


def normalize_md(md: str) -> str:
    md = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        s = text[text.find("{"): text.rfind("}") + 1]
        return json.loads(s)


def parse_defaultguide_sections(md: str) -> List[Dict[str, str]]:
    md = normalize_md(md)
    out = []
    for sid, title, body in SECTION_RE.findall(md):
        out.append({"id": sid.strip(), "title": title.strip(), "md": body.strip()})
    return out


def build_defaultguide_from_sections(sections: List[Dict[str, str]]) -> str:
    blocks = []
    for s in sections:
        sid = s["id"].strip()
        title = s["title"].strip()
        body = normalize_md(s.get("md", ""))
        blocks.append(f"<!-- BEGIN_SECTION: {sid} | TITLE: {title} -->\n{body}\n<!-- END_SECTION -->")
    return normalize_md("\n\n\n".join(blocks))


def is_standard_defaultguide(md: str) -> bool:
    sections = parse_defaultguide_sections(md)
    if not sections:
        return False
    for s in sections:
        if not (s["id"].startswith("tw_") or s["id"].startswith("k510_")):
            return False
    return True


def is_standard_defaultdataset(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "tw_cases" not in obj or "k510_checklists" not in obj:
        return False
    if not isinstance(obj["tw_cases"], dict) or not isinstance(obj["k510_checklists"], dict):
        return False
    for _, v in obj["tw_cases"].items():
        if not isinstance(v, dict):
            return False
        if "title" not in v or "cases" not in v or not isinstance(v["cases"], list):
            return False
    for _, v in obj["k510_checklists"].items():
        if not isinstance(v, dict):
            return False
        if "title" not in v or "items" not in v or not isinstance(v["items"], list):
            return False
    return True


def deterministic_standardize_defaultdataset(obj: Any, meta: Optional[dict] = None) -> Dict[str, Any]:
    meta = meta or {}
    out = {"tw_cases": {}, "k510_checklists": {}, "meta": {"generated_at": now_iso(), "generated_by": meta}}
    if isinstance(obj, dict):
        tw = obj.get("tw_cases", {}) if isinstance(obj.get("tw_cases", {}), dict) else {}
        k510 = obj.get("k510_checklists", {}) if isinstance(obj.get("k510_checklists", {}), dict) else {}
        out["tw_cases"] = tw
        out["k510_checklists"] = k510
        if isinstance(obj.get("meta"), dict):
            out["meta"] = obj["meta"]
            out["meta"].setdefault("generated_at", now_iso())
            out["meta"].setdefault("generated_by", meta)

    # Normalize checklist items minimal keys
    for cid, c in (out.get("k510_checklists") or {}).items():
        if not isinstance(c, dict):
            continue
        c.setdefault("title", str(cid))
        c.setdefault("items", [])
        items = c.get("items") if isinstance(c.get("items"), list) else []
        norm_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            norm_items.append({
                "section": str(it.get("section", "")).strip(),
                "item": str(it.get("item", "")).strip(),
                "expected": str(it.get("expected", "")).strip(),
                "notes": str(it.get("notes", "")).strip(),
            })
        c["items"] = norm_items

    return out


def diff_text(a: str, b: str) -> str:
    a_lines = (a or "").splitlines(keepends=True)
    b_lines = (b or "").splitlines(keepends=True)
    return "".join(difflib.unified_diff(a_lines, b_lines, fromfile="A", tofile="B")).strip()


# ============================================================
# Agents Config Loading
# ============================================================
def load_agents_cfg_from_disk() -> Dict[str, Any]:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "agents" not in cfg:
            cfg["agents"] = {}
        return cfg
    except Exception:
        return {"agents": {}}


def agent_cfg(agent_id: str) -> Dict[str, Any]:
    return (st.session_state.agents_cfg.get("agents") or {}).get(agent_id, {})


def run_agent(agent_id: str, user_prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    cfg = agent_cfg(agent_id)
    system_prompt = cfg.get("system_prompt", "")
    return call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens, temperature=temperature)


def standardize_dataset_with_agent(raw_text: str, model: str, max_tokens: int) -> Dict[str, Any]:
    cfg = agent_cfg("dataset_standardizer")
    out = call_llm(
        model=model,
        system_prompt=cfg.get("system_prompt", ""),
        user_prompt=f"RAW_DATASET_INPUT:\n{raw_text}",
        max_tokens=max_tokens,
        temperature=0.0
    )
    return safe_json_loads(out)


def standardize_guide_with_agent(raw_text: str, model: str, max_tokens: int) -> str:
    cfg = agent_cfg("guide_standardizer")
    out = call_llm(
        model=model,
        system_prompt=cfg.get("system_prompt", ""),
        user_prompt=f"RAW_GUIDE_INPUT:\n{raw_text}",
        max_tokens=max_tokens,
        temperature=0.0
    )
    return normalize_md(out)


def standardize_agents_yaml_with_agent(raw_text: str, model: str, max_tokens: int) -> Dict[str, Any]:
    cfg = agent_cfg("agents_yaml_standardizer")
    out = call_llm(
        model=model,
        system_prompt=cfg.get("system_prompt", ""),
        user_prompt=raw_text,
        max_tokens=max_tokens,
        temperature=0.0
    )
    clean = out.replace("```yaml", "").replace("```", "").strip()
    data = yaml.safe_load(clean) or {}
    if "agents" not in data:
        data["agents"] = {}
    return data


# ============================================================
# PDF: preview + page range + OCR (NEW)
# ============================================================
def pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        r = PdfReader(BytesIO(pdf_bytes))
        return len(r.pages)
    except Exception:
        return 0


def extract_pdf_pages_text(pdf_bytes: bytes, start_page: int, end_page: int) -> str:
    try:
        r = PdfReader(BytesIO(pdf_bytes))
        n = len(r.pages)
        start = max(1, int(start_page))
        end = min(n, int(end_page))
        texts = []
        for i in range(start - 1, end):
            texts.append(r.pages[i].extract_text() or "")
        return normalize_md("\n\n".join(texts))
    except Exception as e:
        return normalize_md(f"[System] PDF extraction failed: {e}")


def ocr_pdf_pages_text(pdf_bytes: bytes, start_page: int, end_page: int, lang: str = "eng+chi_tra") -> str:
    if pytesseract is None or convert_from_bytes is None:
        return "[System] OCR requested but pytesseract/pdf2image/PIL not available in this environment."

    try:
        start = max(1, int(start_page))
        end = max(start, int(end_page))
        images = convert_from_bytes(pdf_bytes, first_page=start, last_page=end)
        out = []
        for img in images:
            out.append(pytesseract.image_to_string(img, lang=lang))
        return normalize_md("\n\n".join(out))
    except Exception as e:
        return normalize_md(f"[System] OCR failed (often requires poppler/tesseract system packages): {e}")


def show_pdf_bytes(pdf_bytes: bytes, height: int = 600):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}"></iframe>""",
        unsafe_allow_html=True
    )


# ============================================================
# Markdown table extraction for Harmonization outputs (NEW)
# ============================================================
def extract_first_markdown_table(md: str) -> Optional[pd.DataFrame]:
    """
    Best-effort parse of the FIRST markdown table found.
    Expects a pipe table.
    """
    lines = (md or "").splitlines()
    # find header line like | a | b |
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and line.strip().endswith("|") and "|" in line.strip()[1:-1]:
            # next line must contain --- separators
            if i + 1 < len(lines) and re.search(r"\|\s*-{3,}", lines[i + 1]):
                start_idx = i
                break
    if start_idx is None:
        return None

    # collect contiguous table lines
    tbl = []
    for j in range(start_idx, len(lines)):
        if not lines[j].strip().startswith("|"):
            break
        tbl.append(lines[j].strip())
    if len(tbl) < 2:
        return None

    # parse header
    header = [c.strip() for c in tbl[0].strip("|").split("|")]
    rows = []
    for row_line in tbl[2:]:  # skip header + separator
        cols = [c.strip() for c in row_line.strip("|").split("|")]
        # pad/truncate to header length
        if len(cols) < len(header):
            cols += [""] * (len(header) - len(cols))
        cols = cols[:len(header)]
        rows.append(cols)
    return pd.DataFrame(rows, columns=header)


# ============================================================
# UI: Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## Global Settings")
    st.session_state.settings["model"] = st.selectbox(
        "Default model", ALL_MODELS,
        index=ALL_MODELS.index(st.session_state.settings["model"]) if st.session_state.settings["model"] in ALL_MODELS else 0
    )
    st.session_state.settings["max_tokens"] = st.number_input(
        "Default max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000
    )
    st.session_state.settings["temperature"] = st.slider(
        "Temperature", 0.0, 1.0, float(st.session_state.settings["temperature"]), 0.05
    )

    st.markdown("---")
    st.markdown("## SKILL Injection (Shared Knowledge)")
    st.session_state.settings["inject_skill"] = st.checkbox(
        "Auto-inject SKILL.md into ALL agents as system prompt context",
        value=bool(st.session_state.settings.get("inject_skill", True))
    )
    st.session_state.settings["inject_skill_max_chars"] = st.number_input(
        "Max chars injected from SKILL.md (guardrail)",
        1000, 50000, int(st.session_state.settings.get("inject_skill_max_chars", 6000)), 500
    )

    st.markdown("---")
    st.markdown("## API Keys")

    def api_key_row(label: str, env_var: str, provider: str):
        if env_key_present(env_var):
            st.caption(f"{label}: Active (Env)")
        else:
            st.session_state.api_keys[provider] = st.text_input(
                f"{label} API Key", value=st.session_state.api_keys[provider], type="password"
            )

    api_key_row("OpenAI", "OPENAI_API_KEY", "openai")
    api_key_row("Gemini", "GEMINI_API_KEY", "gemini")
    api_key_row("Anthropic", "ANTHROPIC_API_KEY", "anthropic")
    api_key_row("Grok (xAI)", "GROK_API_KEY", "grok")

    st.markdown("---")
    st.markdown("## Config")
    if st.button("Reload agents.yaml from disk"):
        st.session_state.agents_cfg = load_agents_cfg_from_disk()
        st.success("Reloaded agents.yaml")
        st.rerun()


# Load agents on first render if empty
if not (st.session_state.agents_cfg.get("agents") or {}):
    st.session_state.agents_cfg = load_agents_cfg_from_disk()


# ============================================================
# Header
# ============================================================
st.markdown("# Antigravity Agentic Data+Guidance Studio")
st.caption(
    "Upload → Standardize → Edit → Combine → Generate bundles (defaultdataset.json + defaultguide.md), "
    "plus FDA guidance authoring tools. Includes SKILL.md shared knowledge injection."
)


# ============================================================
# Shared UI Blocks
# ============================================================
def bundle_editors():
    st.markdown("### Current Bundle Editors")

    c1, c2 = st.columns(2)
    with c1:
        ds_text = st.text_area(
            "defaultdataset.json (editable)",
            value=json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2),
            height=440,
            key="ds_editor",
        )
        cA, cB = st.columns(2)
        with cA:
            if st.button("Apply dataset edits"):
                obj = safe_json_loads(ds_text)
                st.session_state.bundle["defaultdataset"] = deterministic_standardize_defaultdataset(obj)
                st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                st.success("Applied dataset edits (normalized).")
                st.rerun()
        with cB:
            st.download_button(
                "Download defaultdataset.json",
                data=json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="defaultdataset.json",
                mime="application/json",
            )

    with c2:
        gd_text = st.text_area(
            "defaultguide.md (editable)",
            value=st.session_state.bundle["defaultguide"] or "",
            height=440,
            key="gd_editor",
        )
        cC, cD = st.columns(2)
        with cC:
            if st.button("Apply guide edits"):
                st.session_state.bundle["defaultguide"] = normalize_md(gd_text)
                st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                st.success("Applied guide edits.")
                st.rerun()
        with cD:
            st.download_button(
                "Download defaultguide.md",
                data=(st.session_state.bundle["defaultguide"] or "").encode("utf-8"),
                file_name="defaultguide.md",
                mime="text/markdown",
            )


def run_any_agent_on_results_ui():
    st.markdown("### Run any agent on current results")
    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    if not agents:
        st.warning("No agents available. Upload/standardize agents.yaml in Agents+Skills Studio.")
        return

    target = st.radio("Target", ["defaultguide.md", "defaultdataset.json", "Both (concatenate)"], horizontal=True)
    agent_id = st.selectbox("Agent", agents, index=0)
    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]))
    max_tokens = st.number_input("max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000)
    prompt = st.text_area(
        "User prompt (prepended to target content)",
        height=140,
        value="Analyze and improve the content. Do not invent facts; mark TBD where needed."
    )

    if target == "defaultguide.md":
        content = st.session_state.bundle["defaultguide"] or ""
    elif target == "defaultdataset.json":
        content = json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2)
    else:
        content = (
            "=== defaultguide.md ===\n" + (st.session_state.bundle["defaultguide"] or "") +
            "\n\n=== defaultdataset.json ===\n" + json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2)
        )

    if st.button("Run agent on target"):
        out = run_agent(
            agent_id=agent_id,
            user_prompt=prompt + "\n\n---\n\n" + content,
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(st.session_state.settings["temperature"])
        )
        st.session_state.history.append({"ts": now_iso(), "agent": agent_id, "model": model, "target": target})
        st.text_area("Agent output (editable)", value=out, height=260, key="any_agent_out")

        st.markdown("Overwrite options")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Overwrite current defaultguide.md with output"):
                st.session_state.bundle["defaultguide"] = normalize_md(out)
                st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                st.success("Overwritten defaultguide.md")
                st.rerun()
        with colB:
            if st.button("Try overwrite defaultdataset.json with output (JSON parse)"):
                try:
                    obj = safe_json_loads(out)
                    st.session_state.bundle["defaultdataset"] = deterministic_standardize_defaultdataset(obj)
                    st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                    st.success("Overwritten defaultdataset.json")
                    st.rerun()
                except Exception as e:
                    st.error(f"Output is not valid JSON: {e}")


# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "1) Dataset+Guide Studio",
    "2) Mock Bundle Generator",
    "3) Multi-pack Combiner",
    "4) Guidance Ingestor → Bundle",
    "5) Agents+Skills Studio",
    "6) FDA Tool: Outline Builder",
    "7) FDA Tool: Harmonization Mapper",
    "8) FDA Tool: Plain Language + FAQ",
    "9) FDA Tool: Public Comment Analyzer",
    "10) Dashboard",
])


# ============================================================
# 1) Dataset+Guide Studio
# ============================================================
with tabs[0]:
    st.markdown("## Dataset+Guide Studio")
    st.caption("Upload defaultdataset.json & defaultguide.md. If not standardized, system will standardize then allow edit + download.")

    col1, col2 = st.columns(2)

    with col1:
        up_ds = st.file_uploader("Upload defaultdataset.json", type=["json"], key="up_ds")
        ds_model = st.selectbox("Standardizer model (dataset)", ALL_MODELS, index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0)

        if st.button("Load + Standardize dataset", disabled=(up_ds is None)):
            raw = up_ds.getvalue().decode("utf-8", errors="ignore")
            try:
                obj = safe_json_loads(raw)
                obj = deterministic_standardize_defaultdataset(obj)
                if not is_standard_defaultdataset(obj):
                    obj = standardize_dataset_with_agent(raw, model=ds_model, max_tokens=12000)
                obj = deterministic_standardize_defaultdataset(obj)
                st.session_state.bundle["defaultdataset"] = obj
                st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                st.success("Dataset loaded + standardized.")
                st.rerun()
            except Exception as e:
                st.error(f"Dataset load failed: {e}")

    with col2:
        up_gd = st.file_uploader("Upload defaultguide.md (md/txt)", type=["md", "txt"], key="up_gd")
        gd_model = st.selectbox("Standardizer model (guide)", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0)

        if st.button("Load + Standardize guide", disabled=(up_gd is None)):
            raw = up_gd.getvalue().decode("utf-8", errors="ignore")
            try:
                md = normalize_md(raw)
                if not is_standard_defaultguide(md):
                    md = standardize_guide_with_agent(md, model=gd_model, max_tokens=12000)
                st.session_state.bundle["defaultguide"] = normalize_md(md)
                st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                st.success("Guide loaded + standardized.")
                st.rerun()
            except Exception as e:
                st.error(f"Guide load failed: {e}")

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()


# ============================================================
# 2) Mock Bundle Generator
# ============================================================
with tabs[1]:
    st.markdown("## Mock Bundle Generator")
    st.caption("Give instructions to generate new mock defaultdataset.json + defaultguide.md (standardized).")

    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]), key="gen_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000, key="gen_mt")

    prompt = st.text_area(
        "Instructions (prompt)",
        height=220,
        value=(
            "請用繁體中文產生一組 mock bundle：\n"
            "1) defaultdataset.json 需包含：tw_cases（至少 2 組資料集）與 k510_checklists（至少 1 組清單）\n"
            "2) defaultguide.md 需包含：tw_ 與 k510_ sections（至少各 1 段），並使用 BEGIN_SECTION 格式\n"
            "3) 每組 dataset 請給 2-3 筆案例，內容要明確標示為範例/合成\n"
            "4) 若你需要引用標準/法規但來源未提供，請標示 TBD，不可捏造。\n"
            "輸出格式必須是 JSON：{defaultdataset_json:..., defaultguide_md:'...'}"
        ),
        key="gen_prompt"
    )

    if st.button("Generate mock bundle"):
        cfg = agent_cfg("mock_bundle_generator")
        sys_p = cfg.get("system_prompt", "")
        out = call_llm(
            model=model,
            system_prompt=sys_p,
            user_prompt=prompt.strip(),
            max_tokens=int(max_tokens),
            temperature=float(st.session_state.settings["temperature"])
        )

        try:
            obj = safe_json_loads(out)
            ds = obj.get("defaultdataset_json", {})
            gd = obj.get("defaultguide_md", "")

            ds = deterministic_standardize_defaultdataset(ds, meta={"model": model, "prompt": prompt, "generated_at": now_iso()})
            gd = normalize_md(gd)
            if not is_standard_defaultguide(gd):
                gd = standardize_guide_with_agent(gd, model="gemini-2.5-flash" if "gemini-2.5-flash" in ALL_MODELS else model, max_tokens=12000)

            st.session_state.bundle["defaultdataset"] = ds
            st.session_state.bundle["defaultguide"] = gd
            st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()

            # keep prompt on results
            st.session_state.bundle["saved_prompts"].append({"ts": now_iso(), "module": "Mock Bundle Generator", "model": model, "prompt": prompt})

            st.success("Generated + standardized bundle loaded into editors (prompt saved).")
            st.rerun()
        except Exception as e:
            st.error(f"Generator output parse failed: {e}")
            st.text_area("Raw output", value=out, height=260)

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()


# ============================================================
# 3) Multi-pack Combiner
# ============================================================
with tabs[2]:
    st.markdown("## Multi-pack Combiner")
    st.caption("Upload multiple defaultdataset.json and defaultguide.md, standardize each, then combine into one.")

    up_ds_multi = st.file_uploader("Upload multiple defaultdataset.json", type=["json"], accept_multiple_files=True, key="up_ds_multi")
    up_gd_multi = st.file_uploader("Upload multiple defaultguide.md", type=["md", "txt"], accept_multiple_files=True, key="up_gd_multi")

    model_ds = st.selectbox("Dataset standardizer model", ALL_MODELS, index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0, key="cmb_ds_model")
    model_gd = st.selectbox("Guide standardizer model", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0, key="cmb_gd_model")

    def combine_datasets(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = {"tw_cases": {}, "k510_checklists": {}, "meta": {"generated_at": now_iso(), "generated_by": {"tool": "combiner"}}}
        suffix = 2
        for o in objs:
            o = deterministic_standardize_defaultdataset(o)
            for k, v in (o.get("tw_cases") or {}).items():
                kk = k
                while kk in merged["tw_cases"]:
                    kk = f"{k}__merge{suffix}"
                    suffix += 1
                merged["tw_cases"][kk] = v
            for k, v in (o.get("k510_checklists") or {}).items():
                kk = k
                while kk in merged["k510_checklists"]:
                    kk = f"{k}__merge{suffix}"
                    suffix += 1
                merged["k510_checklists"][kk] = v
        return merged

    def combine_guides(mds: List[str]) -> str:
        all_sections = []
        used_ids = set()
        suffix = 2
        for md in mds:
            md = normalize_md(md)
            if not is_standard_defaultguide(md):
                md = standardize_guide_with_agent(md, model=model_gd, max_tokens=12000)
            secs = parse_defaultguide_sections(md)
            for s in secs:
                sid = s["id"]
                while sid in used_ids:
                    sid = f"{s['id']}__merge{suffix}"
                    suffix += 1
                used_ids.add(sid)
                all_sections.append({"id": sid, "title": s["title"], "md": s["md"]})
        return build_defaultguide_from_sections(all_sections)

    if st.button("Standardize + Combine"):
        ds_objs = []
        gd_mds = []

        for f in (up_ds_multi or []):
            raw = f.getvalue().decode("utf-8", errors="ignore")
            try:
                obj = safe_json_loads(raw)
                obj = deterministic_standardize_defaultdataset(obj)
                if not is_standard_defaultdataset(obj):
                    obj = standardize_dataset_with_agent(raw, model=model_ds, max_tokens=12000)
                ds_objs.append(deterministic_standardize_defaultdataset(obj))
            except Exception as e:
                st.error(f"Dataset '{f.name}' failed: {e}")

        for f in (up_gd_multi or []):
            gd_mds.append(normalize_md(f.getvalue().decode("utf-8", errors="ignore")))

        if ds_objs:
            st.session_state.bundle["defaultdataset"] = combine_datasets(ds_objs)
        if gd_mds:
            st.session_state.bundle["defaultguide"] = combine_guides(gd_mds)

        st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
        st.session_state.bundle["saved_prompts"].append({"ts": now_iso(), "module": "Multi-pack Combiner", "model": None, "prompt": "Standardize + Combine"})
        st.success("Combined bundle loaded into editors.")
        st.rerun()

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()


# ============================================================
# 4) Guidance Ingestor → Bundle (PDF page range + OCR + preview) (NEW)
# ============================================================
with tabs[3]:
    st.markdown("## Guidance Ingestor → Bundle Builder")
    st.caption("Paste/upload multiple guidance docs (txt/md/pdf). Preview PDFs, choose page ranges, optional OCR, then build bundle.")

    model = st.selectbox(
        "Model", ALL_MODELS,
        index=ALL_MODELS.index("claude-3-5-sonnet-20241022") if "claude-3-5-sonnet-20241022" in ALL_MODELS else 0,
        key="ing_model"
    )
    max_tokens = st.number_input("max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000, key="ing_mt")

    pasted = st.text_area("Paste guidance (optional; separate docs with '---')", height=160, key="ing_paste")
    uploads = st.file_uploader("Upload guidance files", type=["pdf", "md", "txt"], accept_multiple_files=True, key="ing_files")

    st.markdown("### Preview + Extraction Controls (PDF: page-range + OCR)")
    extracted_parts = []

    if pasted.strip():
        extracted_parts.append("=== PASTED ===\n" + pasted.strip())

    if uploads:
        for i, f in enumerate(uploads):
            name = f.name
            suffix = name.lower().rsplit(".", 1)[-1]
            with st.expander(f"{name}", expanded=False):
                if suffix == "pdf":
                    pdf_bytes = f.getvalue()
                    n = pdf_page_count(pdf_bytes)
                    st.write({"pages": n})

                    show_pdf_bytes(pdf_bytes, height=520)

                    cA, cB, cC = st.columns([1.0, 1.0, 1.2])
                    with cA:
                        start_p = st.number_input(f"From page ({name})", 1, max(1, n), 1, 1, key=f"pdf_from_{i}")
                    with cB:
                        end_p = st.number_input(f"To page ({name})", 1, max(1, n), min(3, n) if n else 1, 1, key=f"pdf_to_{i}")
                    with cC:
                        use_ocr = st.checkbox(f"OCR this range ({name})", value=False, key=f"pdf_ocr_{i}")

                    if st.button(f"Extract selected pages text ({name})", key=f"pdf_extract_btn_{i}"):
                        if use_ocr:
                            text = ocr_pdf_pages_text(pdf_bytes, start_p, end_p, lang="eng+chi_tra")
                        else:
                            text = extract_pdf_pages_text(pdf_bytes, start_p, end_p)
                        st.text_area("Extracted text (preview)", value=text, height=220, key=f"pdf_extracted_preview_{i}")
                        # Persist extraction to session for build step
                        st.session_state[f"pdf_extracted_text_{i}"] = text

                    # if already extracted, show it
                    prev = st.session_state.get(f"pdf_extracted_text_{i}", "")
                    if prev.strip():
                        st.markdown("**Current extracted text used for ingestion:**")
                        st.text_area("Used text", value=prev, height=180, key=f"pdf_used_text_{i}")

                else:
                    text = normalize_md(f.getvalue().decode("utf-8", errors="ignore"))
                    st.markdown(text[:4000] + ("\n\n...(truncated preview)" if len(text) > 4000 else ""))
                    extracted_parts.append(f"=== FILE: {name} ===\n{text}")

    # collect extracted texts from PDFs (if any)
    if uploads:
        for i, f in enumerate(uploads):
            if f.name.lower().endswith(".pdf"):
                text = st.session_state.get(f"pdf_extracted_text_{i}", "")
                if text.strip():
                    extracted_parts.append(f"=== FILE: {f.name} (PDF extracted range) ===\n{text}")

    prompt = st.text_area(
        "Ingestor prompt (editable; will be saved on results)",
        height=180,
        value=(
            "請將以下多份 guidance 內容彙整為標準 defaultguide.md（BEGIN_SECTION 格式，section id 以 tw_/k510_ 開頭）。\n"
            "並基於 guidance 主題產生 mock defaultdataset.json：\n"
            "- tw_cases：至少 1 組資料集、每組 2 筆案例（合成示例）\n"
            "- k510_checklists：至少 1 組 checklist，至少 8 個 items\n"
            "Harmonization 用語請標示 TBD，不可捏造官方引用。\n"
            "輸出格式必須是 JSON：{defaultdataset_json:..., defaultguide_md:'...'}"
        ),
        key="ing_prompt"
    )

    if st.button("Build bundle from extracted guidances"):
        raw_all = "\n\n---\n\n".join([p for p in extracted_parts if p.strip()]).strip()
        if not raw_all:
            st.warning("No guidance content available. Paste text or extract from files first.")
        else:
            cfg = agent_cfg("guidance_ingestor_to_bundle")
            sys_p = cfg.get("system_prompt", "")
            user_p = prompt.strip() + "\n\n---\n\n" + raw_all
            out = call_llm(
                model=model,
                system_prompt=sys_p,
                user_prompt=user_p,
                max_tokens=int(max_tokens),
                temperature=float(st.session_state.settings["temperature"])
            )
            try:
                obj = safe_json_loads(out)
                ds = deterministic_standardize_defaultdataset(
                    obj.get("defaultdataset_json", {}),
                    meta={"model": model, "prompt": prompt, "generated_at": now_iso(), "module": "Guidance Ingestor"}
                )
                gd = normalize_md(obj.get("defaultguide_md", ""))

                if not is_standard_defaultguide(gd):
                    gd = standardize_guide_with_agent(
                        gd,
                        model="gemini-2.5-flash" if "gemini-2.5-flash" in ALL_MODELS else model,
                        max_tokens=12000
                    )

                st.session_state.bundle["defaultdataset"] = ds
                st.session_state.bundle["defaultguide"] = gd
                st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                st.session_state.bundle["saved_prompts"].append({"ts": now_iso(), "module": "Guidance Ingestor", "model": model, "prompt": prompt})

                st.success("Bundle built + standardized, loaded into editors (prompt saved).")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to parse model output: {e}")
                st.text_area("Raw output", value=out, height=260)

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()


# ============================================================
# 5) Agents+Skills Studio (SKILL is used for injection) (NEW)
# ============================================================
with tabs[4]:
    st.markdown("## Agents+Skills Studio")
    st.caption("Upload/standardize/edit/download agents.yaml and SKILL.md. SKILL.md can be auto-injected into agents.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### agents.yaml")
        up_agents = st.file_uploader("Upload agents.yaml (yaml/yml)", type=["yaml", "yml"], key="up_agents")
        std_model = st.selectbox("Agents standardizer model", ALL_MODELS, index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0, key="agents_std_model")

        if st.button("Load agents.yaml (standardize if needed)", disabled=(up_agents is None)):
            raw = up_agents.getvalue().decode("utf-8", errors="ignore")
            try:
                obj = yaml.safe_load(raw) or {}
                if not (isinstance(obj, dict) and isinstance(obj.get("agents"), dict) and obj["agents"]):
                    obj = standardize_agents_yaml_with_agent(raw, model=std_model, max_tokens=8000)
                st.session_state.agents_cfg = obj
                st.success("agents.yaml loaded (and standardized if needed).")
                st.rerun()
            except Exception as e:
                st.error(f"agents.yaml load failed: {e}")

        agents_text = st.text_area(
            "agents.yaml (editable)",
            value=yaml.dump(st.session_state.agents_cfg, allow_unicode=True, sort_keys=False),
            height=420,
            key="agents_editor"
        )
        colA, colB = st.columns(2)
        with colA:
            if st.button("Apply agents.yaml edits to session"):
                try:
                    obj = yaml.safe_load(agents_text) or {}
                    if not (isinstance(obj, dict) and "agents" in obj and isinstance(obj["agents"], dict)):
                        st.error("Invalid schema: missing top-level 'agents' dict.")
                    else:
                        st.session_state.agents_cfg = obj
                        st.success("Applied agents.yaml to session.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Parse error: {e}")
        with colB:
            st.download_button("Download agents.yaml", data=agents_text.encode("utf-8"), file_name="agents.yaml", mime="text/yaml")

    with c2:
        st.markdown("### SKILL.md (shared knowledge injected into agents)")
        up_skill = st.file_uploader("Upload SKILL.md", type=["md", "txt"], key="up_skill")
        if st.button("Load SKILL.md", disabled=(up_skill is None)):
            st.session_state.skill_md = up_skill.getvalue().decode("utf-8", errors="ignore")
            st.success("Loaded SKILL.md into editor.")
            st.rerun()

        skill_text = st.text_area("SKILL.md (editable)", value=st.session_state.skill_md, height=420, key="skill_editor")
        colC, colD = st.columns(2)
        with colC:
            if st.button("Apply SKILL.md edits"):
                st.session_state.skill_md = skill_text
                st.success("Applied SKILL.md edits in session.")
                st.rerun()
        with colD:
            st.download_button("Download SKILL.md", data=skill_text.encode("utf-8"), file_name="SKILL.md", mime="text/markdown")


# ============================================================
# 6) FDA Tool: Outline Builder → one-click to defaultguide section (NEW)
# ============================================================
with tabs[5]:
    st.markdown("## FDA Guidance Tool: Outline Builder")
    st.caption("Generate a guidance outline. Then one-click convert into defaultguide.md section format and append to current defaultguide.md.")

    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    default_agent = "fda_guidance_outline_builder" if "fda_guidance_outline_builder" in agents else (agents[0] if agents else "")
    agent_id = st.selectbox("Agent", agents, index=agents.index(default_agent) if default_agent in agents else 0, key="fda1_agent")

    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gpt-4.1-mini") if "gpt-4.1-mini" in ALL_MODELS else 0, key="fda1_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, 12000, 1000, key="fda1_mt")

    prompt = st.text_area(
        "Prompt",
        height=220,
        value=(
            "請為一份 FDA guidance 產生詳細大綱（非官方、僅作為草稿協助）：\n"
            "- 主題：\n- 適用裝置/範圍：\n- 目標讀者：\n- 主要風險與證據期待：\n"
            "輸出包含：章節結構、每章目的、需要的 evidence 類型（bench/software/cyber/biocompat/clinical 等）、TBD 標記。\n"
        ),
        key="fda1_prompt"
    )

    if st.button("Generate outline"):
        out = run_agent(
            agent_id=agent_id,
            user_prompt=prompt,
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(st.session_state.settings["temperature"])
        )
        st.session_state["fda_outline_out"] = out
        st.text_area("Outline output (editable)", value=out, height=380, key="fda1_out")

    outline_out = st.session_state.get("fda_outline_out", "")
    if outline_out.strip():
        st.markdown("### One-click: Convert outline → defaultguide.md section")
        sec_id = st.text_input("Section ID (must start with tw_ or k510_)", value="k510_fda_guidance_outline_v1", key="fda1_sec_id")
        sec_title = st.text_input("Section title", value="（FDA）Guidance Outline（草稿）", key="fda1_sec_title")

        converter_model = st.selectbox(
            "Converter model (guide standardizer)",
            ALL_MODELS,
            index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0,
            key="fda1_conv_model"
        )

        if st.button("Convert & append to current defaultguide.md"):
            # Use guide standardizer to ensure BEGIN_SECTION format, but allow deterministic wrap if needed.
            raw_section = f"<!-- BEGIN_SECTION: {sec_id} | TITLE: {sec_title} -->\n{outline_out}\n<!-- END_SECTION -->"
            md = normalize_md(raw_section)

            # If section id invalid, still try standardizer (it will fix ids)
            if not is_standard_defaultguide(md):
                md = standardize_guide_with_agent(md, model=converter_model, max_tokens=12000)

            # Append to existing guide (merge sections, avoid ID conflict)
            existing = st.session_state.bundle["defaultguide"] or ""
            existing_secs = parse_defaultguide_sections(existing) if existing.strip() else []
            new_secs = parse_defaultguide_sections(md)

            used = {s["id"] for s in existing_secs}
            suffix = 2
            for s in new_secs:
                sid = s["id"]
                while sid in used:
                    sid = f"{s['id']}__outline{suffix}"
                    suffix += 1
                used.add(sid)
                existing_secs.append({"id": sid, "title": s["title"], "md": s["md"]})

            st.session_state.bundle["defaultguide"] = build_defaultguide_from_sections(existing_secs)
            st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
            st.success("Appended outline as standardized section into defaultguide.md")
            st.rerun()


# ============================================================
# 7) FDA Tool: Harmonization Mapper with fixed table schema (NEW)
# ============================================================
with tabs[6]:
    st.markdown("## FDA Guidance Tool: Harmonization & Standards Mapper")
    st.caption("Produces fixed-column mapping table suitable for dashboards + counts by Status.")

    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    default_agent = "fda_harmonization_mapper" if "fda_harmonization_mapper" in agents else (agents[0] if agents else "")
    agent_id = st.selectbox("Agent", agents, index=agents.index(default_agent) if default_agent in agents else 0, key="fda2_agent")

    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0, key="fda2_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, 12000, 1000, key="fda2_mt")

    draft = st.text_area("Paste guidance draft (markdown/text)", height=220, key="fda2_draft")
    prompt = st.text_area(
        "Prompt (fixed table required)",
        height=160,
        value=(
            "請分析此 guidance 草稿，輸出必須包含以下固定欄位之 Markdown 表格（不可更動欄名）：\n"
            "| Standard/Citation | Clause/Section | Guidance Section Ref | Evidence Expected | Status | Notes/Action |\n"
            "其中 Status 僅能用：Pass / Concern / Gap / TBD。\n"
            "規則：不得捏造引用；若草稿未提供 citation，Standard/Citation 寫 TBD。\n"
            "另外再輸出：Consistency Checklist（條列）與 Gaps Summary（條列）。"
        ),
        key="fda2_prompt"
    )

    if st.button("Run harmonization mapping"):
        out = run_agent(
            agent_id=agent_id,
            user_prompt=prompt + "\n\n---\n\n" + draft,
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(st.session_state.settings["temperature"])
        )
        st.session_state["fda_harmon_out"] = out

    out = st.session_state.get("fda_harmon_out", "")
    if out.strip():
        st.text_area("Mapping output (editable)", value=out, height=320, key="fda2_out")

        st.markdown("### Dashboard-ready table extraction")
        df = extract_first_markdown_table(out)
        if df is None:
            st.warning("Could not parse a markdown table. Ensure the output contains a pipe table with the required header.")
        else:
            st.dataframe(df, use_container_width=True)

            # Status chart if column exists
            if "Status" in df.columns:
                vc = df["Status"].fillna("").astype(str).str.strip().replace("", "TBD")
                stats = vc.value_counts().reset_index()
                stats.columns = ["Status", "Count"]

                chart = alt.Chart(stats).mark_bar().encode(
                    x=alt.X("Status:N", sort="-y"),
                    y="Count:Q",
                    color="Status:N",
                    tooltip=["Status", "Count"]
                )
                st.altair_chart(chart, use_container_width=True)


# ============================================================
# 8) FDA Tool: Plain Language + FAQ + Change Tracking Table (NEW)
# ============================================================
with tabs[7]:
    st.markdown("## FDA Guidance Tool: Plain Language + FAQ")
    st.caption("Rewrite into plain language + FAQs + glossary, and include a change-tracking table (Original → New → Rationale).")

    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    default_agent = "fda_plain_language_rewriter" if "fda_plain_language_rewriter" in agents else (agents[0] if agents else "")
    agent_id = st.selectbox("Agent", agents, index=agents.index(default_agent) if default_agent in agents else 0, key="fda3_agent")

    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("grok-4-fast-reasoning") if "grok-4-fast-reasoning" in ALL_MODELS else 0, key="fda3_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, 12000, 1000, key="fda3_mt")

    draft = st.text_area("Paste technical guidance draft", height=220, key="fda3_draft")
    prompt = st.text_area(
        "Prompt (must include change-tracking table)",
        height=170,
        value=(
            "請輸出以下三個區塊（以 Markdown 標題分隔）：\n"
            "## A) Plain-Language Rewrite\n"
            "- 將草稿改寫成一般大眾可理解的版本，保留原意，不可新增規範。\n\n"
            "## B) Change Tracking Table\n"
            "請輸出 Markdown 表格，欄位固定：| Original | New | Rationale |\n"
            "- Original/New 請用短句或片段（避免整篇貼上），Rationale 說明為何這樣改（例如更清楚/更一致/避免誤解）。\n\n"
            "## C) FAQ + Glossary\n"
            "- FAQ 10–15 題\n"
            "- Glossary（名詞解釋）\n"
            "不得捏造要求；不確定處標示 TBD。"
        ),
        key="fda3_prompt"
    )

    if st.button("Rewrite + FAQ + Change Tracking"):
        out = run_agent(
            agent_id=agent_id,
            user_prompt=prompt + "\n\n---\n\n" + draft,
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(st.session_state.settings["temperature"])
        )
        st.session_state["fda_plain_out"] = out

    out = st.session_state.get("fda_plain_out", "")
    if out.strip():
        st.text_area("Output (editable)", value=out, height=380, key="fda3_out")


# ============================================================
# 9) FDA Tool: Public Comment Analyzer (NEW)
# ============================================================
with tabs[8]:
    st.markdown("## FDA Guidance Tool: Public Comment Analyzer")
    st.caption("Upload comment CSV → classify themes/sentiment/priority and propose responses. Includes basic dashboard charts.")

    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]), key="pc_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, 12000, 1000, key="pc_mt")

    up = st.file_uploader("Upload comments CSV", type=["csv"], key="pc_csv")
    if up is not None:
        df = pd.read_csv(up).fillna("")
        st.markdown("### Raw comments preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("### Column mapping")
        cols = list(df.columns)
        text_col = st.selectbox("Comment text column", cols, index=0, key="pc_text_col")
        id_col = st.selectbox("Comment ID column (optional)", ["(none)"] + cols, index=0, key="pc_id_col")
        author_col = st.selectbox("Commenter/Author column (optional)", ["(none)"] + cols, index=0, key="pc_author_col")
        date_col = st.selectbox("Date column (optional)", ["(none)"] + cols, index=0, key="pc_date_col")

        st.markdown("### Analyzer prompt")
        prompt = st.text_area(
            "Prompt (JSON output required)",
            height=200,
            value=(
                "你是一位 FDA guidance 公眾意見分析助手。請對每則 comment 做分類與建議回覆。\n"
                "輸出必須是 JSON，格式：\n"
                "{\n"
                '  "summary": {\n'
                '    "themes": [{"theme":"...","count":1,"notes":"..."}],\n'
                '    "top_risks": ["..."],\n'
                '    "recommended_revisions": ["..."]\n'
                "  },\n"
                '  "items": [\n'
                "    {\n"
                '      "comment_id":"...",\n'
                '      "theme":"...",\n'
                '      "sentiment":"support|neutral|concern|oppose",\n'
                '      "priority":"high|medium|low",\n'
                '      "requested_change":"...",\n'
                '      "suggested_response":"..."\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "規則：\n"
                "- 不可捏造官方立場；回覆以中性、感謝、說明、TBD/將考量為主。\n"
                "- theme 請盡量歸一化（例如『定義不清』『證據要求』『資安』『軟體文件』『臨床資料』『過度負擔』等）。\n"
            ),
            key="pc_prompt"
        )

        sample_n = st.number_input("Analyze first N comments (for cost control)", 1, min(5000, len(df)), min(50, len(df)), 1, key="pc_n")

        if st.button("Analyze comments with AI"):
            items = []
            for idx in range(int(sample_n)):
                row = df.iloc[idx]
                cid = str(row[id_col]).strip() if id_col != "(none)" else str(idx)
                author = str(row[author_col]).strip() if author_col != "(none)" else ""
                dt = str(row[date_col]).strip() if date_col != "(none)" else ""
                txt = str(row[text_col]).strip()

                items.append({
                    "comment_id": cid,
                    "author": author,
                    "date": dt,
                    "text": txt
                })

            payload = {
                "context": {
                    "document": "FDA Guidance Public Comments",
                    "language": "Traditional Chinese",
                    "analyze_count": len(items)
                },
                "comments": items
            }

            user_prompt = prompt + "\n\n---\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)

            # Use a dedicated agent if present; else run direct as a generic call with empty system prompt.
            agent_id = "public_comment_analyzer" if "public_comment_analyzer" in (st.session_state.agents_cfg.get("agents") or {}) else None
            try:
                if agent_id:
                    out = run_agent(agent_id, user_prompt=user_prompt, model=model, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))
                else:
                    out = call_llm(model=model, system_prompt="You analyze public comments for a guidance draft.", user_prompt=user_prompt, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))

                st.session_state["pc_out_raw"] = out
            except Exception as e:
                st.error(f"AI analysis failed: {e}")

        out_raw = st.session_state.get("pc_out_raw", "")
        if out_raw.strip():
            st.markdown("### AI Output (raw)")
            st.text_area("Raw JSON output (editable)", value=out_raw, height=260, key="pc_out_raw_editor")

            st.markdown("### Parsed dashboard")
            try:
                obj = safe_json_loads(out_raw)
                summary = obj.get("summary", {})
                items = obj.get("items", [])

                st.markdown("#### Summary")
                st.json(summary)

                if isinstance(items, list) and items:
                    df_items = pd.DataFrame(items).fillna("")
                    st.dataframe(df_items, use_container_width=True)

                    # Charts
                    c1, c2 = st.columns(2)
                    with c1:
                        if "sentiment" in df_items.columns:
                            s = df_items["sentiment"].astype(str).value_counts().reset_index()
                            s.columns = ["sentiment", "count"]
                            chart = alt.Chart(s).mark_bar().encode(
                                x=alt.X("sentiment:N", sort="-y"),
                                y="count:Q",
                                color="sentiment:N",
                                tooltip=["sentiment", "count"]
                            )
                            st.altair_chart(chart, use_container_width=True)

                    with c2:
                        if "priority" in df_items.columns:
                            p = df_items["priority"].astype(str).value_counts().reset_index()
                            p.columns = ["priority", "count"]
                            chart = alt.Chart(p).mark_bar().encode(
                                x=alt.X("priority:N", sort="-y"),
                                y="count:Q",
                                color="priority:N",
                                tooltip=["priority", "count"]
                            )
                            st.altair_chart(chart, use_container_width=True)

                    st.download_button(
                        "Download analysis items.csv",
                        data=df_items.to_csv(index=False).encode("utf-8"),
                        file_name="public_comment_analysis.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.warning(f"Could not parse JSON for dashboard: {e}")


# ============================================================
# 10) Dashboard
# ============================================================
with tabs[9]:
    st.markdown("## Dashboard")
    st.caption("Session activity, standard checks, prompt memory, quick diff.")

    ds_ok = is_standard_defaultdataset(st.session_state.bundle["defaultdataset"])
    gd_ok = is_standard_defaultguide(st.session_state.bundle["defaultguide"] or "")
    st.markdown("### Bundle status")
    st.write({
        "defaultdataset_standard": ds_ok,
        "defaultguide_standard": gd_ok,
        "last_updated": st.session_state.bundle["bundle_meta"]["last_updated"],
        "skill_injection_enabled": bool(st.session_state.settings.get("inject_skill", True)),
        "skill_chars_injected": min(len(st.session_state.skill_md or ""), int(st.session_state.settings.get("inject_skill_max_chars", 6000))),
    })

    st.markdown("### Saved prompts (Keep prompt on results)")
    sp = st.session_state.bundle.get("saved_prompts", [])
    if sp:
        st.dataframe(pd.DataFrame(sp), use_container_width=True)
    else:
        st.info("No saved prompts yet (generator/ingestor/combiner will save).")

    st.markdown("### Quick diff helper")
    c1, c2 = st.columns(2)
    with c1:
        a = st.text_area("A text", height=160, key="diff_a")
    with c2:
        b = st.text_area("B text", height=160, key="diff_b")
    if st.button("Show diff"):
        st.code(diff_text(a, b), language="diff")

    st.markdown("### Run history")
    st.dataframe(
        pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame(columns=["ts", "agent", "model", "target"]),
        use_container_width=True
    )
