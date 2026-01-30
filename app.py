import os, json, re, base64, difflib
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import yaml
import pandas as pd
from pypdf import PdfReader

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Antigravity Agentic Data+Guidance Studio", layout="wide", initial_sidebar_state="expanded")


# -------------------------
# Models
# -------------------------
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
    if model in OPENAI_MODELS: return "openai"
    if model in GEMINI_MODELS: return "gemini"
    if model in ANTHROPIC_MODELS: return "anthropic"
    if model in GROK_MODELS: return "grok"
    raise ValueError(f"Unknown model: {model}")


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


def call_llm(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
    provider = get_provider(model)
    key = get_api_key(provider)
    if not key:
        raise RuntimeError(f"Missing API key for provider: {provider}")

    if provider == "openai":
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system_prompt or ""},{"role":"user","content":user_prompt or ""}],
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
            messages=[{"role":"user","content":user_prompt or ""}],
        )
        return resp.content[0].text

    if provider == "grok":
        with httpx.Client(base_url="https://api.x.ai/v1", timeout=120) as client:
            resp = client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role":"system","content":system_prompt or ""},{"role":"user","content":user_prompt or ""}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError("Unsupported provider")


# -------------------------
# Utilities
# -------------------------
SECTION_RE = re.compile(
    r"<!--\s*BEGIN_SECTION:\s*(.*?)\s*\|\s*TITLE:\s*(.*?)\s*-->(.*?)<!--\s*END_SECTION\s*-->",
    re.DOTALL
)

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def normalize_md(md: str) -> str:
    md = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md

def show_pdf_bytes(pdf_bytes: bytes, height: int = 600):
    if not pdf_bytes:
        return
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}"></iframe>""", unsafe_allow_html=True)

def extract_pdf_text(file) -> str:
    try:
        reader = PdfReader(file)
        texts = []
        for p in reader.pages:
            texts.append(p.extract_text() or "")
        return normalize_md("\n\n".join(texts))
    except Exception as e:
        return normalize_md(f"[System] PDF extraction failed: {e}")

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
        body = normalize_md(s.get("md",""))
        blocks.append(f"<!-- BEGIN_SECTION: {sid} | TITLE: {title} -->\n{body}\n<!-- END_SECTION -->")
    return normalize_md("\n\n\n".join(blocks))

def diff_text(a: str, b: str) -> str:
    a_lines = (a or "").splitlines(keepends=True)
    b_lines = (b or "").splitlines(keepends=True)
    return "".join(difflib.unified_diff(a_lines, b_lines, fromfile="A", tofile="B")).strip()

def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        # best effort extraction
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise

def is_standard_defaultdataset(obj: Any) -> bool:
    if not isinstance(obj, dict): return False
    if "tw_cases" not in obj or "k510_checklists" not in obj:
        return False
    if not isinstance(obj["tw_cases"], dict) or not isinstance(obj["k510_checklists"], dict):
        return False
    # loose validation of internal entries
    for _, v in obj["tw_cases"].items():
        if not isinstance(v, dict): return False
        if "title" not in v or "cases" not in v: return False
        if not isinstance(v["cases"], list): return False
    for _, v in obj["k510_checklists"].items():
        if not isinstance(v, dict): return False
        if "title" not in v or "items" not in v: return False
        if not isinstance(v["items"], list): return False
    return True

def deterministic_standardize_defaultdataset(obj: Any, meta: Optional[dict]=None) -> Dict[str, Any]:
    meta = meta or {}
    out = {"tw_cases": {}, "k510_checklists": {}, "meta": {"generated_at": now_iso(), "generated_by": meta}}
    if isinstance(obj, dict):
        tw = obj.get("tw_cases", {}) if isinstance(obj.get("tw_cases", {}), dict) else {}
        k510 = obj.get("k510_checklists", {}) if isinstance(obj.get("k510_checklists", {}), dict) else {}
        out["tw_cases"] = tw
        out["k510_checklists"] = k510
        # preserve meta if present
        if isinstance(obj.get("meta"), dict):
            out["meta"] = obj["meta"]
            out["meta"].setdefault("generated_at", now_iso())
            out["meta"].setdefault("generated_by", meta)
    return out

def is_standard_defaultguide(md: str) -> bool:
    sections = parse_defaultguide_sections(md)
    if not sections:
        return False
    for s in sections:
        if not (s["id"].startswith("tw_") or s["id"].startswith("k510_")):
            return False
    return True

def load_agents_cfg() -> Dict[str, Any]:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {"agents": {}}
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
    system_prompt = cfg.get("system_prompt", "")
    user_prompt = f"RAW_DATASET_INPUT:\n{raw_text}"
    out = call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens, temperature=0.0)
    return safe_json_loads(out)

def standardize_guide_with_agent(raw_text: str, model: str, max_tokens: int) -> str:
    cfg = agent_cfg("guide_standardizer")
    system_prompt = cfg.get("system_prompt", "")
    user_prompt = f"RAW_GUIDE_INPUT:\n{raw_text}"
    out = call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens, temperature=0.0)
    return normalize_md(out)

def standardize_agents_yaml_with_agent(raw_text: str, model: str, max_tokens: int) -> Dict[str, Any]:
    cfg = agent_cfg("agents_yaml_standardizer")
    system_prompt = cfg.get("system_prompt", "")
    out = call_llm(model=model, system_prompt=system_prompt, user_prompt=raw_text, max_tokens=max_tokens, temperature=0.0)
    clean = out.replace("```yaml", "").replace("```", "").strip()
    return yaml.safe_load(clean) or {"agents": {}}


# -------------------------
# State init
# -------------------------
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {"openai":"", "gemini":"", "anthropic":"", "grok":""}

if "settings" not in st.session_state:
    st.session_state.settings = {
        "model": "gpt-4o-mini",
        "max_tokens": 12000,
        "temperature": 0.2,
    }

if "agents_cfg" not in st.session_state:
    st.session_state.agents_cfg = load_agents_cfg()

if "skill_md" not in st.session_state:
    try:
        with open("SKILL.md","r",encoding="utf-8") as f:
            st.session_state.skill_md = f.read()
    except Exception:
        st.session_state.skill_md = ""

if "bundle" not in st.session_state:
    st.session_state.bundle = {
        "defaultdataset": {"tw_cases": {}, "k510_checklists": {}, "meta": {"generated_at": now_iso(), "generated_by": {}}},
        "defaultguide": "",
        "bundle_meta": {"last_updated": now_iso()}
    }

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("## Global Settings")

    st.session_state.settings["model"] = st.selectbox("Default model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]))
    st.session_state.settings["max_tokens"] = st.number_input("Default max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000)
    st.session_state.settings["temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state.settings["temperature"]), 0.05)

    st.markdown("---")
    st.markdown("## API Keys")

    def api_key_row(label: str, env_var: str, provider: str):
        if env_key_present(env_var):
            st.caption(f"{label}: Active (Env)")
        else:
            st.session_state.api_keys[provider] = st.text_input(f"{label} API Key", value=st.session_state.api_keys[provider], type="password")

    api_key_row("OpenAI", "OPENAI_API_KEY", "openai")
    api_key_row("Gemini", "GEMINI_API_KEY", "gemini")
    api_key_row("Anthropic", "ANTHROPIC_API_KEY", "anthropic")
    api_key_row("Grok (xAI)", "GROK_API_KEY", "grok")

    st.markdown("---")
    st.markdown("## Config Files")
    if st.button("Reload agents.yaml from disk"):
        st.session_state.agents_cfg = load_agents_cfg()
        st.success("Reloaded agents.yaml")
        st.rerun()


# -------------------------
# Header
# -------------------------
st.markdown("# Antigravity Agentic Data+Guidance Studio")
st.caption("Upload → Standardize → Edit → Combine → Generate bundles (defaultdataset.json + defaultguide.md), plus FDA guidance authoring tools.")


# -------------------------
# Core UI helpers
# -------------------------
def bundle_editors():
    st.markdown("### Current Bundle Editors")

    c1, c2 = st.columns(2)
    with c1:
        ds_text = st.text_area(
            "defaultdataset.json (editable)",
            value=json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2),
            height=420,
            key="ds_editor",
        )
        if st.button("Apply dataset edits"):
            obj = safe_json_loads(ds_text)
            # deterministic normalize
            st.session_state.bundle["defaultdataset"] = deterministic_standardize_defaultdataset(obj)
            st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
            st.success("Applied dataset edits (normalized).")
            st.rerun()

        st.download_button(
            "Download defaultdataset.json",
            data=json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="defaultdataset.json",
            mime="application/json"
        )

    with c2:
        gd_text = st.text_area(
            "defaultguide.md (editable)",
            value=st.session_state.bundle["defaultguide"] or "",
            height=420,
            key="gd_editor",
        )
        if st.button("Apply guide edits"):
            st.session_state.bundle["defaultguide"] = normalize_md(gd_text)
            st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
            st.success("Applied guide edits.")
            st.rerun()

        st.download_button(
            "Download defaultguide.md",
            data=(st.session_state.bundle["defaultguide"] or "").encode("utf-8"),
            file_name="defaultguide.md",
            mime="text/markdown"
        )


def run_any_agent_on_results_ui():
    st.markdown("### Run any agent on current results")
    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    if not agents:
        st.warning("No agents available.")
        return

    target = st.radio("Target", ["defaultguide.md", "defaultdataset.json", "Both (concatenate)"], horizontal=True)
    agent_id = st.selectbox("Agent", agents, index=0)
    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]))
    max_tokens = st.number_input("max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000)
    prompt = st.text_area("User prompt (will be prepended to target content)", height=140, value="Analyze and improve the content. Do not invent facts.")

    if target == "defaultguide.md":
        content = st.session_state.bundle["defaultguide"] or ""
    elif target == "defaultdataset.json":
        content = json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2)
    else:
        content = "=== defaultguide.md ===\n" + (st.session_state.bundle["defaultguide"] or "") + "\n\n=== defaultdataset.json ===\n" + json.dumps(st.session_state.bundle["defaultdataset"], ensure_ascii=False, indent=2)

    if st.button("Run agent on target"):
        out = run_agent(agent_id, user_prompt=prompt + "\n\n---\n\n" + content, model=model, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))
        st.session_state.history.append({"ts": now_iso(), "agent": agent_id, "model": model, "target": target})
        st.text_area("Agent output (editable)", value=out, height=260, key="any_agent_out")

        # optional overwrite helpers
        st.markdown("Overwrite options")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Overwrite current defaultguide.md with output"):
                st.session_state.bundle["defaultguide"] = normalize_md(out)
                st.success("Overwritten defaultguide.md")
                st.rerun()
        with colB:
            if st.button("Try overwrite defaultdataset.json with output (JSON parse)"):
                try:
                    obj = safe_json_loads(out)
                    st.session_state.bundle["defaultdataset"] = deterministic_standardize_defaultdataset(obj)
                    st.success("Overwritten defaultdataset.json")
                    st.rerun()
                except Exception as e:
                    st.error(f"Output is not valid JSON: {e}")


# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "1) Dataset+Guide Studio",
    "2) Mock Bundle Generator",
    "3) Multi-pack Combiner",
    "4) Guidance Ingestor → Bundle",
    "5) Agents+Skills Studio",
    "6) FDA Tool: Outline Builder",
    "7) FDA Tool: Harmonization Mapper",
    "8) FDA Tool: Plain Language + FAQ",
    "9) Dashboard",
])

# 1) Upload/Standardize/Edit/Download
with tabs[0]:
    st.markdown("## Dataset+Guide Studio")
    st.caption("Upload defaultdataset.json & defaultguide.md. If not standardized, the system will standardize, then you can edit and download.")

    col1, col2 = st.columns(2)
    with col1:
        up_ds = st.file_uploader("Upload defaultdataset.json", type=["json"], key="up_ds")
        ds_model = st.selectbox("Standardizer model (dataset)", ALL_MODELS, index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0)
        if st.button("Load + Standardize dataset", disabled=(up_ds is None)):
            raw = up_ds.read().decode("utf-8", errors="ignore")
            try:
                obj = safe_json_loads(raw)
                obj = deterministic_standardize_defaultdataset(obj)
                if not is_standard_defaultdataset(obj):
                    obj = standardize_dataset_with_agent(raw, model=ds_model, max_tokens=12000)
                obj = deterministic_standardize_defaultdataset(obj)
                st.session_state.bundle["defaultdataset"] = obj
                st.success("Dataset loaded + standardized.")
                st.rerun()
            except Exception as e:
                st.error(f"Dataset load failed: {e}")

    with col2:
        up_gd = st.file_uploader("Upload defaultguide.md (md/txt)", type=["md", "txt"], key="up_gd")
        gd_model = st.selectbox("Standardizer model (guide)", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0)
        if st.button("Load + Standardize guide", disabled=(up_gd is None)):
            raw = up_gd.read().decode("utf-8", errors="ignore")
            try:
                md = normalize_md(raw)
                if not is_standard_defaultguide(md):
                    md = standardize_guide_with_agent(md, model=gd_model, max_tokens=12000)
                st.session_state.bundle["defaultguide"] = normalize_md(md)
                st.success("Guide loaded + standardized.")
                st.rerun()
            except Exception as e:
                st.error(f"Guide load failed: {e}")

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()

# 2) Generator
with tabs[1]:
    st.markdown("## Mock Bundle Generator")
    st.caption("Give instructions to generate new mock defaultdataset.json + defaultguide.md (standardized).")

    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]))
    max_tokens = st.number_input("max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000)
    prompt = st.text_area(
        "Instructions (prompt)",
        height=220,
        value=(
            "請用繁體中文產生一組 mock bundle：\n"
            "1) defaultdataset.json 需包含：tw_cases（至少 2 組資料集）與 k510_checklists（至少 1 組清單）\n"
            "2) defaultguide.md 需包含：tw_ 與 k510_ sections（至少各 1 段），並使用 BEGIN_SECTION 格式\n"
            "3) 每組 dataset 請給 2-3 筆案例，內容要明確標示為範例/合成\n"
        )
    )

    if st.button("Generate mock bundle"):
        cfg = agent_cfg("mock_bundle_generator")
        sys_p = cfg.get("system_prompt", "")
        user_p = prompt.strip()
        out = call_llm(model=model, system_prompt=sys_p, user_prompt=user_p, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))
        try:
            obj = safe_json_loads(out)
            ds = obj.get("defaultdataset_json", {})
            gd = obj.get("defaultguide_md", "")
            ds = deterministic_standardize_defaultdataset(ds, meta={"model": model, "prompt": prompt, "generated_at": now_iso()})
            gd = normalize_md(gd)
            # enforce guide standard if needed
            if not is_standard_defaultguide(gd):
                gd = standardize_guide_with_agent(gd, model="gemini-2.5-flash" if "gemini-2.5-flash" in ALL_MODELS else model, max_tokens=12000)
            st.session_state.bundle["defaultdataset"] = ds
            st.session_state.bundle["defaultguide"] = gd
            st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
            st.success("Generated + standardized bundle loaded into editors.")
            st.rerun()
        except Exception as e:
            st.error(f"Generator output parse failed: {e}")
            st.text_area("Raw output", value=out, height=260)

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()

# 3) Multi-pack Combiner
with tabs[2]:
    st.markdown("## Multi-pack Combiner")
    st.caption("Upload multiple defaultdataset.json and defaultguide.md files, standardize each, then combine into one bundle.")

    up_ds_multi = st.file_uploader("Upload multiple defaultdataset.json", type=["json"], accept_multiple_files=True, key="up_ds_multi")
    up_gd_multi = st.file_uploader("Upload multiple defaultguide.md", type=["md", "txt"], accept_multiple_files=True, key="up_gd_multi")

    model_ds = st.selectbox("Dataset standardizer model", ALL_MODELS, index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0, key="cmb_ds_model")
    model_gd = st.selectbox("Guide standardizer model", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0, key="cmb_gd_model")

    def combine_datasets(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = {"tw_cases": {}, "k510_checklists": {}, "meta": {"generated_at": now_iso(), "generated_by": {"tool":"combiner"}}}
        idx = 1
        for o in objs:
            o = deterministic_standardize_defaultdataset(o)
            for k, v in (o.get("tw_cases") or {}).items():
                kk = k
                while kk in merged["tw_cases"]:
                    kk = f"{k}__merge{idx}"
                    idx += 1
                merged["tw_cases"][kk] = v
            for k, v in (o.get("k510_checklists") or {}).items():
                kk = k
                while kk in merged["k510_checklists"]:
                    kk = f"{k}__merge{idx}"
                    idx += 1
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
        # datasets
        for f in (up_ds_multi or []):
            raw = f.read().decode("utf-8", errors="ignore")
            try:
                o = safe_json_loads(raw)
                o = deterministic_standardize_defaultdataset(o)
                if not is_standard_defaultdataset(o):
                    o = standardize_dataset_with_agent(raw, model=model_ds, max_tokens=12000)
                ds_objs.append(deterministic_standardize_defaultdataset(o))
            except Exception as e:
                st.error(f"Dataset file '{f.name}' failed: {e}")
        # guides
        for f in (up_gd_multi or []):
            raw = f.read().decode("utf-8", errors="ignore")
            gd_mds.append(normalize_md(raw))

        if ds_objs:
            st.session_state.bundle["defaultdataset"] = combine_datasets(ds_objs)
        if gd_mds:
            st.session_state.bundle["defaultguide"] = combine_guides(gd_mds)

        st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
        st.success("Combined bundle loaded into editors.")
        st.rerun()

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()

# 4) Guidance Ingestor
with tabs[3]:
    st.markdown("## Guidance Ingestor → Bundle Builder")
    st.caption("Paste or upload multiple guidance documents (txt/md/pdf). Preview them, then generate defaultguide.md + mock defaultdataset.json.")

    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("claude-3-5-sonnet-20241022") if "claude-3-5-sonnet-20241022" in ALL_MODELS else 0, key="ing_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, int(st.session_state.settings["max_tokens"]), 1000, key="ing_mt")

    pasted = st.text_area("Paste guidance (optional; you can paste multiple separated by '---')", height=180, key="ing_paste")
    uploads = st.file_uploader("Upload guidance files", type=["pdf", "md", "txt"], accept_multiple_files=True, key="ing_files")

    # Preview area
    st.markdown("### Preview")
    if uploads:
        for i, f in enumerate(uploads):
            with st.expander(f"Preview: {f.name}", expanded=False):
                suffix = f.name.lower().rsplit(".", 1)[-1]
                if suffix == "pdf":
                    b = f.read()
                    show_pdf_bytes(b, height=520)
                    # reset pointer not needed since we used bytes
                else:
                    text = f.read().decode("utf-8", errors="ignore")
                    st.markdown(normalize_md(text))

    prompt = st.text_area(
        "Ingestor prompt (editable; will be stored as meta)",
        height=180,
        value=(
            "請將以下多份 guidance 內容彙整為標準 defaultguide.md（BEGIN_SECTION 格式，section id 以 tw_/k510_ 開頭）。\n"
            "並基於 guidance 主題產生 mock defaultdataset.json：\n"
            "- tw_cases：至少 1 組資料集、每組 2 筆案例（合成示例）\n"
            "- k510_checklists：至少 1 組 checklist，至少 8 個 items\n"
            "請標註為範例/合成，勿冒充官方要求；若來源未包含明確要求，請標示 TBD。\n"
            "輸出格式必須是 JSON：{defaultdataset_json:..., defaultguide_md:'...'}"
        ),
        key="ing_prompt"
    )

    if st.button("Build bundle from guidances"):
        # build raw combined guidance text
        parts = []
        if pasted.strip():
            parts.append("=== PASTED ===\n" + pasted.strip())
        for f in (uploads or []):
            suffix = f.name.lower().rsplit(".", 1)[-1]
            if suffix == "pdf":
                pdf_bytes = f.getvalue()
                tmp = BytesIO(pdf_bytes)
                parts.append(f"=== FILE: {f.name} (PDF extracted) ===\n" + extract_pdf_text(tmp))
            else:
                parts.append(f"=== FILE: {f.name} ===\n" + normalize_md(f.getvalue().decode("utf-8", errors="ignore")))

        raw_all = "\n\n---\n\n".join(parts).strip()
        if not raw_all:
            st.warning("No guidance provided.")
        else:
            cfg = agent_cfg("guidance_ingestor_to_bundle")
            sys_p = cfg.get("system_prompt","")
            user_p = prompt.strip() + "\n\n---\n\n" + raw_all
            out = call_llm(model=model, system_prompt=sys_p, user_prompt=user_p, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))
            try:
                obj = safe_json_loads(out)
                ds = deterministic_standardize_defaultdataset(obj.get("defaultdataset_json", {}), meta={"model": model, "prompt": prompt, "generated_at": now_iso()})
                gd = normalize_md(obj.get("defaultguide_md", ""))
                if not is_standard_defaultguide(gd):
                    gd = standardize_guide_with_agent(gd, model="gemini-2.5-flash" if "gemini-2.5-flash" in ALL_MODELS else model, max_tokens=12000)
                st.session_state.bundle["defaultdataset"] = ds
                st.session_state.bundle["defaultguide"] = gd
                st.session_state.bundle["bundle_meta"]["last_updated"] = now_iso()
                st.success("Bundle built and loaded into editors.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to parse output: {e}")
                st.text_area("Raw output", value=out, height=260)

    st.markdown("---")
    bundle_editors()
    st.markdown("---")
    run_any_agent_on_results_ui()

# 5) Agents+Skills Studio
with tabs[4]:
    st.markdown("## Agents+Skills Studio")
    st.caption("Upload/standardize/edit/download agents.yaml and SKILL.md. If agents.yaml is not standardized, the system will standardize it.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### agents.yaml")
        up_agents = st.file_uploader("Upload agents.yaml (yaml/yml)", type=["yaml", "yml"], key="up_agents")
        std_model = st.selectbox("Agents standardizer model", ALL_MODELS, index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0, key="agents_std_model")

        if st.button("Load agents.yaml (standardize if needed)", disabled=(up_agents is None)):
            raw = up_agents.read().decode("utf-8", errors="ignore")
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
        st.markdown("### SKILL.md")
        up_skill = st.file_uploader("Upload SKILL.md", type=["md", "txt"], key="up_skill")
        if st.button("Load SKILL.md", disabled=(up_skill is None)):
            st.session_state.skill_md = up_skill.read().decode("utf-8", errors="ignore")
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

# 6) FDA Tool: Outline Builder
with tabs[5]:
    st.markdown("## FDA Guidance Tool: Outline Builder")
    st.caption("Create a comprehensive guidance outline with evidence expectations and stakeholder considerations.")

    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    agent_id = "fda_guidance_outline_builder" if "fda_guidance_outline_builder" in agents else (agents[0] if agents else "")
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
        out = run_agent(agent_id, user_prompt=prompt, model=model, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))
        st.text_area("Outline output (editable)", value=out, height=420, key="fda1_out")

# 7) FDA Tool: Harmonization Mapper
with tabs[6]:
    st.markdown("## FDA Guidance Tool: Harmonization & Standards Mapper")
    st.caption("Analyze a draft and create mapping tables + consistency checklist. No fabricated citations; mark TBD.")

    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    agent_id = "fda_harmonization_mapper" if "fda_harmonization_mapper" in agents else (agents[0] if agents else "")
    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0, key="fda2_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, 12000, 1000, key="fda2_mt")
    draft = st.text_area("Paste guidance draft (markdown/text)", height=260, key="fda2_draft")
    prompt = st.text_area(
        "Prompt",
        height=140,
        value="請為此 guidance 草稿建立：1) standards/citation mapping table 2) consistency checklist 3) gaps/TBD 列表。不得捏造引用。",
        key="fda2_prompt"
    )
    if st.button("Run harmonization mapping"):
        out = run_agent(agent_id, user_prompt=prompt + "\n\n---\n\n" + draft, model=model, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))
        st.text_area("Mapping output (editable)", value=out, height=420, key="fda2_out")

# 8) FDA Tool: Plain Language + FAQ
with tabs[7]:
    st.markdown("## FDA Guidance Tool: Plain Language + FAQ")
    st.caption("Rewrite into public-friendly language + FAQs + glossary. Avoid inventing requirements.")

    agents = sorted((st.session_state.agents_cfg.get("agents") or {}).keys())
    agent_id = "fda_plain_language_rewriter" if "fda_plain_language_rewriter" in agents else (agents[0] if agents else "")
    model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("grok-4-fast-reasoning") if "grok-4-fast-reasoning" in ALL_MODELS else 0, key="fda3_model")
    max_tokens = st.number_input("max_tokens", 1000, 120000, 12000, 1000, key="fda3_mt")
    draft = st.text_area("Paste technical guidance draft", height=260, key="fda3_draft")
    prompt = st.text_area(
        "Prompt",
        height=140,
        value="請改寫成 plain language 版本（保留原意），並產出 FAQ 10-15 題與 glossary。不得新增規範；不確定處標示 TBD。",
        key="fda3_prompt"
    )
    if st.button("Rewrite + FAQ"):
        out = run_agent(agent_id, user_prompt=prompt + "\n\n---\n\n" + draft, model=model, max_tokens=int(max_tokens), temperature=float(st.session_state.settings["temperature"]))
        st.text_area("Plain language output (editable)", value=out, height=420, key="fda3_out")

# 9) Dashboard
with tabs[8]:
    st.markdown("## Dashboard")
    st.caption("Session activity & quick checks")

    st.markdown("### Bundle status")
    ds_ok = is_standard_defaultdataset(st.session_state.bundle["defaultdataset"])
    gd_ok = is_standard_defaultguide(st.session_state.bundle["defaultguide"] or "")
    st.write({
        "defaultdataset_standard": ds_ok,
        "defaultguide_standard": gd_ok,
        "last_updated": st.session_state.bundle["bundle_meta"]["last_updated"]
    })

    st.markdown("### Quick diff helper")
    c1, c2 = st.columns(2)
    with c1:
        a = st.text_area("A text", height=160, key="diff_a")
    with c2:
        b = st.text_area("B text", height=160, key="diff_b")
    if st.button("Show diff"):
        st.code(diff_text(a, b), language="diff")

    st.markdown("### Run history")
    st.dataframe(pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame(columns=["ts","agent","model","target"]), use_container_width=True)
