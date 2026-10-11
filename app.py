"""Apna Data Apni Baten — upload your data, then talk to it."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import ai_agent
import data_tools

APP_NAME = "Apna Data Apni Baten"
VERSION = "1.0.0"
_HERE = Path(__file__).parent
SAMPLE = next(
    (p for p in (_HERE / "sample_data" / "pk_online_sales_2025.csv",
                 _HERE / "pk_online_sales_2025.csv") if p.exists()),
    _HERE / "sample_data" / "pk_online_sales_2025.csv",
)

st.set_page_config(page_title=APP_NAME, page_icon="💬", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container{ padding-top:2.2rem; max-width:1200px }
  .hero{
    background:linear-gradient(120deg,#1d2b64 0%,#5b3a8e 55%,#8e3a6b 100%);
    border-radius:14px; padding:26px 30px; margin-bottom:22px; color:#fff;
  }
  .hero h1{ margin:0 0 6px; font-size:30px; font-weight:700 }
  .hero p{ margin:0; opacity:.9; font-size:15.5px }
  .pill{
    display:inline-block; background:rgba(255,255,255,.17); border-radius:99px;
    padding:4px 13px; font-size:12.5px; margin:10px 7px 0 0;
  }
  div[data-testid="stMetric"]{
    background:rgba(130,150,255,.08); border:1px solid rgba(130,150,255,.2);
    border-radius:10px; padding:14px 16px;
  }
  .alert-warn{ background:rgba(255,170,60,.12); border-left:3px solid #ffaa3c;
               padding:9px 13px; border-radius:5px; margin-bottom:7px; font-size:14px }
  .alert-info{ background:rgba(110,160,255,.1); border-left:3px solid #6ea0ff;
               padding:9px 13px; border-radius:5px; margin-bottom:7px; font-size:14px }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- state
def init():
    d = {
        "df": None, "profile": None, "profile_text": "", "source": "",
        "messages": [], "ai_name": "Baten",
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)


init()


def load_into_state(file_or_path, label):
    try:
        df = data_tools.load_data(file_or_path)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return
    profile = data_tools.profile_data(df)
    st.session_state.update(
        df=df,
        profile=profile,
        profile_text=data_tools.profile_to_text(df, profile),
        source=label,
        messages=[],
    )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### 💬 {APP_NAME}")
    st.caption(f"v{VERSION}")

    st.markdown("#### ① Connect an AI")
    provider = st.radio("Provider", list(ai_agent.PROVIDERS), horizontal=True,
                        label_visibility="collapsed")
    cfg = ai_agent.PROVIDERS[provider]

    secret_key = ""
    try:
        secret_key = st.secrets.get(cfg["secret"], "")
    except Exception:
        pass
    secret_key = secret_key or os.environ.get(cfg["secret"], "")

    if secret_key:
        st.success(f"{provider} key loaded", icon="🔑")
        api_key = secret_key
    else:
        api_key = st.text_input(
            f"{provider} API key", type="password", placeholder=cfg["prefix"],
            help=f"Get one at {cfg['keys_url']} — it stays in this browser session only.",
        )
        st.caption(f"[Get a {provider} key]({cfg['keys_url']})")

    choices = [m[0] for m in cfg["models"]] + ["Other \u2014 type a model name"]
    notes = dict(cfg["models"])

    picked = st.selectbox(
        "Model", choices, index=0,
        format_func=lambda m: f"{m}  \u00b7  {notes[m]}" if m in notes else m,
        help="Cheapest is at the top. Prices are per 1M tokens and change often.",
    )
    if picked == choices[-1]:
        model = st.text_input("Model name", placeholder="exact id from the provider's docs").strip()
    else:
        model = picked
    st.caption(f"[Current pricing]({cfg['prices_url']})")

    st.markdown("#### ② Name your AI")
    st.session_state.ai_name = st.text_input(
        "What should I call it?", value=st.session_state.ai_name, max_chars=24,
        help="Just a name. It'll introduce itself with this.",
    ) or "Baten"

    st.markdown("#### ③ Add data")
    upload = st.file_uploader("Upload a file", type=data_tools.SUPPORTED,
                              label_visibility="collapsed")
    st.caption("CSV, TSV, Excel, JSON or Parquet")

    if upload is not None and st.session_state.source != upload.name:
        load_into_state(upload, upload.name)

    if st.button("✨ Try the sample data", use_container_width=True):
        if SAMPLE.exists():
            with open(SAMPLE, "rb") as fh:
                class _F:
                    name = SAMPLE.name
                    read = fh.read
                load_into_state(_F(), SAMPLE.name)
        else:
            st.error("sample_data/pk_online_sales_2025.csv is missing from the folder.")

    if st.session_state.df is not None:
        st.divider()
        st.caption(f"Loaded: **{st.session_state.source}**")
        c1, c2 = st.columns(2)
        if c1.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        transcript = "\n\n".join(
            f"**{m['role']}:** {m['content']}" for m in st.session_state.messages
        )
        c2.download_button("Export", transcript or "No messages yet.",
                           file_name="chat.md", use_container_width=True,
                           disabled=not st.session_state.messages)

    st.divider()
    st.caption(f"Your file stays in memory for this session only. The data profile and your "
               f"questions are sent to {provider}.")


# ---------------------------------------------------------------- header
name = st.session_state.ai_name
st.markdown(f"""
<div class="hero">
  <h1>💬 {APP_NAME}</h1>
  <p>Apna data upload karein, phir {name} se poochein — English, Urdu ya Roman Urdu mein.
     Woh khud code likh kar asli numbers aur charts dega.</p>
  <span class="pill">OpenAI</span><span class="pill">pandas</span><span class="pill">Plotly</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- empty state
if st.session_state.df is None:
    a, b, c = st.columns(3)
    a.markdown("#### 📁 Bring your data\nCSV, Excel, JSON or Parquet — or press "
               "**Try the sample data** in the sidebar.")
    b.markdown("#### 🔎 Instant profile\nTypes, missing values, duplicates, outliers "
               "and a quality score out of 100.")
    c.markdown(f"#### 💬 Ask {name}\nReal numbers computed with code, plus the code itself "
               "so you can check it.")
    st.info("← Add your OpenAI API key and some data in the sidebar to start.")
    st.stop()


df = st.session_state.df
profile = st.session_state.profile

tab_chat, tab_profile, tab_data = st.tabs(["💬 Chat", "📊 Profile", "🗂 Data"])


# ---------------------------------------------------------------- profile tab
with tab_profile:
    k = st.columns(6)
    k[0].metric("Rows", f"{profile['rows']:,}")
    k[1].metric("Columns", profile["cols"])
    k[2].metric("Missing", f"{profile['missing_pct']}%")
    k[3].metric("Duplicates", profile["duplicate_rows"])
    k[4].metric("Memory", f"{profile['memory_mb']} MB")
    k[5].metric("Quality", f"{profile['quality_score']}/100")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("##### Column types")
        counts = profile["kind_counts"]
        fig = px.pie(names=list(counts), values=list(counts.values()), hole=0.55,
                     color_discrete_sequence=px.colors.sequential.Purp_r)
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(f"##### Quality notes · {len(profile['alerts'])}")
        if not profile["alerts"]:
            st.success("Nothing worrying found.")
        for a in profile["alerts"]:
            st.markdown(f"<div class='alert-{a['level']}'>{a['text']}</div>",
                        unsafe_allow_html=True)

    st.markdown("##### Columns")
    st.dataframe(pd.DataFrame(profile["columns"]), use_container_width=True, hide_index=True)

    num = df.select_dtypes("number")
    if not num.empty:
        st.markdown("##### Numeric summary")
        st.dataframe(num.describe().round(2), use_container_width=True)

        if num.shape[1] >= 2:
            st.markdown("##### Correlations")
            fig = px.imshow(num.corr(numeric_only=True).round(2), text_auto=True,
                            aspect="auto", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1)
            fig.update_layout(height=420, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇ Download profile report",
                       data_tools.profile_to_markdown(profile),
                       file_name="data_profile.md")


# ---------------------------------------------------------------- data tab
with tab_data:
    st.dataframe(df, use_container_width=True, height=560)
    st.download_button("⬇ Download as CSV", df.to_csv(index=False), file_name="data.csv")


# ---------------------------------------------------------------- chat tab
with tab_chat:
    if not st.session_state.messages:
        st.caption("Try one of these:")
        suggestions = [
            "Give me an executive summary of this data",
            "Which columns have problems?",
            "Show me the biggest totals as a bar chart",
            "Sab se zyada kya bik raha hai?",
            "How does this app profile my data?",
        ]
        cols = st.columns(len(suggestions))
        for col, text in zip(cols, suggestions):
            if col.button(text, use_container_width=True):
                st.session_state.pending = text
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "💬"):
            st.markdown(m["content"])
            for i, fig in enumerate(m.get("figures", [])):
                st.plotly_chart(fig, use_container_width=True, key=f"{id(m)}-{i}")
            if m.get("steps"):
                with st.expander(f"How {name} worked it out · {len(m['steps'])} steps"):
                    for n, s in enumerate(m["steps"], 1):
                        st.markdown(f"**Step {n}**")
                        st.code(s["code"], language="python")
                        st.text(s["output"][:2000])

    question = st.chat_input(f"Ask {name} about your data…")
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")

    if question:
        if not api_key:
            st.error(f"Add your {provider} API key in the sidebar first.")
            st.stop()
        if not model:
            st.error("Pick a model, or type a model name, in the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="💬"):
            slot = st.empty()
            history = [{"role": m["role"], "content": m["content"]}
                       for m in st.session_state.messages[:-1]]
            answer = None

            try:
                stream = ai_agent.ask(question, history, df, st.session_state.profile_text,
                                      api_key, model, name, provider)
                with st.status(f"{name} is working…", expanded=False) as status:
                    for kind, payload in stream:
                        if kind == "step":
                            st.code(payload["code"], language="python")
                            st.text(payload["output"][:800])
                        else:
                            answer = payload
                            status.update(label=f"Done · {len(payload['steps'])} steps",
                                          state="complete")
            except ai_agent.FriendlyError as exc:
                slot.error(str(exc))
                st.session_state.messages.pop()
                st.stop()

            if answer:
                st.markdown(answer["content"])
                for i, fig in enumerate(answer["figures"]):
                    st.plotly_chart(fig, use_container_width=True, key=f"live-{i}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer["content"],
                    "figures": answer["figures"],
                    "steps": answer["steps"],
                })
