# Token Counter & Prompt Playground (Streamlit)
# Features:
# - Token counting (tiktoken)
# - Multi-model cost estimates (GPT, Claude, Gemini)
# - File upload: .pdf, .txt, .docx (extract text and compute tokens/cost)
# - Context-window visualizer for several models
# - Compare Outputs UI: paste model responses or (optionally) call APIs for OpenAI/Anthropic/Google
#
# Usage: pip install streamlit tiktoken pypdf python-docx openai anthropic google-generative-ai
# Run: streamlit run token_counter_app.py

import streamlit as st
from io import BytesIO
import json
import math

# Tokenizer
import tiktoken

# PDF & DOCX helpers (optional libs)
try:
    from pypdf import PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        PdfReader = None

try:
    import docx
    have_docx = True
except Exception:
    have_docx = False

# Optional API SDK imports are attempted only when making calls

# -------------------------
# CONFIG: model prices & context windows (snapshot - may change)
# -------------------------
MODEL_PRICES = {
    "gpt-3.5": 0.0005,
    "gpt-4": 0.01,
    "claude-opus": 0.015,
    "claude-sonnet": 0.003,
    "gemini-pro": 0.002,
}

MODEL_CONTEXT = {
    "gpt-3.5": 4000,
    "gpt-4": 128000,
    "claude-opus": 200000,
    "claude-sonnet": 100000,
    "gemini-pro": 128000,
}

FALLBACK_ENCODING = "cl100k_base"

# -------------------------
# Helpers
# -------------------------

def get_encoding_for_model(model_name: str = "gpt-4"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        # fallback
        try:
            return tiktoken.get_encoding(FALLBACK_ENCODING)
        except Exception:
            # last resort: use encoding_for_model('gpt-4') that usually exists
            return tiktoken.encoding_for_model("gpt-4")

# Use a single encoding for token counting (works well as an approximation across models)
ENC = get_encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    # tiktoken expects str
    return len(ENC.encode(text))


def estimate_cost(tokens: int, model_key: str) -> float:
    price = MODEL_PRICES.get(model_key)
    if price is None:
        return None
    return (tokens / 1000.0) * price


# File parsers

def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    if PdfReader is None:
        return ""  # no pdf lib installed
    try:
        reader = PdfReader(file_bytes)
        text_parts = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text_parts.append(t)
        return "\n".join(text_parts)
    except Exception as e:
        return ""


def extract_text_from_docx(file_bytes: BytesIO) -> str:
    if not have_docx:
        return ""
    try:
        # python-docx accepts a path; create temporary in-memory
        from tempfile import NamedTemporaryFile
        tmp = NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(file_bytes.read())
        tmp.flush()
        tmp.close()
        doc = docx.Document(tmp.name)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception:
        return ""

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Token Counter & Prompt Playground", page_icon="🤖", layout="wide")
st.title("🤖 Token Counter & Prompt Playground — Multi-Model")
st.markdown(
    "This app helps you: count tokens, estimate costs across multiple LLMs, upload documents (PDF/TXT/DOCX), visualize context-window usage, and compare model outputs.\n\n" 
    "**Security note:** If you enter API keys in the sidebar they are stored only in your browser session (Streamlit session). Do not paste production keys here on shared machines."
)

# Sidebar: API keys and settings
st.sidebar.header("Settings & API keys (optional)")
openai_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")
anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password")
google_key = st.sidebar.text_input("Google Generative API Key", type="password")

st.sidebar.markdown("---")
max_output_tokens = st.sidebar.number_input("Estimated max response tokens (for cost est.)", min_value=0, value=256, step=64)
st.sidebar.markdown("Prices snapshot used in this app (USD per 1K tokens):")
for k, v in MODEL_PRICES.items():
    st.sidebar.write(f"- {k}: ${v}/1K tokens")

# Tabs for features
tab1, tab2, tab3, tab4 = st.tabs(["Token & Cost", "Upload & Analyze Doc", "Context Visualizer", "Compare Outputs"])

# -------------------------
# Tab 1: Token & Cost
# -------------------------
with tab1:
    st.header("Token Counter & Multi-Model Cost Estimator")
    text = st.text_area("Paste text or a prompt here:", height=220)
    if text:
        tokens = count_tokens(text)
        st.metric("Token Count", tokens)

        st.subheader("Cost estimates (input tokens only)")
        cols = st.columns(3)
        model_map = {
            "GPT-3.5 Turbo": "gpt-3.5",
            "GPT-4 Turbo": "gpt-4",
            "Claude 3 Opus": "claude-opus",
            "Claude 3.5 Sonnet": "claude-sonnet",
            "Gemini 1.5 Pro": "gemini-pro",
        }
        for i, (name, key) in enumerate(model_map.items()):
            cost = estimate_cost(tokens + max_output_tokens, key)  # naive total cost estimate
            cols[i % 3].write(f"**{name}**")
            cols[i % 3].write(f"Input tokens: {tokens}")
            cols[i % 3].write(f"Est. total tokens incl. output: {tokens + max_output_tokens}")
            cols[i % 3].write(f"Est. cost (USD): ${cost:.6f}")

        st.markdown("---")
        st.subheader("Prompting examples")
        st.markdown("**Zero-shot**: `Summarize the story of Cinderella in 3 sentences.`")
        st.markdown("**Few-shot**: Provide 1–2 example Q→A pairs before your new Q."
                    )
        st.markdown("**Chain-of-Thought**: `Explain step by step how you solved this and then give the final answer.`")

# -------------------------
# Tab 2: Upload & Analyze Document
# -------------------------
with tab2:
    st.header("Upload a PDF / TXT / DOCX and get tokens + cost")
    uploaded = st.file_uploader("Upload file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"]) 
    if uploaded is not None:
        file_bytes = BytesIO(uploaded.read())
        uploaded.seek(0)
        name = uploaded.name
        st.write(f"**File uploaded:** {name} — size: {file_bytes.getbuffer().nbytes} bytes")

        # extract text
        text_out = ""
        if name.lower().endswith('.pdf'):
            text_out = extract_text_from_pdf(file_bytes)
        elif name.lower().endswith('.docx'):
            file_bytes.seek(0)
            text_out = extract_text_from_docx(file_bytes)
        else:
            file_bytes.seek(0)
            try:
                text_out = file_bytes.read().decode('utf-8')
            except Exception:
                text_out = ""

        if not text_out:
            st.warning("Could not extract text from file or file is empty. Ensure required libraries are installed (.pdf/.docx support).")
        else:
            tokens = count_tokens(text_out)
            st.metric("Extracted token count", tokens)
            st.download_button("Download extracted text as .txt", data=text_out, file_name=f"{name}.extracted.txt")

            st.subheader("Cost estimates for this document (input only)")
            for model_name, key in model_map.items():
                cost = estimate_cost(tokens, key)
                st.write(f"- {model_name}: {tokens} tokens → ${cost:.6f}")

            # Optionally show a small preview
            if len(text_out) > 5000:
                st.info("Showing first 5000 characters as preview — full text is available to download.")
                st.text_area("Preview (truncated)", value=text_out[:5000], height=300)
            else:
                st.text_area("Extracted text", value=text_out, height=300)

# -------------------------
# Tab 3: Context Visualizer
# -------------------------
with tab3:
    st.header("Context Window Visualizer")
    sample = st.text_area("Paste text to visualize context usage:", height=220)
    st.write("Tip: Use this to check how much of each model's memory your text will occupy.")
    if sample:
        tokens = count_tokens(sample)
        st.metric("Token count", tokens)

        rows = []
        for model_name, key in model_map.items():
            ctx = MODEL_CONTEXT.get(key, None)
            pct = None
            if ctx:
                pct = min(1.0, tokens / ctx)
            rows.append((model_name, key, ctx, pct))

        for name, key, ctx, pct in rows:
            if ctx:
                st.write(f"**{name}** — context window: {ctx} tokens")
                st.progress(pct)
                st.write(f"Usage: {tokens} / {ctx} tokens — {pct*100:.4f}%")
                # recommended chunk size
                recommended = max(256, int(ctx * 0.75))
                st.write(f"Recommended chunk size for processing: ~{recommended} tokens (approx 75% of context)")
            else:
                st.write(f"**{name}** — context unknown")

# -------------------------
# Tab 4: Compare Outputs
# -------------------------
with tab4:
    st.header("Compare Outputs: paste responses or call APIs (optional)")
    st.markdown("You can either: (A) paste responses you received from different tools, or (B) supply API keys in the sidebar and let the app call models for you. API calls are optional and may incur cost on your account.")

    prompt = st.text_area("Prompt to send to models (or to use for token counting):", height=160)
    col_a, col_b = st.columns([1,1])
    with col_a:
        st.subheader("Options")
        selected_models = st.multiselect("Models to compare (if using API)", list(model_map.keys()), default=["GPT-3.5 Turbo", "Claude 3 Opus"])
        use_api = st.checkbox("Attempt API call for selected models (requires keys in sidebar)")
        st.write("If you don't have API keys, uncheck and paste outputs manually below.")
        est_in_tokens = count_tokens(prompt) if prompt else 0
        st.write(f"Prompt tokens: {est_in_tokens}")
        st.write(f"Estimated input cost (GPT-4 example): ${estimate_cost(est_in_tokens, 'gpt-4'):.6f}")
        if use_api:
            st.write("Max response tokens for API calls:")
            st.write(max_output_tokens)

    with col_b:
        st.subheader("Manual/pasted responses")
        paste_resp_openai = st.text_area("Paste OpenAI/ChatGPT response (optional)", key="paste_openai", height=120)
        paste_resp_claude = st.text_area("Paste Claude response (optional)", key="paste_claude", height=120)
        paste_resp_gemini = st.text_area("Paste Gemini response (optional)", key="paste_gemini", height=120)

    # Buttons
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        do_call = st.button("Run compare (API + paste)")

    # Output area
    if do_call:
        results = {}

        # 1) If use_api & openai key present and user selected GPT model(s)
        if use_api and openai_key and any(m.startswith('GPT') for m in selected_models):
            try:
                import openai
                openai.api_key = openai_key
                if any('GPT-3.5' in m for m in selected_models):
                    resp = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{"role":"user","content":prompt}], max_tokens=max_output_tokens)
                    results['GPT-3.5 Turbo'] = resp['choices'][0]['message']['content']
                if any('GPT-4' in m for m in selected_models):
                    resp = openai.ChatCompletion.create(model='gpt-4', messages=[{"role":"user","content":prompt}], max_tokens=max_output_tokens)
                    results['GPT-4 Turbo'] = resp['choices'][0]['message']['content']
            except Exception as e:
                st.warning(f"OpenAI call failed: {e}")

        # 2) Anthropic (Claude) example
        if use_api and anthropic_key and any('Claude' in m for m in selected_models):
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=anthropic_key)
                # simple prompt wrapper
                anthropic_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                resp = client.completions.create(model='claude-3-opus', prompt=anthropic_prompt, max_tokens_to_sample=max_output_tokens)
                results['Claude 3 Opus'] = resp['completion']
            except Exception as e:
                st.warning(f"Anthropic call failed: {e}")

        # 3) Google Gemini example (illustrative)
        if use_api and google_key and any('Gemini' in m for m in selected_models):
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                resp = genai.generate_text(model='gemini-1.5-pro', prompt=prompt, max_output_tokens=max_output_tokens)
                # The resp structure depends on SDK; this is illustrative
                results['Gemini 1.5 Pro'] = resp.text
            except Exception as e:
                st.warning(f"Google/Gemini call failed: {e}")

        # 4) Add pasted responses (manual) if present
        if paste_resp_openai:
            results['OpenAI (pasted)'] = paste_resp_openai
        if paste_resp_claude:
            results['Claude (pasted)'] = paste_resp_claude
        if paste_resp_gemini:
            results['Gemini (pasted)'] = paste_resp_gemini

        if not results:
            st.info("No results to show — either provide pasted responses or enable API calls with valid keys and selected models.")
        else:
            st.subheader("Comparison results")
            # Show side-by-side
            cols = st.columns(len(results))
            for i, (k, v) in enumerate(results.items()):
                with cols[i]:
                    st.markdown(f"**{k}**")
                    st.write(f"Tokens (approx): {count_tokens(v)}")
                    st.text_area(label=f"{k} output", value=v, height=300)

            # Quick similarity check (basic)
            st.markdown("---")
            st.subheader("Quick similarity check (basic)")
            keys = list(results.keys())
            pairs = []
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    a = results[keys[i]]
                    b = results[keys[j]]
                    # naive similarity: Jaccard on words (very rough)
                    aset = set(a.split())
                    bset = set(b.split())
                    inter = len(aset & bset)
                    union = len(aset | bset) or 1
                    jacc = inter/union
                    pairs.append((keys[i], keys[j], jacc))
            for p in pairs:
                st.write(f"Similarity ({p[0]} vs {p[1]}): Jaccard={p[2]:.3f}")

            # Allow user to download JSON
            out_json = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button("Download comparison JSON", data=out_json, file_name="comparison_results.json")


# -------------------------
# Footer: quick tips
# -------------------------
st.markdown("---")
st.write("**Quick tips:** 1) Use few-shot examples to guide style. 2) Use Chain-of-Thought for step-by-step reasoning. 3) Chunk long docs into ~75% of model context. 4) Keep API keys private.")


# End of app
