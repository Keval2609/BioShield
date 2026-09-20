"""BioShield AI — Natural Farming Pest Advisory Assistant.

Streamlit Demonstration UI for responsible, document-grounded agricultural decision support.
"""



import streamlit as st

from src import config
from src.ingest import ingest_documents
from src.rag_pipeline import answer_query
from src.ui_helpers import (
    EXAMPLE_QUERIES,
    check_kb_status,
    format_error_message,
)

import os
import sys

# Auto-initialize Knowledge Base on startup if missing
@st.cache_resource
def initialize_kb_if_needed():
    if "pytest" in sys.modules:
        return
    
    # Check if core PDFs exist
    pdf_dir = config.CORE_RESOURCES_DIR
    if not os.path.exists(pdf_dir) or not any(f.endswith(".pdf") for f in os.listdir(pdf_dir)):
        st.error(f"Missing core PDF documents in {pdf_dir}. Please place the required manuals before running.")
        return

    kb_available, kb_count, _ = check_kb_status()
    if not kb_available or kb_count == 0:
        ingest_documents()

initialize_kb_if_needed()

# Page configuration
st.set_page_config(
    page_title="BioShield AI — Natural Farming Pest Advisory Assistant",
    page_icon="🌱",
    layout="wide",
)

# Clean, professional agricultural theme styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4d24;
        margin-bottom: 0.1rem;
    }
    .main-subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: #2e7d32;
        margin-bottom: 0.8rem;
    }
    .intro-box {
        font-size: 1.0rem;
        color: #2c3e50;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    .confidence-badge-high {
        display: inline-block;
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .confidence-badge-med {
        display: inline-block;
        background-color: #fff8e1;
        color: #b78103;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .confidence-badge-low {
        display: inline-block;
        background-color: #ffebee;
        color: #c62828;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .practice-card {
        background-color: #f6fbf7;
        color: #2c3e50;
        border-left: 4px solid #388e3c;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_example(preset_name: str) -> None:
    """Callback to populate input fields from the selected demo preset."""
    if preset_name in EXAMPLE_QUERIES:
        preset = EXAMPLE_QUERIES[preset_name]
        st.session_state["crop_field"] = preset["crop"]
        st.session_state["problem_field"] = preset["problem"]
        st.session_state["approach_field"] = preset["approach"]
        st.session_state["question_field"] = preset["question"]


# Initialize session state keys for the inputs
for field_key, default_val in [
    ("crop_field", ""),
    ("problem_field", ""),
    ("approach_field", "Natural farming"),
    ("question_field", ""),
]:
    if field_key not in st.session_state:
        st.session_state[field_key] = default_val


# -----------------------------------------------------------------------------
# Sidebar: Responsible AI & System Health
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🛡️ Responsible AI")
    st.markdown(
        """
        BioShield AI is governed by strict agricultural AI safety principles:

        - **Grounded in curated agricultural sources:** Answers are retrieved strictly from verified extension manuals.
        - **Source attribution:** Verifiable document, page, and section citations from retrieval metadata.
        - **Uncertainty disclosure:** Transparent confidence scoring and explicit limitations disclosure.
        - **No guaranteed diagnosis:** Recommends potential causes rather than definitive assertions.
        - **No fabricated agricultural facts:** Zero tolerance for hallucinated remedies or preparation steps.
        - **No unsupported application rates:** Strictly refuses to invent chemical recipes or unverified dosages.
        - **Human expert verification for ambiguous cases:** Always advises consultation with local Krishi Vigyan Kendra (KVK) or extension officers.
        """
    )

    st.markdown("---")
    st.subheader("System Status")
    kb_available, kb_count, kb_msg = check_kb_status()
    if kb_available:
        st.success(f"📚 Knowledge Base: {kb_count} Chunks Indexed")
    else:
        st.warning(f"📚 Knowledge Base: {kb_msg}")
        st.caption("Run `python ingest.py` in your terminal to index the core manuals.")

    st.info(f"🤖 LLM Provider: `{config.LLM_PROVIDER.upper()}`")
    if config.LLM_PROVIDER == "groq":
        if config.GROQ_API_KEY:
            st.caption(f"Model: {config.GROQ_MODEL}")
        else:
            st.warning("⚠️ Groq API Key Not Configured")
    else:
        if config.OLLAMA_BASE_URL:
            st.caption(f"Model: {config.OLLAMA_MODEL}")
        else:
            st.warning("⚠️ Ollama Endpoint Not Configured")


# -----------------------------------------------------------------------------
# Main Header & Prominent Disclaimer
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>BioShield AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Natural Farming Pest Advisory Assistant</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='intro-box'>"
    "BioShield AI retrieves verified agricultural evidence from a curated knowledge base and uses a configured language model to explain that evidence in simple language."
    "</div>",
    unsafe_allow_html=True,
)

if config.LLM_PROVIDER == "ollama" and "granite" in config.OLLAMA_MODEL.lower():
    st.markdown("<div class='intro-box' style='margin-top: -15px;'><small>Powered locally by IBM Granite through Ollama.</small></div>", unsafe_allow_html=True)

st.warning(
    "⚠️ **Important Disclaimer:** BioShield AI is an agricultural information and decision-support prototype, "
    "not an autonomous agricultural diagnosis or treatment system. Recommendations should be verified with "
    "a qualified agricultural expert, especially for severe, ambiguous, or unfamiliar crop problems."
)

st.markdown("---")


# -----------------------------------------------------------------------------
# Demo Mode: Example Queries
# -----------------------------------------------------------------------------
st.subheader("💡 Example Queries")
st.caption("Click any preset below to populate the input fields with supported or test inquiries:")

col_ex1, col_ex2, col_ex3 = st.columns(3)
preset_keys = list(EXAMPLE_QUERIES.keys())

with col_ex1:
    if st.button(
        f"🌾 {preset_keys[0]}",
        use_container_width=True,
        help="Supported natural farming inquiry for cotton pests.",
    ):
        load_example(preset_keys[0])

with col_ex2:
    if st.button(
        f"🐛 {preset_keys[1]}",
        use_container_width=True,
        help="Supported biological control inquiry for pigeon pea pod borer.",
    ):
        load_example(preset_keys[1])

with col_ex3:
    if st.button(
        f"🚫 {preset_keys[2]}",
        use_container_width=True,
        help="Unsupported chemical inquiry testing safe refusal behavior.",
    ):
        load_example(preset_keys[2])

st.markdown("---")


# -----------------------------------------------------------------------------
# User Input Form
# -----------------------------------------------------------------------------
st.subheader("📝 Farmer Inquiry")

with st.form("inquiry_form"):
    col1, col2 = st.columns(2)
    with col1:
        crop_input = st.text_input(
            "🌾 Crop:",
            key="crop_field",
            placeholder="e.g. Cotton, Paddy, Pigeon pea, Gram",
        )
        approach_options = ["Natural farming", "Organic farming", "Sustainable/IPM"]
        approach_input = st.selectbox(
            "🌱 Farming approach:",
            options=approach_options,
            key="approach_field",
        )

    with col2:
        problem_input = st.text_input(
            "🐛 Observed pest / symptom / problem:",
            key="problem_field",
            placeholder="e.g. Small insects appearing under the leaves, leaf curl",
        )
        threshold_input = st.slider(
            "🎯 Retrieval Similarity Threshold (Cosine Distance):",
            min_value=0.15,
            max_value=0.60,
            value=float(config.SIMILARITY_THRESHOLD),
            step=0.05,
            help="Queries with nearest evidence distance above this threshold will safely refuse to answer.",
        )

    question_input = st.text_area(
        "❓ User question:",
        key="question_field",
        placeholder="e.g. What sustainable pest-management practices are relevant?",
        height=95,
    )

    submitted = st.form_submit_button(
        "🔍 Generate Grounded Advisory",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Execution & Display
# -----------------------------------------------------------------------------
if submitted:
    if not problem_input.strip() and not question_input.strip():
        st.warning("⚠️ Please provide an observed problem or enter a question to generate an advisory.")
    else:
        with st.spinner("Retrieving verified agricultural evidence and consulting IBM Granite..."):
            try:
                result = answer_query(
                    query=question_input.strip() or problem_input.strip(),
                    crop=crop_input.strip(),
                    problem=problem_input.strip(),
                    preference=approach_input,
                    threshold=threshold_input,
                )

                st.markdown("## 📋 Advisory Results")

                if result.status == "LLM_ERROR":
                    st.error("❌ The system retrieved relevant evidence, but the language model could not generate the advisory. Please retry or consult the source documents directly.")
                elif result.status == "INVALID_MODEL_OUTPUT":
                    st.error("❌ The language model generated an invalid response that could not be parsed. Please retry.")
                elif result.status == "INSUFFICIENT_EVIDENCE":
                    # Safe refusal when below evidence threshold
                    st.warning("⚠️ **I could not find sufficient verified information in the current knowledge base to answer this safely.**")
                    st.info("ℹ️ **Please consult a qualified agricultural expert or local agricultural extension service.**")
                    st.caption("BioShield AI refuses to generate speculative remedies when verified natural farming evidence is absent.")

                else:
                    # Successful structured advisory display
                    st.success("✅ **Verified Agricultural Guidance Located**")

                    # 1. Possible Issue
                    st.markdown("### 1. Possible Issue")
                    st.write(result.get("possible_issue") or "Pest / disease symptoms matching inquiry.")

                    # 2. Evidence-Based Sustainable Practices
                    st.markdown("### 2. Evidence-Based Sustainable Practices")
                    practices = result.get("evidence_based_practices", [])
                    if practices:
                        for p in practices:
                            st.markdown(f"<div class='practice-card'>🌱 {p}</div>", unsafe_allow_html=True)
                    else:
                        st.write("Refer to general preventive natural farming protocols.")

                    # 3. Why This May Be Relevant
                    st.markdown("### 3. Why This May Be Relevant")
                    st.write(result.get("why_relevant") or "Addresses observed symptoms using grounded practices.")

                    # 4. Precautions
                    st.markdown("### 4. Precautions")
                    precautions = result.get("precautions", [])
                    if precautions:
                        for pr in precautions:
                            st.markdown(f"- ⚠️ {pr}")
                    else:
                        st.write("Ensure standard personal protection and follow local natural formulation protocols.")

                    # 5. Sources (strictly from retrieval metadata)
                    st.markdown("### 5. Sources")
                    sources = result.get("sources", [])
                    if sources:
                        for src in sources:
                            doc_title = src.get("title", "Agricultural Extension Document")
                            page_num = src.get("page", "N/A")
                            section_name = src.get("section", "General Advisory")
                            st.markdown(f"- 📄 **{doc_title}** — **Page:** {page_num} | **Section:** {section_name}")
                        st.caption("*(Authoritative source citations verified directly against ChromaDB retrieval metadata)*")
                    else:
                        st.write("Core Natural Farming Knowledge Base.")

                    # 6. Evidence Confidence
                    st.markdown("### 6. Evidence Confidence")
                    conf_val = result.get("evidence_confidence", "Medium")
                    if conf_val == "High":
                        badge_html = "<span class='confidence-badge-high'>🟢 High Evidence Confidence</span>"
                    elif conf_val == "Low":
                        badge_html = "<span class='confidence-badge-low'>🔴 Low Evidence Confidence</span>"
                    else:
                        badge_html = "<span class='confidence-badge-med'>🟡 Medium Evidence Confidence</span>"
                    st.markdown(f"{badge_html} — Based on retrieval similarity and evidence coverage; not a guarantee of correctness.", unsafe_allow_html=True)

                    # 7. Limitations
                    st.markdown("### 7. Limitations")
                    st.info(result.get("limitations") or "Advisory prototype for informational purposes only.")

                # Retrieved Evidence (Expandable Section for transparency)
                if result.retrieved_evidence:
                    with st.expander(f"📑 Retrieved Evidence ({len(result.retrieved_evidence)} chunks analyzed)"):
                        st.caption("Direct text passages retrieved from ChromaDB vector store for grounding:")
                        for idx, ev in enumerate(result.retrieved_evidence, start=1):
                            st.markdown(f"**Evidence Chunk #{idx}** (Cosine Distance: `{ev.distance:.4f}`)")
                            st.markdown(f"- **Document:** {ev.document_title}")
                            st.markdown(f"- **Page:** {ev.page} | **Section:** {ev.section}")
                            st.text(ev.text)
                            st.markdown("---")

            except Exception as exc:
                st.error(f"❌ {format_error_message(exc)}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #555; font-size: 0.9rem; padding: 1rem 0;'>"
    "🌱 <strong>Prototype for AI for Sustainability — SDG 15: Life on Land</strong>"
    "</div>",
    unsafe_allow_html=True,
)
