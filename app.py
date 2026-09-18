"""BioShield AI — Natural Farming Pest Advisory Assistant.

Streamlit Demonstration UI for document-grounded agricultural decision support.
"""

import streamlit as st
from src import config
from src.rag_pipeline import SAFE_FALLBACK, answer_query
from src.retriever import ChromaRetriever

st.set_page_config(
    page_title="BioShield AI — Natural Farming Pest Advisory Assistant",
    page_icon="🌿",
    layout="wide",
)

# Custom CSS for polished aesthetics
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e4620;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: #2e7d32;
        margin-bottom: 1rem;
    }
    .sdg-badge {
        display: inline-block;
        background-color: #e8f5e9;
        border: 1px solid #81c784;
        border-radius: 6px;
        padding: 0.35rem 0.65rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #1b5e20;
    }
    .advisory-box {
        background-color: #f9fbf9;
        border-left: 5px solid #2e7d32;
        border-radius: 4px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    .evidence-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .status-badge-ok {
        color: #2e7d32;
        font-weight: bold;
    }
    .status-badge-warn {
        color: #e65100;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_kb_stats():
    """Retrieve indexed document count from ChromaDB."""
    try:
        retriever = ChromaRetriever()
        coll = retriever._get_collection()
        if coll is not None:
            return coll.count()
    except Exception:
        pass
    return 0


# Sidebar content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/natural-food.png", width=64)
    st.title("BioShield AI")
    st.markdown("**Program:** 1M1B AI for Sustainability Virtual Internship (IBM SkillsBuild & AICTE)")

    st.markdown("---")
    st.subheader("Global SDG Alignment")
    st.markdown(
        """
        <div class="sdg-badge">🌱 SDG 15 — Life on Land</div><br>
        <small>Promoting natural biological management to preserve soil vitality, beneficial insects, and terrestrial ecosystems.</small>
        <br><br>
        <div class="sdg-badge">♻️ SDG 12 — Responsible Consumption</div><br>
        <small>Fostering safe, low-chemical input practices in sustainable agricultural production.</small>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown(
        """
        - **Embedding**: `sentence-transformers`
        - **Vector DB**: `ChromaDB` (Local)
        - **Inference**: `IBM Granite`
        - **Pipeline**: Simple Grounded RAG
        """
    )

    st.markdown("---")
    st.subheader("System Health")
    kb_count = get_kb_stats()
    if kb_count > 0:
        st.markdown(f"📚 Knowledge Base: <span class='status-badge-ok'>{kb_count} Chunks Indexed</span>", unsafe_allow_html=True)
    else:
        st.markdown("📚 Knowledge Base: <span class='status-badge-warn'>Empty (Run ingestion)</span>", unsafe_allow_html=True)
        st.caption("Run: `python -m src.ingest` in terminal to populate.")

    if config.GRANITE_BASE_URL:
        st.markdown(f"🤖 LLM Model: <span class='status-badge-ok'>{config.GRANITE_MODEL}</span>", unsafe_allow_html=True)
    else:
        st.markdown("🤖 LLM Model: <span class='status-badge-warn'>Endpoint Not Set</span>", unsafe_allow_html=True)
        st.caption("Configure `GRANITE_BASE_URL` in `.env` or use local Ollama.")

    st.markdown("---")
    st.markdown(
        "**Responsible AI Notice:**\n"
        "BioShield AI is strictly grounded in verified agricultural extension manuals. "
        "It will refuse to invent unverified dosages, chemical formulas, or guaranteed cures."
    )


# Main layout
st.markdown("<div class='main-title'>BioShield AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Natural Farming Pest Advisory Assistant — Grounded Decision Support</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "Ask about sustainable, biological, and natural pest-management practices. "
    "Advisories are retrieved directly from verified agricultural documentation and are **not** a substitute for professional agricultural diagnosis."
)

st.markdown("---")

# Quick example selector
example_selected = st.selectbox(
    "💡 Or choose a sample inquiry:",
    [
        "Custom Input",
        "Pigeon Pea: Pod borer sustainable biological management (Supported in KB)",
        "Rice: Brown planthopper control (Absent from current KB - Tests safe fallback)",
        "Cotton: Leaves turning yellow (Ambiguous symptom - Tests safety refusal)",
    ],
)

default_crop = ""
default_problem = ""
default_query = ""
default_pref = "Natural / Biological pest management"

if example_selected == "Pigeon Pea: Pod borer sustainable biological management (Supported in KB)":
    default_crop = "Pigeon pea"
    default_problem = "I am seeing pod borer damage and larvae entering the pods."
    default_query = "What sustainable biological practices and cultural controls can I consider?"
    default_pref = "Natural / Biological pest management"
elif example_selected == "Rice: Brown planthopper control (Absent from current KB - Tests safe fallback)":
    default_crop = "Rice"
    default_problem = "Brown planthopper hopper burn symptoms on leaves."
    default_query = "What bio-pesticide dosage can I apply?"
    default_pref = "Natural / Biological pest management"
elif example_selected == "Cotton: Leaves turning yellow (Ambiguous symptom - Tests safety refusal)":
    default_crop = "Cotton"
    default_problem = "My crop leaves are turning yellow."
    default_query = "What is the exact disease and give me the chemical recipe?"
    default_pref = "Any practice"

# User Input Form
with st.form("advisory_form"):
    col1, col2 = st.columns(2)
    with col1:
        crop_input = st.text_input("🌾 Target Crop:", value=default_crop, placeholder="e.g. Pigeon pea, Chickpea, Cotton")
        pref_input = st.selectbox(
            "🌱 Farming Preference:",
            [
                "Natural / Biological pest management",
                "Cultural & Preventive practices",
                "Microbial biopesticides (Bt / NPV)",
                "Botanical extracts (Neem / NSKE)",
                "General sustainable advisory",
            ],
            index=0 if default_pref.startswith("Natural") else 4,
        )

    with col2:
        problem_input = st.text_input(
            "🐛 Observed Pest / Symptoms:",
            value=default_problem,
            placeholder="e.g. Pod borer damage, caterpillars on flower buds",
        )
        threshold_val = st.slider(
            "🎯 Retrieval Sensitivity Threshold (Cosine Distance):",
            min_value=0.10,
            max_value=0.80,
            value=config.SIMILARITY_THRESHOLD,
            step=0.05,
            help="Lower distance values require closer semantic match to consider evidence sufficient.",
        )

    query_input = st.text_area(
        "❓ Farmer's Question:",
        value=default_query,
        placeholder="e.g. What sustainable practices can I consider to protect my crop?",
        height=90,
    )

    submitted = st.form_submit_button("🔍 Generate Grounded Advisory", type="primary", use_container_width=True)

if submitted:
    if not query_input.strip() and not problem_input.strip():
        st.warning("⚠️ Please provide an observed pest/problem or enter a question.")
    else:
        with st.spinner("Analyzing agricultural knowledge base & grounding advisory..."):
            try:
                result = answer_query(
                    query=query_input,
                    threshold=threshold_val,
                    crop=crop_input,
                    problem=problem_input,
                    preference=pref_input,
                )

                st.markdown("### 📋 Advisory Results")

                if result.evidence_status == "insufficient":
                    st.warning("⚠️ **Evidence Status: Insufficient Information in Current Knowledge Base**")
                    st.info(result.answer)
                    st.markdown(
                        "> **Responsible AI Note:** The system refused to fabricate remedies because no verified "
                        "guidance meeting the threshold was located in the current knowledge base. "
                        "Please consult your local Krishi Vigyan Kendra (KVK) or State Agricultural Extension Office."
                    )
                else:
                    st.success("✅ **Evidence Status: Verified Guidance Located**")
                    st.markdown(result.answer)

                    # Display Retrieved Evidence in an Expander
                    if result.retrieved_evidence:
                        with st.expander(f"📑 Inspect Retrieved Evidence ({len(result.retrieved_evidence)} chunks used)"):
                            for i, ev in enumerate(result.retrieved_evidence, start=1):
                                st.markdown(f"**Evidence Chunk #{i}** (Distance: `{ev.distance:.4f}`)")
                                st.markdown(f"- **Title**: {ev.metadata.get('title', 'Unknown')}")
                                st.markdown(f"- **Source**: {ev.metadata.get('source', 'Agricultural Extension')}")
                                st.markdown(f"- **Page**: {ev.metadata.get('page', 'N/A')}")
                                st.text(ev.text)
                                st.markdown("---")

                    # Sources Display
                    if result.sources:
                        st.subheader("📚 Verified References")
                        for src in result.sources:
                            st.markdown(
                                f"- **{src.get('title', 'Agricultural Guide')}** — "
                                f"{src.get('source', 'Research Institution')} (Page: {src.get('page', 'N/A')})"
                            )

                    # Limitations & Disclaimer
                    st.caption(
                        "**Disclaimer:** BioShield AI is an informational decision-support prototype created for the "
                        "1M1B AI for Sustainability Virtual Internship. It does not replace field visits or certified agricultural advice."
                    )

            except Exception as exc:
                st.error(f"❌ An error occurred while executing advisory: {exc}")
                if "GRANITE_BASE_URL" in str(exc) or "Failed to reach" in str(exc):
                    st.info(
                        "💡 **Granite Endpoint Configuration Reminder:**\n\n"
                        "To connect to IBM Granite, set your credentials in the `.env` file:\n"
                        "```bash\n"
                        "GRANITE_MODEL=ibm/granite-3-8b-instruct\n"
                        "GRANITE_BASE_URL=http://localhost:11434/v1  # or your watsonx endpoint\n"
                        "GRANITE_API_KEY=your_key_if_applicable\n"
                        "```"
                    )
