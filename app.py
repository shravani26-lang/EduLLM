import time

import streamlit as st


st.set_page_config(
    page_title="EduLLM | Your CS Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --edu-purple: #8b5cf6;
        --edu-blue: #38bdf8;
        --edu-muted: #a5b4fc;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #27204a 0, transparent 38%),
            radial-gradient(ellipse at 95% 10%, #132d4a 0, transparent 32%),
            #0b1020;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17162d 0%, #111827 100%);
        border-right: 1px solid rgba(167, 139, 250, .18);
    }
    .edu-hero {
        padding: 1.65rem 1.8rem;
        margin: .4rem 0 1.2rem;
        border: 1px solid rgba(167, 139, 250, .25);
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(91, 55, 170, .34), rgba(22, 85, 125, .22));
        box-shadow: 0 16px 55px rgba(0, 0, 0, .18);
    }
    .edu-eyebrow {
        color: #c4b5fd;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .edu-title {
        margin: .35rem 0 .45rem;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        letter-spacing: -.045em;
        background: linear-gradient(90deg, #fff, #c4b5fd 55%, #7dd3fc);
        color: transparent;
        background-clip: text;
        -webkit-background-clip: text;
    }
    .edu-subtitle { color: #cbd5e1; font-size: 1.02rem; }
    .edu-section-title { color: #e2e8f0; font-weight: 700; }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, .12);
        border-radius: 16px;
        background: rgba(15, 23, 42, .62);
    }
    [data-testid="stChatInput"] {
        border-color: rgba(167, 139, 250, .42);
    }
    [data-testid="stChatInput"] textarea { color: #f8fafc; }
    div.stButton > button {
        border: 1px solid rgba(167, 139, 250, .28);
        border-radius: 12px;
        transition: border-color .15s ease, transform .15s ease;
    }
    div.stButton > button:hover {
        border-color: #8b5cf6;
        transform: translateY(-1px);
    }
    .edu-note { color: #94a3b8; font-size: .85rem; }
    @media (max-width: 700px) {
        .edu-hero { padding: 1.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_edullm():
    from model import load_model

    return load_model()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("## 🎓 EduLLM")
    st.caption("Your computer science learning companion")
    st.divider()

    st.markdown("### Model")
    st.markdown("**Qwen2.5-0.5B-Instruct**")
    st.caption("Fine-tuned with LoRA · PyTorch · PEFT")
    st.caption("Runs on an available CUDA GPU or CPU.")
    st.caption("The model loads on your first question; this page is ready now.")

    st.divider()
    st.markdown("### Suggested questions")
    suggestions = [
        "What is a stack?",
        "Explain binary search.",
        "What is supervised learning?",
        "Explain normalization in DBMS.",
        "What is Dijkstra's algorithm?",
    ]
    for question in suggestions:
        if st.button(question, key=f"suggest_{question}", use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

    st.divider()
    if st.button("🧹  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()


st.markdown(
    """
    <div class="edu-hero">
      <div class="edu-eyebrow">Learn · Explore · Understand</div>
      <div class="edu-title">Meet your AI study partner.</div>
      <div class="edu-subtitle">
        Ask computer science questions and get clear, example-driven explanations.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown("#### 👋 What would you like to learn today?")
    st.caption("Choose a suggested question from the sidebar, or ask your own below.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("duration_seconds") is not None:
            st.caption(f"Response time: {message['duration_seconds']:.2f} seconds")

prompt = st.chat_input("Ask EduLLM a computer science question...")
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")

if prompt and prompt.strip():
    previous_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            start_time = time.perf_counter()
            with st.spinner("Loading EduLLM and preparing your explanation..."):
                chatbot = load_edullm()
                answer = chatbot.generate(
                    prompt,
                    history=[
                        message
                        for message in previous_history
                        if message["role"] in {"user", "assistant"}
                        and not message.get("error")
                    ],
                )
            elapsed = time.perf_counter() - start_time
            st.markdown(answer)
            st.caption(
                f"Response time: {elapsed:.2f} seconds "
                "(the first response may include model loading)"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "duration_seconds": elapsed,
                }
            )
        except Exception as error:
            error_message = f"EduLLM couldn't generate a response: {error}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "error": True}
            )

st.markdown(
    '<div class="edu-note">EduLLM · Fine-tuned Qwen2.5 with your local LoRA adapter</div>',
    unsafe_allow_html=True,
)
