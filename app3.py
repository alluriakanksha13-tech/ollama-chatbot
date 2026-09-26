import streamlit as st
import ollama

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Role-Based AI Chatbot", layout="wide")
st.title("🤖 Role-Based AI Chatbot (Ollama + Phi-3)")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    role = st.selectbox(
        "Choose Role",
        ["Teacher", "Student", "Data Analyst", "Developer", "HR Assistant"]
    )

    model = st.selectbox("Model", ["phi3", "llama3", "mistral"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 500, 200)

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# ---------------- ROLE SYSTEM PROMPT ----------------
role_prompts = {
    "Teacher": "You are a helpful teacher. Explain concepts simply with examples.",
    "Student": "You are a student assistant. Help with learning and doubts clearly.",
    "Data Analyst": "You are a data analyst. Give structured, logical, technical answers.",
    "Developer": "You are a senior software engineer. Provide code, best practices, and debugging help.",
    "HR Assistant": "You are an HR assistant. Answer professionally about hiring, resumes, and careers."
}

system_prompt = role_prompts[role]

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Ask something...")

# ---------------- CHAT LOGIC ----------------
if prompt:
    # add system prompt dynamically if role changes
    if st.session_state.messages[0]["content"] != system_prompt:
        st.session_state.messages = [
            {"role": "system", "content": system_prompt}
        ]

    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""

        stream = ollama.chat(
            model=model,
            messages=st.session_state.messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        for chunk in stream:
            response += chunk["message"]["content"]
            placeholder.markdown(response)

    # save response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

# ---------------- FOOTER ----------------
st.caption(" Role-based AI chatbot powered by Streamlit + Ollama (Phi-3)")