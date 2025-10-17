"""
Simple Streamlit Chatbot with Groq API (Streaming, Fixed)
"""
import os, requests, json
import streamlit as st

# defaults
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
CHAT_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="Groq Chatbot", layout="wide")

# helper for streaming
def groq_chat_stream(messages, max_tokens=500):
    url = f"{GROQ_API_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=body, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue

# sidebar
with st.sidebar:
    st.header("Settings")
    temp_api = st.text_input("Paste GROQ API key", type="password")
    temp_url = st.text_input("API URL", value=GROQ_API_URL)
    model_choice = st.text_input("Chat Model", value=CHAT_MODEL)
    if temp_api:
        st.session_state["GROQ_API_KEY_TEMP"] = temp_api
    if temp_url:
        st.session_state["GROQ_API_URL_TEMP"] = temp_url
    if model_choice:
        st.session_state["CHAT_MODEL_TEMP"] = model_choice

if "GROQ_API_KEY_TEMP" in st.session_state:
    GROQ_API_KEY = st.session_state["GROQ_API_KEY_TEMP"]
if "GROQ_API_URL_TEMP" in st.session_state:
    GROQ_API_URL = st.session_state["GROQ_API_URL_TEMP"]
if "CHAT_MODEL_TEMP" in st.session_state:
    CHAT_MODEL = st.session_state["CHAT_MODEL_TEMP"]

# init chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "system", "content": "You are a helpful assistant"}]

st.title("🤖 Simple Groq Chatbot")

# show past messages (skip system prompt)
for msg in st.session_state["chat_history"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# input box pinned at bottom
if query := st.chat_input("Type your message here..."):
    # append user message immediately
    st.session_state["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # streaming bot reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for delta in groq_chat_stream(st.session_state["chat_history"]):
            full_response += delta
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    # save assistant reply to history
    st.session_state["chat_history"].append({"role": "assistant", "content": full_response})

st.caption("Minimal chatbot using Groq API with streaming responses")
