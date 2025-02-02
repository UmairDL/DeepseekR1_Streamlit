import streamlit as st
import ollama
from ollama import Client

# Initialize Ollama client
client = Client(host='http://localhost:11434')

# Set page config
st.set_page_config(
    page_title="DeepSeek Chat",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if Ollama is running
try:
    client.list()
except Exception as e:
    st.error(f"Ollama connection failed: {str(e)}")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.title("Settings")
    model_name = st.selectbox("Select Model", ["deepseek-r1:1.5b"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.01)
    max_length = st.slider("Max Length", 256, 4096, 2048)

# Main chat interface
st.title("🧠 DeepSeek Local Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Generate response
            response = client.chat(
                model=model_name,
                messages=st.session_state.messages,
                stream=True,
                options={
                    'temperature': temperature,
                    'num_predict': max_length
                }
            )
            
            # Stream the response
            for chunk in response:
                if chunk['message']['content']:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            full_response = "Sorry, I couldn't process that request."
            response_placeholder.markdown(full_response)
        
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
