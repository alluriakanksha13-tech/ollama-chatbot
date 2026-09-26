import streamlit as st
st.write("Hello,Streamlit!")


import ollama
user_input = st.text_input("You:")
if st.button("Send"):
    if user_input:
        response = ollama.chat(
            model = 'llama3.2:3b',
            messages= [
                {"role" : "user",
                 "content" : user_input}

            ]

    )
    st.write(response["message"]['content'])