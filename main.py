import os
import streamlit as st
from streamlit_chat import message
import openai


from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Does it work?
from langchain.callbacks.streamlit import StreamlitCallbackHandler


st.set_page_config(page_title="探究アシスタント", layout="wide")
st.title("探究アシスタント（ChatGPT3）")
st.caption("Created by Daiki Ito")

st.write('右のリンクから「Sign in」をしてAPIキーを取得してください',
         'https://beta.openai.com/account/api-keys')

st.write("こんにちは！何でも聞いてください（あくまで参考に）")
input_apikey = st.text_input("取得したAPIキーを貼り付けてください")

if input_apikey == "":
    st.error("apiキーを貼り付けてください。")
else:
    openai.api_key = input_apikey

    system_message = """
    あなたは研究アシスタントです。ユーザは高校生で、あなたに研究に関する質問を投げかけます。
    アシスタントとして、論文執筆や研究の遂行に役立つ回答を、できる限り根拠を示した上で返してください。"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


    @st.cache_resource
    def load_conversation():
        llm = ChatOpenAI(
            streaming=True,
            callback_manager=CallbackManager([
                StreamlitCallbackHandler(),
                StreamingStdOutCallbackHandler()
            ]),
            verbose=True,
            temperature=0,
            max_tokens=1024
        )
        memory = ConversationBufferMemory(return_messages=True)
        conversation_message = ConversationChain(
            memory=memory,
            prompt=prompt,
            llm=llm
        )
        return conversation_message


    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "past" not in st.session_state:
        st.session_state.past = []

    with st.form("探究アシスタントに質問する（あくまでも参考に）"):
        user_message = st.text_area("質問を入力してください")

        submitted = st.form_submit_button("質問する")
        if submitted:
            conversation_message = load_conversation()
            answer = conversation_message.predict(input=user_message)

            st.session_state.past.append(user_message)
            st.session_state.generated.append(answer)

            if st.session_state["generated"]:
                for i in range(len(st.session_state.generated) - 1, -1, -1):
                    message(st.session_state.generated[i], key=str(i))
                    message(st.session_state.past[i], is_user=True,
                            key=str(i) + "_user")
