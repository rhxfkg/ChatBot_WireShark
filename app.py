import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

class ChatLLM:
    def __init__(self):
        # Model 설정
        self._model = ChatOllama(model="gemma2:2b", temperature=3)

        # Prompt 설정
        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요.

            Question: {question}
        """
        self._prompt = ChatPromptTemplate.from_template(self._template)

        # Chain 연결
        self._chain = (
            {'question': RunnablePassthrough()}
            | self._prompt
            | self._model
            | StrOutputParser()
        )

    def invoke(self, user_input):
        response = self._chain.invoke({'question': user_input})
        return response

class ChatWeb:
    def __init__(self, llm, page_title="Gazzi Chatbot", page_icon=":books:"):
        self._llm = llm
        self._page_title = page_title
        self._page_icon = page_icon

    def print_messages(self):
        if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
            for chat_message in st.session_state["messages"]:
                st.chat_message(chat_message.role).write(chat_message.content)

    def run(self):
        # 웹 페이지 기본 설정
        st.set_page_config(
            page_title=self._page_title,
            page_icon=self._page_icon)
        st.title(self._page_title)

        # 대화 기록 초기화
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # 이전 대화 기록 출력
        self.print_messages()

        # 사용자 입력과 AI 응답 처리
        if user_input := st.chat_input("질문을 입력해 주세요."):
            # 사용자의 입력 출력
            st.chat_message("user").write(f"{user_input}")
            st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

            # AI 응답 생성
            response = self._llm.invoke(user_input)
            with st.chat_message("assistant"):
                msg_assistant = response
                st.write(msg_assistant)
                st.session_state["messages"].append(ChatMessage(role="assistant", content=msg_assistant))

if __name__ == '__main__':
    llm = ChatLLM()
    web = ChatWeb(llm=llm)
    web.run()
