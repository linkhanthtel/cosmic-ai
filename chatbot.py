"""
Minimal LangChain chatbot using a free Hugging Face model.

This is a practice starting point. The interesting part is `get_response`,
which runs a simple LangChain chain:  prompt | llm | output parser.

Extend it however you like — add memory, RAG, tools, streaming, etc.

Requires a free Hugging Face token in HF_TOKEN
(get one at https://huggingface.co/settings/tokens, type: Read).
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class ChatBot:
    def __init__(self):
        self.history = []
        self.llm = self._build_llm()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are Cosmic AI, a helpful assistant. Be clear and concise."),
                ("human", "{question}"),
            ]
        )
        # A LangChain "chain": prompt -> model -> plain-text output.
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_llm(self):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Get a free token at "
                "https://huggingface.co/settings/tokens and set HF_TOKEN "
                "(locally in .env, in production: fly secrets set HF_TOKEN=hf_...)."
            )

        endpoint = HuggingFaceEndpoint(
            repo_id=os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            task="text-generation",
            huggingfacehub_api_token=token,
            temperature=float(os.environ.get("HF_TEMPERATURE", "0.3")),
            max_new_tokens=int(os.environ.get("HF_MAX_NEW_TOKENS", "512")),
            provider=os.environ.get("HF_PROVIDER", "auto"),
        )
        return ChatHuggingFace(llm=endpoint)

    def get_response(self, message):
        answer = self.chain.invoke({"question": message}).strip()
        self.history.append({"user": message})
        self.history.append({"bot": answer})
        return answer

    def clear_conversation(self):
        self.history = []
        return "Conversation cleared."
