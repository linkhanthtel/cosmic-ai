"""
Minimal LangChain chatbot using a free Hugging Face model.

Text chat:  prompt | llm | output parser
Images:     HF vision model describes the image, then the text LLM answers.

Requires HF_TOKEN — https://huggingface.co/settings/tokens (type: Read)
"""

import base64
import mimetypes
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from huggingface_hub import InferenceClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class ChatBot:
    def __init__(self):
        self.history = []
        self.llm = self._build_llm()
        self._vision_client = None

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Cosmic AI, a helpful assistant. Be clear and concise. "
                    "When image analysis is provided in the user's message, use it to answer accurately.",
                ),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    @staticmethod
    def _hf_token() -> str:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Get a free token at "
                "https://huggingface.co/settings/tokens and set HF_TOKEN "
                "(locally in .env, in production: fly secrets set HF_TOKEN=hf_...)."
            )
        return token

    def _vision_client_instance(self) -> InferenceClient:
        if self._vision_client is None:
            # Vision models route through third-party providers; "auto" picks one that works.
            self._vision_client = InferenceClient(
                token=self._hf_token(),
                provider=os.environ.get("HF_VISION_PROVIDER", "auto"),
            )
        return self._vision_client

    @staticmethod
    def _image_mime_type(image_path: Path) -> str:
        mime, _ = mimetypes.guess_type(image_path.name)
        return mime or "image/jpeg"

    def _build_llm(self):
        endpoint = HuggingFaceEndpoint(
            repo_id=os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            task="text-generation",
            huggingfacehub_api_token=self._hf_token(),
            temperature=float(os.environ.get("HF_TEMPERATURE", "0.3")),
            max_new_tokens=int(os.environ.get("HF_MAX_NEW_TOKENS", "512")),
            provider=os.environ.get("HF_PROVIDER", "auto"),
        )
        return ChatHuggingFace(llm=endpoint)

    def describe_image(self, image_path: Path, question: str = "") -> str:
        """Use a Hugging Face vision chat model to describe an uploaded image."""
        client = self._vision_client_instance()
        model = os.environ.get("HF_VISION_MODEL", "google/gemma-3-12b-it")
        image_bytes = image_path.read_bytes()
        mime_type = self._image_mime_type(image_path)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        prompt = question.strip() or "Describe this image in detail."

        try:
            result = client.chat_completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=int(os.environ.get("HF_VISION_MAX_TOKENS", "256")),
            )
        except StopIteration as exc:
            raise RuntimeError(
                f"Vision model failed ({model}): no inference provider supports this model. "
                "Try HF_VISION_MODEL=google/gemma-3-12b-it with HF_VISION_PROVIDER=auto."
            ) from exc
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            raise RuntimeError(f"Vision model failed ({model}): {detail}") from exc

        content = result.choices[0].message.content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    def get_response(self, message: str) -> str:
        answer = self.chain.invoke({"question": message}).strip()
        self.history.append({"user": message})
        self.history.append({"bot": answer})
        return answer

    def get_response_with_attachments(
        self,
        message: str,
        image_paths: list[Path],
        video_names: list[str] | None = None,
    ) -> str:
        video_names = video_names or []
        context_parts: list[str] = []

        for index, path in enumerate(image_paths, start=1):
            label = path.name
            image_question = message or "Describe this image in detail."
            description = self.describe_image(path, question=image_question)
            context_parts.append(f"Image {index} ({label}): {description}")

        if video_names:
            names = ", ".join(video_names)
            context_parts.append(
                f"Videos attached ({names}): video analysis is not supported yet — "
                "only images can be analyzed for now."
            )

        if context_parts:
            analysis = "\n".join(context_parts)
            if message:
                full_question = (
                    f"The user shared media. Here is the analysis:\n{analysis}\n\n"
                    f"User message: {message}"
                )
            else:
                full_question = (
                    f"The user shared media. Here is the analysis:\n{analysis}\n\n"
                    "Describe what you see and respond helpfully based on the analysis above."
                )
        else:
            full_question = message

        return self.get_response(full_question)

    def clear_conversation(self):
        self.history = []
        return "Conversation cleared."
