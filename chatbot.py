"""
Cosmic AI chatbot powered by LangChain (RAG over training_data.json).

Embeddings (EMBEDDINGS_BACKEND):
  - local (default): local sentence-transformers model (needs PyTorch, ~500MB RAM).
  - hf_api: compute embeddings via the Hugging Face Inference API (low memory,
    recommended for small hosts like Fly.io 512MB). Needs HF_TOKEN.

Answer generation (auto-detected):
  - HF_TOKEN set        -> free Hugging Face LLM (ChatHuggingFace).
  - OPENAI_API_KEY set  -> OpenAI.
  - neither             -> retrieval-only (best matching stored answer).
"""

import json
import os
import re
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
except ImportError:
    ChatHuggingFace = None
    HuggingFaceEndpoint = None

# Remote (API) embeddings: no local PyTorch/sentence-transformers download,
# which keeps memory low enough for small hosts (e.g. Fly.io 512 MB).
try:
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
except ImportError:
    HuggingFaceEndpointEmbeddings = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


class ChatBot:
  FAISS_DIR = "models/faiss_index"
  EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

  def __init__(self):
    self.training_data_file = "data/training_data.json"
    self.training_data = []

    self.conversation_history = []
    self.user_context = {}
    self.conversation_topics = []
    self.follow_up_questions = []

    self.personality_traits = {
      "enthusiasm": 0.7,
      "empathy": 0.8,
      "humor": 0.3,
      "formality": 0.6,
      "curiosity": 0.8,
      "eagerness": 0.7,
      "openness": 0.8,
      "professionalism": 0.9,
    }
    self.conversation_mood = "neutral"
    self.user_name = None
    self.conversation_start_time = None

    self._embeddings = None
    self.vectorstore = None
    self.retriever = None
    self._llm = None
    self._llm_provider = None
    self._rag_prompt = None

    self.load_training_data()
    self._ensure_vectorstore()

  # ------------------------------------------------------------------ data
  def load_training_data(self):
    if os.path.exists(self.training_data_file):
      try:
        with open(self.training_data_file, "r", encoding="utf-8") as f:
          self.training_data = json.load(f)
      except Exception as e:
        print(f"Error loading training data: {e}")
        self.training_data = []
    else:
      self.training_data = []

  def save_training_data(self):
    os.makedirs(os.path.dirname(self.training_data_file), exist_ok=True)
    with open(self.training_data_file, "w", encoding="utf-8") as f:
      json.dump(self.training_data, f, ensure_ascii=False, indent=2)

  def add_training_data(self, question, answer, include_timestamp=False):
    try:
      entry = {"question": question, "answer": answer}
      if include_timestamp:
        entry["timestamp"] = datetime.now().isoformat()
      self.training_data.append(entry)
      self.save_training_data()
      self._rebuild_vectorstore()
      return True
    except Exception as e:
      print(f"Error adding training data: {e}")
      return False

  def delete_training_data(self, index):
    try:
      if 0 <= index < len(self.training_data):
        del self.training_data[index]
        self.save_training_data()
        self._rebuild_vectorstore()
        return True
      return False
    except Exception as e:
      print(f"Error deleting training data: {e}")
      return False

  def get_training_data(self):
    return self.training_data

  def train_model(self, training_data=None):
    if training_data is not None:
      self.training_data = training_data
      self.save_training_data()
    if not self.training_data:
      return {"trained_samples": 0, "message": "No training data available"}
    try:
      self._rebuild_vectorstore()
      return {
        "trained_samples": len(self.training_data),
        "message": "Vector index rebuilt successfully (LangChain + FAISS)",
      }
    except Exception as e:
      print(f"Error training: {e}")
      return {"trained_samples": 0, "message": f"Training failed: {str(e)}"}

  def retrain_model(self):
    return self.train_model()

  # ----------------------------------------------------------- vector store
  @staticmethod
  def _hf_token():
    return (
      os.environ.get("HUGGINGFACEHUB_API_TOKEN")
      or os.environ.get("HUGGINGFACE_API_KEY")
      or os.environ.get("HF_TOKEN")
    )

  def _get_embeddings(self):
    """Build the embedding backend.

    EMBEDDINGS_BACKEND=hf_api  -> compute embeddings on Hugging Face servers
      (no local PyTorch download; low memory; good for small hosts like Fly.io).
    EMBEDDINGS_BACKEND=local (default) -> local sentence-transformers model.

    Both use the SAME model, so a FAISS index built one way is compatible
    with queries embedded the other way.
    """
    if self._embeddings is not None:
      return self._embeddings

    backend = os.environ.get("EMBEDDINGS_BACKEND", "local").strip().lower()
    token = self._hf_token()

    if backend == "hf_api":
      if HuggingFaceEndpointEmbeddings is None:
        raise RuntimeError(
          "EMBEDDINGS_BACKEND=hf_api but langchain-huggingface is not installed. "
          "Install it or set EMBEDDINGS_BACKEND=local."
        )
      if not token:
        # Do NOT silently fall back to local: on a small host that loads PyTorch
        # and gets OOM-killed with no catchable error. Fail loudly instead.
        raise RuntimeError(
          "EMBEDDINGS_BACKEND=hf_api requires a Hugging Face token. "
          "Set the HF_TOKEN secret (fly secrets set HF_TOKEN=hf_...) "
          "or use EMBEDDINGS_BACKEND=local."
        )
      self._embeddings = HuggingFaceEndpointEmbeddings(
        model=self.EMBEDDING_MODEL,
        huggingfacehub_api_token=token,
      )
      return self._embeddings

    # Local backend (default). Imported lazily so PyTorch is only loaded when
    # actually running local embeddings.
    from langchain_huggingface import HuggingFaceEmbeddings

    self._embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
    return self._embeddings

  def _documents_from_training_data(self):
    docs = []
    for item in self.training_data:
      q = item.get("question", "").strip()
      a = item.get("answer", "").strip()
      if not q or not a:
        continue
      docs.append(
        Document(
          page_content=f"Question: {q}\nAnswer: {a}",
          metadata={"question": q, "answer": a},
        )
      )
    return docs

  def _rebuild_vectorstore(self):
    docs = self._documents_from_training_data()
    if not docs:
      self.vectorstore = None
      self.retriever = None
      self._llm = None
      self._rag_prompt = None
      return

    embeddings = self._get_embeddings()
    self.vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(self.FAISS_DIR, exist_ok=True)
    self.vectorstore.save_local(self.FAISS_DIR)
    self._setup_retriever()
    self._setup_llm_chain()

  def _ensure_vectorstore(self):
    if self.vectorstore is not None:
      return
    if not self.training_data:
      return
    index_path = os.path.join(self.FAISS_DIR, "index.faiss")
    if os.path.exists(index_path):
      try:
        embeddings = self._get_embeddings()
        self.vectorstore = FAISS.load_local(
          self.FAISS_DIR,
          embeddings,
          allow_dangerous_deserialization=True,
        )
        self._setup_retriever()
        self._setup_llm_chain()
        return
      except Exception as e:
        print(f"Could not load FAISS index, rebuilding: {e}")
    self._rebuild_vectorstore()

  def _setup_retriever(self):
    if self.vectorstore is None:
      self.retriever = None
      return
    self.retriever = self.vectorstore.as_retriever(
      search_type="similarity",
      search_kwargs={"k": 4},
    )

  def _build_llm(self):
    """Pick an LLM for RAG generation.

    Priority:
      1. Free Hugging Face Inference API (needs a free HF token)
      2. OpenAI (optional, paid)
      3. None -> caller falls back to retrieval-only answers
    """
    # 1) Hugging Face Inference API (free tier).
    hf_token = (
      os.environ.get("HUGGINGFACEHUB_API_TOKEN")
      or os.environ.get("HUGGINGFACE_API_KEY")
      or os.environ.get("HF_TOKEN")
    )
    if hf_token and ChatHuggingFace is not None and HuggingFaceEndpoint is not None:
      try:
        endpoint = HuggingFaceEndpoint(
          repo_id=os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
          task="text-generation",
          huggingfacehub_api_token=hf_token,
          temperature=float(os.environ.get("HF_TEMPERATURE", "0.3")),
          max_new_tokens=int(os.environ.get("HF_MAX_NEW_TOKENS", "512")),
          provider=os.environ.get("HF_PROVIDER", "auto"),
        )
        llm = ChatHuggingFace(llm=endpoint)
        self._llm_provider = "huggingface"
        return llm
      except Exception as e:
        print(f"Could not initialize Hugging Face LLM, falling back: {e}")

    # 2) OpenAI (optional).
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and ChatOpenAI is not None:
      try:
        llm = ChatOpenAI(
          model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
          temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.3")),
          api_key=api_key,
        )
        self._llm_provider = "openai"
        return llm
      except Exception as e:
        print(f"Could not initialize OpenAI LLM, falling back: {e}")

    # 3) No LLM configured -> retrieval-only mode.
    self._llm_provider = None
    return None

  def _setup_llm_chain(self):
    self._llm = None
    self._llm_provider = None
    self._rag_prompt = None
    if self.retriever is None:
      return

    self._llm = self._build_llm()
    if self._llm is None:
      return

    self._rag_prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "You are Cosmic AI, a helpful assistant. Answer using ONLY the provided context "
          "from the knowledge base. If the context does not contain enough information, "
          "say you do not have that information in your knowledge base. Be concise and accurate.",
        ),
        (
          "human",
          "Context from knowledge base:\n{context}\n\n"
          "Recent conversation:\n{history}\n\n"
          "User question: {question}",
        ),
      ]
    )

  def _llm_rag_response(self, user_input):
    docs = self.retriever.invoke(user_input)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    chain = self._rag_prompt | self._llm | StrOutputParser()
    return chain.invoke(
      {
        "context": context,
        "history": self._format_history(),
        "question": user_input,
      }
    )

  # ---------------------------------------------------------------- responses
  def get_response(self, user_input):
    try:
      if not self.training_data:
        return (
          "I have no training data yet. Add Q&A pairs via the training API, "
          "then call retrain to build the knowledge index."
        )

      self._ensure_vectorstore()
      if self.retriever is None:
        return "Knowledge index is not ready. Please run retrain with your training data."

      self.conversation_history.append(
        {"user": user_input, "timestamp": datetime.now().isoformat()}
      )
      self._update_conversation_context(user_input)

      if self._llm is not None and self._rag_prompt is not None:
        response = self._llm_rag_response(user_input).strip()
      else:
        response = self._retrieval_only_response(user_input)

      if not response:
        response = self._default_fallback(user_input)

      self.conversation_history.append(
        {"bot": response, "timestamp": datetime.now().isoformat()}
      )
      return response

    except Exception as e:
      print(f"Error getting response: {e}")
      return "Sorry, something went wrong while processing your message. Please try again."

  def _retrieval_only_response(self, user_input):
    """No LLM: return the best matching stored answer from FAISS."""
    max_distance = float(os.environ.get("RETRIEVAL_MAX_DISTANCE", "1.15"))
    if self.vectorstore is not None:
      scored = self.vectorstore.similarity_search_with_score(user_input, k=1)
      if not scored:
        return self._default_fallback(user_input)
      best, distance = scored[0]
      if distance > max_distance:
        return self._default_fallback(user_input)
    else:
      docs = self.retriever.invoke(user_input)
      if not docs:
        return self._default_fallback(user_input)
      best = docs[0]

    # Prefer metadata answer; fall back to parsing page_content
    answer = best.metadata.get("answer")
    if answer:
      return answer.strip()

    text = best.page_content
    if "Answer:" in text:
      return text.split("Answer:", 1)[1].strip()
    return text.strip()

  def _format_history(self, max_turns=6):
    lines = []
    for entry in self.conversation_history[-max_turns * 2 :]:
      if "user" in entry:
        lines.append(f"User: {entry['user']}")
      elif "bot" in entry:
        lines.append(f"Assistant: {entry['bot']}")
    return "\n".join(lines) if lines else "(none)"

  def _default_fallback(self, user_input):
    return (
      "I could not find a close match in my knowledge base. "
      "Try rephrasing your question, or add more training data on this topic and retrain."
    )

  def _update_conversation_context(self, user_input):
    words = re.findall(r"[a-zA-Z]{4,}", user_input.lower())
    stop = {
      "what", "when", "where", "which", "that", "this", "with", "from",
      "have", "been", "were", "they", "your", "about", "would", "could",
    }
    for word in words:
      if word not in stop and word not in self.conversation_topics:
        self.conversation_topics.append(word)
    self.conversation_topics = self.conversation_topics[-8:]

  # --------------------------------------------------------- conversation API
  def get_conversation_summary(self):
    if not self.conversation_history:
      return "No conversation yet."
    topics = ", ".join(self.conversation_topics) if self.conversation_topics else "General"
    message_count = len([e for e in self.conversation_history if "user" in e])
    return f"Conversation topics: {topics} | Messages exchanged: {message_count}"

  def clear_conversation(self):
    self.conversation_history = []
    self.user_context = {}
    self.conversation_topics = []
    self.follow_up_questions = []
    self.conversation_mood = "neutral"
    self.user_name = None
    self.conversation_start_time = None
    return "Conversation cleared."

  def adjust_personality(self, trait, value):
    if trait in self.personality_traits and 0 <= value <= 1:
      self.personality_traits[trait] = value
      return f"Adjusted {trait} to {value}"
    return "Invalid trait or value"

  def get_personality_info(self):
    return {
      "traits": self.personality_traits,
      "mood": self.conversation_mood,
      "conversation_length": len(self.conversation_history),
      "langchain_mode": (
        f"{self._llm_provider}_rag" if self._llm else "retrieval_only"
      ),
      "embedding_model": self.EMBEDDING_MODEL,
    }
