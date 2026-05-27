# """
# LangChain agent + Ollama (learning script).

# Real-world pattern:
#   User -> API (FastAPI) -> agent -> tools (DB, weather API, search) -> final answer

# Run:
#   python ollama-learn.py
#   OLLAMA_MODEL=qwen2.5-coder:7b python ollama-learn.py

# Smaller models (e.g. llama3.2) often print fake tool JSON as text instead of
# calling tools. Prefer a model with solid tool support, or use OpenAI in production.
# """

# import json
# import os
# import re
# import subprocess
# import sys

# import requests
# from langchain.agents import create_agent
# from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# DEFAULT_MODEL = "qwen2.5-coder:7b"


# def get_weather(city: str) -> str:
#     """Get current weather for a city using a live API (wttr.in)."""
#     try:
#         resp = requests.get(
#             f"https://wttr.in/{city}",
#             params={"format": "j1"},
#             timeout=10,
#             headers={"User-Agent": "cosmic-ai-learn/1.0"},
#         )
#         resp.raise_for_status()
#         data = resp.json()
#         current = data["current_condition"][0]
#         temp_c = current["temp_C"]
#         humidity = current["humidity"]
#         desc = current["weatherDesc"][0]["value"]
#         return f"{city}: {desc}, {temp_c}°C, humidity {humidity}%"
#     except Exception as exc:
#         return f"Could not fetch weather for {city}: {exc}"


# def list_installed_models() -> list[str]:
#     try:
#         result = subprocess.run(
#             ["ollama", "list"],
#             capture_output=True,
#             text=True,
#             check=True,
#         )
#     except (FileNotFoundError, subprocess.CalledProcessError):
#         return []

#     models = []
#     for line in result.stdout.strip().splitlines()[1:]:
#         if line.strip():
#             models.append(line.split()[0])
#     return models


# def model_is_installed(model_name: str) -> bool:
#     base = model_name.split(":")[0]
#     for tag in list_installed_models():
#         if tag == model_name or tag.split(":")[0] == base:
#             return True
#     return False


# def _try_parse_tool_json(text: str) -> dict | None:
#     """Some local models print tool calls as JSON text instead of using tool_calls."""
#     text = text.strip()
#     if not text.startswith("{"):
#         return None
#     try:
#         data = json.loads(text)
#     except json.JSONDecodeError:
#         match = re.search(r'\{[^{}]*"name"\s*:\s*"get_weather"[^{}]*\}', text, re.DOTALL)
#         if not match:
#             return None
#         try:
#             data = json.loads(match.group(0))
#         except json.JSONDecodeError:
#             return None
#     if data.get("name") == "get_weather":
#         args = data.get("arguments") or data.get("parameters") or {}
#         city = args.get("city")
#         if city:
#             return {"city": city}
#     return None


# def run_tool_fallback(messages: list) -> str | None:
#     """If the agent loop did not run tools, call get_weather ourselves."""
#     if any(isinstance(m, ToolMessage) for m in messages):
#         return None

#     last = messages[-1]
#     if not isinstance(last, AIMessage) or not last.content:
#         return None

#     parsed = _try_parse_tool_json(str(last.content))
#     if not parsed:
#         return None

#     city = parsed["city"]
#     print(f"\n(Fallback) Model printed tool JSON; calling get_weather('{city}') directly...")
#     return get_weather(city)


# def print_agent_trace(messages: list) -> str:
#     """Show what happened (tool calls vs plain text) and return the final answer."""
#     print("\n--- Agent trace ---")
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             print(f"User: {msg.content}")
#         elif isinstance(msg, AIMessage):
#             if msg.tool_calls:
#                 for tc in msg.tool_calls:
#                     print(f"Model -> tool: {tc['name']}({tc['args']})")
#             elif msg.content:
#                 preview = str(msg.content).strip().replace("\n", " ")[:120]
#                 print(f"Model (text only): {preview}...")
#         elif isinstance(msg, ToolMessage):
#             print(f"Tool result: {msg.content}")

#     last = messages[-1]
#     if isinstance(last, AIMessage) and last.content:
#         text = last.content if isinstance(last.content, str) else str(last.content)
#         return text.strip()
#     return "(No final answer — model may not support tools well. Try another OLLAMA_MODEL.)"


# def main():
#     model_name = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

#     if not model_is_installed(model_name):
#         installed = list_installed_models()
#         print(f"Error: Ollama model '{model_name}' is not installed.", file=sys.stderr)
#         if installed:
#             print("Installed:", ", ".join(installed), file=sys.stderr)
#         sys.exit(1)

#     agent = create_agent(
#         model=f"ollama:{model_name}",
#         tools=[get_weather],
#         system_prompt=(
#             "You are a helpful assistant. For weather questions, you MUST call "
#             "the get_weather tool with the city name, then summarize the tool result "
#             "for the user in plain English. Do not invent temperatures."
#         ),
#     )

#     question = os.environ.get(
#         "QUESTION", "What is the current temperature and humidity in Singapore?"
#     )

#     result = agent.invoke({"messages": [{"role": "user", "content": question}]})
#     messages = result["messages"]
#     answer = print_agent_trace(messages)

#     tool_result = run_tool_fallback(messages)
#     if tool_result:
#         print(f"\n--- Final answer (from real API) ---")
#         print(tool_result)
#         return

#     print("\n--- Final answer ---")
#     print(answer)


# if __name__ == "__main__":
#     main()
