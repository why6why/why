from openai import OpenAI
import os
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="../.cache/modelscope/hub/qwen/Qwen2-7B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about China."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print("Chat response:", chat_response.choices[0].message.content)