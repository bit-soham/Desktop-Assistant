

import os
from openai import OpenAI

# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key="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",
# )

# completion = client.chat.completions.create(
#     model="HuggingFaceTB/SmolLM3-3B:hf-inference",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }
#     ],
# )

# print(completion.choices[0].message)

# "===================================================================================="

import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="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",
)

completion = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM3-3B",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)

# "==========================================================================="

# import os
# import requests

# API_URL = "https://router.huggingface.co/v1/chat/completions"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

# response = query({
#     "messages": [
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }
#     ],
#     "model": "HuggingFaceTB/SmolLM3-3B:hf-inference"
# })

# print(response["choices"][0]["message"])