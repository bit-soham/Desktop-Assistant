# # !pip install llama-cpp-python

# from llama_cpp import Llama

# llm = Llama.from_pretrained(
# 	repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
# 	filename="qwen2.5-14b-instruct-fp16-00001-of-00008.gguf",
# )

# llm.create_chat_completion(
# 	messages = [
# 		{
# 			"role": "user",
# 			"content": "What is the capital of France?"
# 		}
# 	]
# )

# from llama_cpp import Llama

# MODEL_PATH = r"C:\Users\bitso\models\llama-3.2-1b-q4_K_M.gguf"  # or your granite file

# llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8)

# output = llm.create_chat_completion(
#     messages=[{"role": "user", "content": "What is the capital of France?"}],
#     max_tokens=50
# )

# print(output["choices"][0]["message"]["content"])
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
messages = [
    {"role": "system", "content": "you are a human named david"},
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))