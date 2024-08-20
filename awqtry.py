from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Specify paths and hyperparameters for quantization
model_path = "./.cache/modelscope/hub/qwen/Qwen2-7B"
quant_path = "./QUmodel"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

# 数据校准
dataset = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me who you are."},
    {"role": "assistant", "content": "I am a large language model named Qwen..."}
]
data = []
for msg in dataset:
    msgarray = []
    msgarray.append(msg)
    text = tokenizer.apply_chat_template(msgarray, tokenize=False, add_generation_prompt=False)
    data.append(text.strip())
model.quantize(tokenizer, quant_config=quant_config, calib_data=data)#校准
#保存量化模型
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)