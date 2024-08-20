
from modelscope import snapshot_download
import torch
# Model Download
# ./home/ubuntu/.cache/modelscope/hub/qwen/Qwen2-7B
# model_dir = snapshot_download(repo_id="qwen/Qwen2-7B", repo_type="model", cache_dir="model/",
#                   local_dir_use_symlinks=False, resume_download=True)
model_dir = "./.cache/modelscope/hub/qwen/Qwen2-7B"



from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
)#此处需要加载模型Loading checkpoint shards
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
prompt = "Give me a short introduction to LMA."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)


inputids = tokenizer.encode(text,return_tensors='pt')#我的添加
attention_mask = torch.ones(inputids.shape,dtype=torch.long,device=device)#我的添加
# Directly use generate() and tokenizer.decode() to get the output.
# Use `max_new_tokens` to control the maximum output length.
generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)