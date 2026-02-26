# CMD命令执行llamafactory-cli指令
import subprocess

# 执行CMD命令并获取返回值
result = subprocess.run('llamafactory-cli train /home/ubuntu/githubs/LLaMA-Factory/examples/train_lora/qwen2__5_3b_lora.sft.yaml', shell=True, text=True, capture_output=True)
print("1")
print(result.stdout)  # 打印输出