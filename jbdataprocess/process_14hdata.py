# 对json处理后生成新json，role:user====》user：内容
import pandas as pd
import json
# Load the CSV file
csv_file_path = 'F:/Working/PMAIP/why/datasets/all_negative_Alpaca.json'
txt_file_path = 'F:/Working/PMAIP/why/datasets/negative_prompt.txt'
# 读取CSV文件

with open(txt_file_path, 'r', encoding='utf-8') as file:
    # 读取所有内容，并存储在一个字符串变量中
    prompt = file.read()

df = pd.read_json(csv_file_path)

# Check if '模型输入' and '模型输出' columns exist

conversations = []

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    conversation = {
                "instruction": "",
                "input": row['input'],
                "output": row['output'],
                "system": prompt,
            }
            
    conversations.append(conversation)
print(index)
# Convert the list of conversations to a JSON string
conversations_json = json.dumps(conversations, ensure_ascii=False, indent=4)

# Save the JSON string to a file
json_file_path = 'F:/Working/PMAIP/why/datasets/all_negative_Alpaca_whyte.json'
with open(json_file_path, 'w', encoding='utf-8') as file:
    file.write(conversations_json)
