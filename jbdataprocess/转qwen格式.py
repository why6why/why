import pandas as pd
import json
import os
# Load the CSV file
csv_file_path = 'C:/Users/Administrator/Desktop/modified_negative.csv'

# 读取CSV文件

df = pd.read_csv(csv_file_path)
# 读取system文本文件内容
system_file_path = 'C:/Users/Administrator/Desktop/negative_prompt.txt'  # 替换为你的文本文件路径
with open(system_file_path, 'r', encoding='utf-8') as file:
    system_content = file.read()

# Check if '模型输入' and '模型输出' columns exist

conversations = []

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    conversation = {
    "instruction": "我是一个物业工作人员，需要你为我筛查群聊对话中，负面情绪强烈的或者一般负面的物业异常事件，让我能够进行向上汇报",
    "input": row["模型输入"],
    "output": row["模型输出"],
    "system": system_content,
  }
    conversations.append(conversation)

# Convert the list of conversations to a JSON string
conversations_json = json.dumps(conversations, ensure_ascii=False, indent=4)

# Save the JSON string to a file
json_file_path = 'C:/Users/Administrator/Desktop/negative_Alpaca.json'
with open(json_file_path, 'w', encoding='utf-8') as file:
    file.write(conversations_json)
