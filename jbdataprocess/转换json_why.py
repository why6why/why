import pandas as pd
import json
# Load the CSV file
csv_file_path = '/home/ubuntu/why/jbdataprocess/negative_train.json'
txt_file_path = '/home/ubuntu/why/jbdataprocess/negative_prompt.txt'
# 读取CSV文件

with open(txt_file_path, 'r', encoding='utf-8') as file:
    # 读取所有内容，并存储在一个字符串变量中
    prompt = file.read()

df = pd.read_json(csv_file_path)

# Check if '模型输入' and '模型输出' columns exist

conversations = []

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    for rows in row['messages']:
        if rows['role'] == 'user':
            input_value=rows['content']
        elif rows['role'] == 'assistant':
            ouput_value=rows['content']
    conversation = {
                "instruction": "你是物业管理者的助手，负责筛查群聊对话中，负面情绪强烈的物业异常事件，进行向上汇报。",
                "input": input_value,
                "output": ouput_value,
                "system": prompt,
            }
    conversations.append(conversation)

# Convert the list of conversations to a JSON string
conversations_json = json.dumps(conversations, ensure_ascii=False, indent=4)

# Save the JSON string to a file
json_file_path = '/home/ubuntu/why/jbdataprocess/negative_train_why.json'
with open(json_file_path, 'w', encoding='utf-8') as file:
    file.write(conversations_json)
