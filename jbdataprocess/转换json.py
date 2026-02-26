import pandas as pd
import json
# Load the CSV file
csv_file_path = 'C:/Users/Administrator/Desktop/modified_negative.csv'

# 读取CSV文件

df = pd.read_csv(csv_file_path)


# Check if '模型输入' and '模型输出' columns exist

conversations = []

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    conversation = {
        "messages": [
            {
                "role": "system",
                "content": "你是物业管理者的助手，负责筛查群聊对话中，负面情绪强烈的物业异常事件，进行向上汇报。"
            },
            {
                "role": "user",
                "content": row['模型输入']
            },
            {
                "role": "assistant",
                "content": row['模型输出']
            }
        ]
    }
    conversations.append(conversation)

# Convert the list of conversations to a JSON string
conversations_json = json.dumps(conversations, ensure_ascii=False, indent=4)

# Save the JSON string to a file
json_file_path = 'C:/Users/Administrator/Desktop/negative.json'
with open(json_file_path, 'w', encoding='utf-8') as file:
    file.write(conversations_json)
