import pandas as pd

# # CSV文件路径
# csv_file_path = 'C:/Users/Administrator/Desktop/negative.csv'
# df = pd.read_csv(csv_file_path)
#
# split_content = df['模型输入'].str.split(r'\n').tolist()
#
# # Flatten the list of lists to get all individual rows
# all_rows = [item for sublist in split_content for item in sublist if
#             item]  # Added a check to avoid empty strings
#
# # Combine all rows into a single string with periods as separators
# combined_content = '。'.join(all_rows)
import pandas as pd

# CSV文件路径
csv_file_path = 'C:/Users/Administrator/Desktop/negative.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

if '模型输入' in df.columns:
    # Replace newline characters with periods in the '模型输入' column
    df['模型输入'] = df['模型输入'].str.replace(r'\n', '。', regex=True)
    # Save the modified DataFrame back to a new CSV file to avoid overwriting the original
    modified_csv_file_path = 'C:/Users/Administrator/Desktop/modified_negative.csv'
    df.to_csv(modified_csv_file_path, index=False)




