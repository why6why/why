# 遍历某文件夹下all_results.json生成csv

import os
import json
import pandas as pd

all_param = []
all_files_content = {}

def read_all_files_in_directory(directory_path):
    entries = os.listdir(directory_path)
    # 遍历每个条目
    for files in entries:
        full_path = os.path.join(directory_path, files)
        file_path = full_path+'/all_results.json' 
        # print(i)
        try:
            # 打开文件并读取内容
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                line_name = [files]
                line_param = [value for key, value in data.items()]
                line_param = line_name + line_param
                all_param.append(line_param)
                # all_files_content[file_path] = content
        except Exception as e:
            # 如果文件无法读取，记录错误
            all_files_content[file_path] = f"Error reading file: {e}"
    df=pd.DataFrame(all_param)
    output_dir = directory_path + 'all_loss.csv'
    df.to_csv(output_dir ,index=False,encoding='utf8')
    return all_files_content    



# 示例路径，这里需要替换为实际的路径
directory_path = '/home/ubuntu/githubs/LLaMA-Factory/outputparam/'
# 请替换上面的路径为你要遍历的实际路径
read_all_files_in_directory(directory_path)