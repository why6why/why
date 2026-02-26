# 输入yaml文件网格搜索后生成多个yaml新文件

import yaml

# 读取YAML文件
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


# 修改YAML文件中的参数
def modify_yaml(data, **kwargs):
    for key, value in kwargs.items():
        data[key] = value
        print('1')
    return data

# 将修改后的内容写入新的YAML文件
def write_yaml(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

# 主函数
def main():
    input_file = 'try/input.yaml'  # 输入的YAML文件路径
    output_file = 'try/output.yaml'  # 输出的YAML文件路径
    
    # 读取YAML文件
    data = read_yaml(input_file)
    
    # 修改参数，这里以修改名为"parameter"的参数为例
    data = modify_yaml(data, parameter='1')
    
    # 将修改后的数据写入新的YAML文件
    # write_yaml(data, output_file)

if __name__ == '__main__':
    main()