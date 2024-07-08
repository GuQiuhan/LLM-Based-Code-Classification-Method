import pandas as pd
import json
import pickle


# 预处理单个代码片段
def preprocess_data(c):
    # 先处理json格式的数据
    try:
        data = json.loads(c)
    except (json.JSONDecodeError, TypeError):
        data=c

    # 提取代码1和代码2，并拼接
    code1 = data['input'].strip()
    code2 = data['output'].strip()
    code = code1 + code2

        # TODO:
        # 删除开头可能出现的package和import语句，便于构建干净的ast 
        # 删除以package开头的行
        # code = re.sub(r"^(package .*?\n)", "", code, flags=re.MULTILINE)
        # 删除以import开头的行及括号里的内容
        # code = re.sub(r"^(import .*?\(\n(?:\s*.*?\n)*?\))", "", code, flags=re.MULTILINE)

    return code


def convert_to_dataframe():
    # 从 .pkl 文件中加载数据
    with open('out.pkl', "rb") as f:  # 注意使用 "rb" 模式读取二进制文件
        data = pickle.load(f)

    # 创建一个空列表，用于存储转换后的 DataFrame
    data_frames = []

    # 遍历加载的数据，将每一项转换为 DataFrame 类型，并添加 'id', 'code', 'label' 这三列
    index=0
    for i, item in enumerate(data):
        # 创建 DataFrame，并指定列名为 'id', 'code', 'label'
        df = pd.DataFrame({'id': [i], 'code': [item], 'label': [2]}) # 2代表异常数据

        # 将 DataFrame 添加到列表中
        data_frames.append(df)
        index=i

    index=index+1

    with open('original_code.json', "r") as file:
        data2 = file.readlines()

    for line in data2:
        line = line.strip()  # 去除行首行尾的空白字符
        code= preprocess_data(line)
        df = pd.DataFrame({'id': [index], 'code': [code], 'label': [1]}) # 1代表正常数据
        data_frames.append(df)
        index=index+1
    
    # 对 data_frames 进行打乱
    import random
    random.shuffle(data_frames)
    
    print("here")
    # 将所有 DataFrame 拼接成一个大的 DataFrame
    result_df = pd.concat(data_frames, ignore_index=True)
    output_file_path = "programs.pkl"
    with open(output_file_path, "wb") as f:
        pickle.dump(result_df, f)


convert_to_dataframe()


