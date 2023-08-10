import os
import json

def process_argument(line):
    # 将一行文本拆分成index和content，例如：'130. Virtual/Augmented Reality Technology should replace ...'
    index, content = line.split('. ', 1)
    index = int(index)
    content = content.strip()
    return {"index": index, "content": content}

def convert_text_to_json(input_file_path, output_file_path):
    debate = {"debate_id": os.path.basename(input_file_path), "arguments": []}
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            argument = process_argument(line)
            debate["arguments"].append(argument)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(debate, output_file, indent=4)

def convert_folder_to_json(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in os.listdir(input_folder_path):
        input_file_path = os.path.join(input_folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(file_name)[0]}.json")
        convert_text_to_json(input_file_path, output_file_path)

if __name__ == "__main__":
    input_folder_path = "input_folder"  # 替换为输入文件夹路径
    output_folder_path = "output_folder"  # 替换为输出文件夹路径

    convert_folder_to_json(input_folder_path, output_folder_path)
