import re

def is_special_structure(text):
    # 定义正则表达式
    pattern = r"-> See \d+(\.\d+)+\."

    # 使用re.match进行匹配
    return bool(re.match(pattern, text))

# 示例文本
text1 = "-> Se 1.1.5.3.7.1.2.1.1.1.1."
text2 = "-> Se 1.1.5.3.7.1.2.1.1.1.1."
if not is_special_structure(text1) and not is_special_structure(text2):
    print("jhahahahha")

print(is_special_structure(text1))  # 应该返回 True
print(is_special_structure(text2))  # 应该返回 False
