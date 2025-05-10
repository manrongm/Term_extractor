import re

input_path = "biophysics10_BIO.txt"  # 替换为你的路径
output_path = "c_biophysics10_BIO.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []

# BIO状态维护
prev_tag = "O"
term_buffer = []  # 临时记录连续术语
max_term_len = 4  # 最长术语长度

# 常见错误术语词表（部分可自定义扩展）
common_non_terms = {
    "page", "method", "model", "new", "results", "structure", "is", "are", "was", "were",
    "it", "this", "that", "these", "those", "we", "they", "you", "i", "can", "may",
    "the", "and", "or", "of", "to", "in", "on", "as", "with", "by", "for", "from", "at",
    "TABLE", "Fig", "APPENDIX", "REFERENCES"
}

for i, line in enumerate(lines):
    line = line.strip()

    if not line:
        # 限制连续术语长度
        if len(term_buffer) > max_term_len:
            for j in range(max_term_len, len(term_buffer)):
                term_buffer[j] = term_buffer[j].split()[0] + " O"
        cleaned_lines.extend(term_buffer)
        cleaned_lines.append("")
        prev_tag = "O"
        term_buffer = []
        continue

    parts = line.split()
    if len(parts) != 2:
        continue

    word, tag = parts

    # 清洗：URL、邮箱、单位名、人名
    if re.search(r"(http|\.com|github\.io)", word.lower()):
        continue
    if re.match(r"\w+@\w+\.\w+", word):
        continue
    if re.match(r"^[A-Z][a-z]+[0-9,]*$", word) and tag.startswith("B-"):
        continue
    if word in {"Inria", "CNRS", "France", "Univ", "ENPC", "LIGM"}:
        continue

    # 删除无意义字符或格式化符号
    if re.match(r"^\d{4,}$", word):
        continue
    if re.match(r"^[.,;:()\[\]{}]$", word) and tag != "O":
        continue
    if len(word) == 1 and tag != "O":
        continue

    # 统一大小写
    w_lower = word.lower()

    # 非术语词强制降级为 O
    if w_lower in common_non_terms and tag != "O":
        tag = "O"

    # 修复BIO语法：I-TERM不能单独存在
    if tag == "I-TERM" and prev_tag == "O":
        tag = "O"

    # 累计term序列以做粒度控制
    if tag.startswith("B-"):
        term_buffer = [f"{word} {tag}"]
        prev_tag = tag
    elif tag.startswith("I-"):
        term_buffer.append(f"{word} {tag}")
        prev_tag = tag
    else:
        if term_buffer:
            if len(term_buffer) > max_term_len:
                for j in range(max_term_len, len(term_buffer)):
                    term_buffer[j] = term_buffer[j].split()[0] + " O"
            cleaned_lines.extend(term_buffer)
            term_buffer = []
        cleaned_lines.append(f"{word} {tag}")
        prev_tag = tag

# 写入结果
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

print(f"✅ 清洗完成并修复BIO语法、术语粒度，输出文件：{output_path}")
