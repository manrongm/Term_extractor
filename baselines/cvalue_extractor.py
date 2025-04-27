import math
import pandas as pd
from collections import defaultdict

def load_noun_phrases(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        phrases = [line.strip().lower() for line in f.readlines() if len(line.strip()) > 0]
    return phrases

def count_phrase_frequencies(phrases):
    """
    统计每个词组的出现次数（包括被嵌套的）
    """
    freq = defaultdict(int)
    for phrase in phrases:
        freq[phrase] += 1
    return freq

def compute_cvalue(freq_dict):
    """
    根据频率字典计算 C-Value 分数
    """
    cvalue_scores = {}
    for phrase in freq_dict:
        sub_phrases = [p for p in freq_dict if p != phrase and phrase in p]
        freq = freq_dict[phrase]
        length = len(phrase.split())

        if not sub_phrases:
            # 没有被嵌套
            cvalue = math.log2(length) * freq
        else:
            # 被嵌套情况：减去嵌套中的平均频率
            nested_freq_sum = sum(freq_dict[sub] for sub in sub_phrases)
            cvalue = math.log2(length) * (freq - (nested_freq_sum / len(sub_phrases)))
        
        cvalue_scores[phrase] = round(cvalue, 4)
    return cvalue_scores

def save_results(cvalue_scores, output_path):
    df = pd.DataFrame(list(cvalue_scores.items()), columns=["term", "cvalue_score"])
    df = df.sort_values(by="cvalue_score", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"✅ 已保存 C-Value 排名术语至：{output_path}")

if __name__ == "__main__":
    # 修改为你的实际路径
    input_path = "processed_nlp/nlp1_noun_phrases.txt"
    output_path = "processed_nlp/nlp1_cvalue_terms.csv"

    phrases = load_noun_phrases(input_path)
    freq_dict = count_phrase_frequencies(phrases)
    cvalue_scores = compute_cvalue(freq_dict)
    save_results(cvalue_scores, output_path)
