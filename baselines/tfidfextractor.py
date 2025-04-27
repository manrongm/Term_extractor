import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_clean_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_noun_phrases(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        phrases = [line.strip().lower() for line in f.readlines()]
    return set(phrases)

def extract_filtered_tfidf_terms(text, noun_phrases=None, ngram_range=(1, 3), top_k=200):
    """
    从文本中提取过滤后的TF-IDF术语候选
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        min_df=1
    )
    X = vectorizer.fit_transform([text])
    tfidf_scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])

    filtered_terms = []
    blacklist = {"figure", "table", "section", "work", "data", "analysis", "result", "score", "method", "model"}

    for term, score in tfidf_scores:
        # 过滤规则
        if len(term.split()) < 2:
            continue
        if re.search(r'\d', term):  # 含数字
            continue
        if any(blk in term for blk in blacklist):
            continue
        if noun_phrases and term not in noun_phrases:
            continue

        filtered_terms.append((term, score))

    sorted_terms = sorted(filtered_terms, key=lambda x: x[1], reverse=True)
    return sorted_terms[:top_k]

def save_terms_to_file(terms, output_path):
    df = pd.DataFrame(terms, columns=["term", "tfidf_score"])
    df.to_csv(output_path, index=False)
    print(f"✅ 已保存前术语至: {output_path}")

if __name__ == "__main__":
    # ✅ 修改路径为你的实际文件路径
    input_text_path = "processed_nlp/nlp1_clean.txt"
    noun_phrase_path = "processed_nlp/nlp1_noun_phrases.txt"  # 可选
    output_csv_path = "processed_nlp/nlp1_tfidf_filtered.csv"

    # 加载数据
    text = load_clean_text(input_text_path)
    noun_phrases = load_noun_phrases(noun_phrase_path)

    # 提取术语
    top_terms = extract_filtered_tfidf_terms(
        text,
        noun_phrases=noun_phrases,
        ngram_range=(1, 3),
        top_k=200
    )

    # 保存输出
    save_terms_to_file(top_terms, output_csv_path)
