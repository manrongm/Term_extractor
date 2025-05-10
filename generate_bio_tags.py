import spacy
import re
from pathlib import Path
from rapidfuzz import fuzz 

nlp = spacy.load("en_core_web_trf")

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\.\,\;\:\(\)\[\]\{\}\'\"]', '', text)
    return text


sentences = Path("processed_cvpr/cvpr15_sentences.txt").read_text(encoding="utf-8").splitlines()
noun_phrases = set(Path("processed_cvpr/cvpr15_noun_phrases.txt").read_text(encoding="utf-8").splitlines())

noun_phrases_norm = set(normalize_text(p) for p in noun_phrases)

unmatched_phrases = set(noun_phrases_norm)
output_lines = []

for sentence in sentences:
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    tags = ["O"] * len(tokens)
    used = [False] * len(tokens)

    i = 0
    while i < len(tokens):
        match_found = False
        for j in range(len(tokens), i, -1):
            span = tokens[i:j]
            span_text = normalize_text(' '.join(span))
            if span_text in noun_phrases_norm and not any(used[i:j]):
                tags[i] = "B-TERM"
                for k in range(i+1, j):
                    tags[k] = "I-TERM"
                for k in range(i, j):
                    used[k] = True
                unmatched_phrases.discard(span_text)
                i = j
                match_found = True
                break
            else:
                for gold in noun_phrases_norm:
                    if fuzz.ratio(span_text, gold) >= 90:
                        if not any(used[i:j]):
                            tags[i] = "B-TERM"
                            for k in range(i+1, j):
                                tags[k] = "I-TERM"
                            for k in range(i, j):
                                used[k] = True
                            unmatched_phrases.discard(gold)
                            i = j
                            match_found = True
                            break
            if match_found:
                break
        if not match_found:
            i += 1

    for token, tag in zip(tokens, tags):
        output_lines.append(f"{token}\t{tag}")
    output_lines.append("")

Path("cvpr15_BIO.txt").write_text("\n".join(output_lines), encoding="utf-8")

print(f"Finished tagging! There are {len(unmatched_phrases)} noun phrases not tagged:")
for phrase in list(unmatched_phrases)[:20]:
    print(f"- {phrase}")
