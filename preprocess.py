import os
import re
import spacy

# load Spacy NLP
nlp = spacy.load("en_core_web_sm")

def clean_text(text):

    text = re.sub(r'\f', ' ', text)                        
    text = re.sub(r'\n+', ' ', text)                        
    text = re.sub(r'\[[0-9]{1,3}\]', '', text)         
    text = re.sub(r'\([0-9]{1,3}\)', '', text)         
    text = re.sub(r'Figure\s*\d+|Table\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\$[^$]*\$|\$\$[^$]*\$\$', '', text)      
    text = re.sub(r'\s+', ' ', text)          
    text = re.sub(r'(references|acknowledgments|acknowledgements|bibliography).*', '', text, flags=re.IGNORECASE) 
    text = re.sub(r'\d{1,3}(\.\d{1,3}){1,3}', '', text)   
    return text.strip()

def preprocess(text_path, output_dir):
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # cleaning text
    cleaned = clean_text(raw_text)

    # spaCy NLP
    doc = nlp(cleaned)

    # sentence result
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    # nonn phrases
    noun_phrases = []
    for chunk in doc.noun_chunks:
        text = chunk.text.strip().lower()
        if len(text.split()) >= 2 and len(text.split()) <= 5:
            noun_phrases.append(text)


    common_cv_terms = {"object detection", "semantic segmentation", "depth estimation", "instance segmentation", "action recognition"}
    noun_phrases.extend(common_cv_terms)

    # output directory
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(text_path))[0]

    # 1：cleaned text
    with open(os.path.join(output_dir, f"{base}_clean.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned)

    # 2. sentences
    with open(os.path.join(output_dir, f"{base}_sentences.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    # 3. noun phrases
    with open(os.path.join(output_dir, f"{base}_noun_phrases.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(set(noun_phrases)))) 

    print(f"✅ finished, saved to {output_dir}")

if __name__ == "__main__":
    input_file = "cvpr_data/cvpr15.txt"
    output_dir = "processed_cvpr"
    preprocess(input_file, output_dir)
