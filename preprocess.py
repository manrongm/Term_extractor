# import re
# import os
# import spacy

# # Load spaCy English model
# nlp = spacy.load("en_core_web_sm")

# def clean_text(text):
#     """
#     Removes formatting artifacts, citations, and normalizes whitespace.
#     """
#     text = re.sub(r'\f', ' ', text)                        # form feeds
#     text = re.sub(r'\n+', '\n', text)                      # multiple newlines
#     text = re.sub(r'\[[0-9]{1,3}\]', '', text)             # references like [12]
#     text = re.sub(r'\s+', ' ', text)                       # multiple spaces
#     text = re.sub(r'(references|acknowledgments).*', '', text, flags=re.IGNORECASE)
#     return text.strip()

# def extract_sentences_and_phrases(text):
#     """
#     Uses spaCy to tokenize text into sentences and extract noun phrases.
#     """
#     doc = nlp(text)
#     sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
#     noun_phrases = [
#         chunk.text.strip().lower() 
#         for chunk in doc.noun_chunks 
#         if 1 < len(chunk.text.split()) <= 5
#     ]
#     return sentences, noun_phrases

# def process_file(input_path, output_dir):
#     with open(input_path, "r", encoding="utf-8") as f:
#         raw = f.read()

#     cleaned_text = clean_text(raw)
#     sentences, phrases = extract_sentences_and_phrases(cleaned_text)

#     base = os.path.splitext(os.path.basename(input_path))[0]

#     os.makedirs(output_dir, exist_ok=True)

#     # Save cleaned full text
#     with open(os.path.join(output_dir, f"{base}_clean.txt"), "w", encoding="utf-8") as f:
#         f.write(cleaned_text)

#     # Save sentence-level file
#     with open(os.path.join(output_dir, f"{base}_sentences.txt"), "w", encoding="utf-8") as f:
#         f.write("\n".join(sentences))

#     # Save noun phrase list
#     with open(os.path.join(output_dir, f"{base}_phrases.txt"), "w", encoding="utf-8") as f:
#         f.write("\n".join(set(phrases)))  # use set() to remove duplicates

#     print(f"Processed '{input_path}' → Output saved in '{output_dir}'")

# # Example usage
# if __name__ == "__main__":
#     input_file = "nlp1.txt"
#     output_directory = "processed_output"
#     process_file(input_file, output_directory)
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
    input_file = "biophysics/biophysics5.txt"
    output_dir = "processed_biophysics"
    preprocess(input_file, output_dir)
