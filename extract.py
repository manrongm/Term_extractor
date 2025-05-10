from pdfminer.high_level import extract_text
text = extract_text("cvpr_data/cvpr15.pdf")
with open("cvpr_data/cvpr15.txt", "w") as f:
    f.write(text)