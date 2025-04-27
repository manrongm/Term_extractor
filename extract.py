from pdfminer.high_level import extract_text
text = extract_text("biophysics/biophysics10.pdf")
with open("biophysics/biophysics10.txt", "w") as f:
    f.write(text)