from pdfminer.high_level import extract_text
text = extract_text("./d2l-en.pdf")
print(f"Characters extracted: {len(text)}")
print(text[:300])