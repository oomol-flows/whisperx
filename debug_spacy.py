#!/usr/bin/env python3
"""Debug spaCy sentence segmentation"""
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Machine learning algorithms can now analyze vast amounts of data in seconds, providing insights that would take humans months to discover."

doc = nlp(text)

print("Detected sentences:")
for i, sent in enumerate(doc.sents, 1):
    print(f"{i}. [{sent.start_char:3d}-{sent.end_char:3d}] {sent.text}")
