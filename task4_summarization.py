import torch
from transformers import pipeline

# Initialize summarization pipeline using BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text to summarize
article = """
Artificial intelligence (AI) is transforming industries worldwide by automating complex tasks, 
enhancing decision-making processes, and driving innovation across healthcare, finance, and education. 
Recent advancements in large language models and generative artificial intelligence have enabled machines 
to perform human-like language understanding, generation, and problem-solving at an unprecedented scale. 
As AI technologies continue to evolve rapidly, organizations and researchers are increasingly focused on 
developing ethical guidelines, safety frameworks, and robust standards to ensure responsible deployment.
"""

# Generate summary
summary = summarizer(article, max_length=50, min_length=20, do_sample=False)

print("--- Original Text Length ---")
print(len(article.split()), "words")

print("\n--- Generated Summary ---")
print(summary[0]['summary_text'])
