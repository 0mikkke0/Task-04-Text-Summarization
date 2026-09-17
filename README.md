# Task 04: Text Summarization with Transformers

This repository contains an abstractive text summarization pipeline built using pre-trained **BART** (`facebook/bart-large-cnn`) via Hugging Face `transformers`.

## Project Overview
The model takes long-form articles or text paragraphs and generates concise, coherent abstractive summaries using a sequence-to-sequence encoder-decoder transformer architecture.

## Project Structure
- `task4_summarization.py`: Script to load the pre-trained summarization pipeline and process text input.

## Requirements
```bash
pip install transformers torch
