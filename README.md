# Product Categorization AI Agent

The goal of this project is to build an automated AI agent that tags and classifies products based on product descriptions and images. When the model’s confidence is low, the agent requests additional information or escalates the case for human supervision.

## Dataset and Curation Process

The dataset was sourced from Hugging Face: the Shopify Product Catalogue dataset, which contains approximately 50,000 products with descriptions, images, and product classifications.

Dataset: https://huggingface.co/datasets/Shopify/product-catalogue

Product descriptions were cleaned and preprocessed using OpenAI’s GPT-4.1-nano. Products without English descriptions were excluded from the project. After filtering and resampling to reduce class imbalance, the final curated dataset contained approximately 28,000 data points.

## Experiment

### Fine-Tuned Llama-3.2-3B

Llama-3.2-3B was fine-tuned using QLoRA for product category prediction, achieving 77% accuracy on the evaluation set.

