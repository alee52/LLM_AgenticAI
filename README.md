# Product Categorization AI Agent

The goal of this project is to build an automated AI agent that tags and classifies products based on product descriptions and images. When the model’s confidence is low, the agent requests additional information or escalates the case for human supervision.

## Dataset and Curation Process

The dataset was sourced from Hugging Face: the Shopify Product Catalogue dataset, which contains approximately 50,000 products with descriptions, images, and product classifications.

Dataset: https://huggingface.co/datasets/Shopify/product-catalogue

Product descriptions were cleaned and preprocessed using OpenAI’s GPT-4.1-nano. Products without English descriptions were excluded from the project. After filtering and resampling to reduce class imbalance, the final curated dataset contained approximately 28,000 data points.

## Model development

### Fine-Tuned LLM

Llama-3.2-3B was fine-tuned using QLoRA for product category prediction, achieving 77.4% accuracy on the testing set. Lora parameters:

### Frontier Model: GPT-5.1
With medium reason effort, a 73.9% accuracy was achieved.

### Retrieval-Augmented Generation (RAG)

Accuracy with RAG reached 83.9% on test set.
