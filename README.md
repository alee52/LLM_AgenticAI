# Product Categorization AI Agent

An agentic product-classification assistant designed to support an e-commerce
product-registration workflow. It classifies products from their descriptions,
combines multiple model predictions, and escalates uncertain or disputed
classifications for human review.

The broader goal is to support product tagging from descriptions and images.
The current implementation focuses on the category-selection step. A customer
fills out a separate product-registration template and talks to the agent when
they need to classify the product. The next planned phase will expand the agent
to support the complete product-registration process.

## Project Goals

This project explores how an AI agent can support the classification portion of
product registration:

- Receive a product description from a customer completing a registration form.
- Normalize the description before classification.
- Predict a high-level product category.
- Compare predictions from a fine-tuned specialist model and a
  retrieval-augmented frontier model.
- Avoid inventing a category when confidence is low.
- Request more information or escalate uncertain cases to a store manager.
- Let a customer dispute a suggested category and request human review.

## Dataset and Curation

The source data is the
[Shopify Product Catalogue dataset](https://huggingface.co/datasets/Shopify/product-catalogue)
from Hugging Face. It contains approximately 50,000 products with descriptions,
images, and product classifications.

The curation pipeline:

1. Cleaned and preprocessed product descriptions with OpenAI GPT-4.1-nano.
2. Excluded products without English descriptions.
3. Filtered and resampled the data to reduce class imbalance.
4. Produced a final curated dataset of approximately 28,000 examples.

The data preparation and experimentation notebooks include:

- `data_curation.ipynb`
- `data_preprocessing.ipynb`
- `prepare_data_fine_tunning.ipynb`
- `Fine_tuning_Llama.ipynb`

## Model Development and Results

### Fine-Tuned Specialist Model

The specialist model is based on `meta-llama/Llama-3.2-3B` and was fine-tuned
for product category prediction using QLoRA and PEFT.

Reported test accuracy: **77.4%**

The runtime supports:

- CUDA with 4-bit `bitsandbytes` quantization.
- Apple Silicon through MPS without `bitsandbytes`.
- CPU execution as a fallback.

Fine-tuning configuration and LoRA experiments are recorded in the training
notebooks.

### Frontier Model

GPT-5.1 was evaluated as a standalone frontier model with medium reasoning
effort.

Reported test accuracy: **73.9%**

### Retrieval-Augmented Generation

The RAG classifier embeds a product description with
`sentence-transformers/all-MiniLM-L6-v2`, retrieves five similar products from
a persistent Chroma collection, and supplies their categories as context to
GPT-5.1.

Reported test accuracy: **83.9%**

### Ensemble Strategy

The ensemble runs both the fine-tuned specialist model and the RAG model after
preprocessing the product description.

- If both models return the same category, the category is accepted.
- If they disagree, the ensemble returns an unable-to-categorize result.
- The classifying agent can then call its manager-notification tool for human
  review instead of fabricating a category.

## Agent Architecture

The main implementation lives in `customized_agents/`.

### `ClassifyingAgent`

Coordinates categorization and escalation. It exposes two internal tools to an
LLM:

- `categorize_function`: runs the ensemble classifier.
- `notify_function`: simulates notifying a store manager.

The LLM first calls the categorization tool. If the ensemble reports that it
cannot classify the product with high confidence, the LLM can make a subsequent
tool call to notify a manager.

### `EnsembleAgent`

Preprocesses the product text and compares predictions from:

- `SpecialistAgent`
- `RAGAgent`

Only an exact agreement is currently treated as a high-confidence result.

### `SpecialistAgent`

Loads the fine-tuned Llama 3.2 3B adapter from Hugging Face and generates a
high-level product category locally.

### `RAGAgent`

Queries the Chroma product vector store, constructs retrieval context, and asks
GPT-5.1 to classify the product.

### Customer-Facing Classification Agent

`agent_deployment.ipynb` uses the OpenAI Agents SDK to wrap the custom
classifier as a tool for a customer-facing assistant named Lana. The assistant
is intended to appear alongside, or be launched from, a product-registration
template.

The assistant can:

- Ask the customer for a product description.
- Delegate classification to the custom ensemble.
- Return the proposed category.
- Call a simulated `ping_manager` tool when the customer disputes the result.

The current agent does not collect or manage every field in the product
registration template. Its responsibility is limited to helping determine the
category and handling classification-related escalation.

The manager tools currently print/log an escalation and return a customer-facing
message. They do not yet send a real email, notification, or support ticket.

## Repository Structure

```text
.
├── agent_deployment.ipynb       # OpenAI Agents SDK + Gradio application
├── customized_agents/           # Custom classification agent implementation
│   ├── classifying_agent.py
│   ├── ensemble_agent.py
│   ├── specialist_agent.py
│   ├── RAG_agent.py
│   └── preprocessor.py
├── products_vectorstore/        # Persistent Chroma product collection
├── data_prep/                   # Dataset parsing and evaluation helpers
├── model_test/                  # Batch and RAG evaluation scripts
├── archive/                     # Earlier experiments and notebook versions
├── Fine_tuning_Llama.ipynb
├── data_curation.ipynb
├── data_preprocessing.ipynb
├── ensemble_models.ipynb
├── frontier_models.ipynb
├── rag.ipynb
├── pyproject.toml
└── requirements.txt
```

## Setup

### Requirements

- Python 3.11 or newer
- An OpenAI API key
- Access to the Hugging Face base model and fine-tuned adapter
- Enough memory to load Llama 3.2 3B locally

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_hugging_face_token
```

`HF_TOKEN` may be required when the base model or adapter requires authenticated
Hugging Face access.

Install the project dependencies with `uv`:

```bash
uv sync
```

Alternatively, install the requirements into an existing environment:

```bash
pip install -r requirements.txt
```

The project uses both:

- `agents`: the package installed by `openai-agents`
- `customized_agents`: this repository's custom agent package

Avoid creating another top-level local package named `agents`, because it will
shadow the OpenAI Agents SDK during import.

## Running the Application

1. Open `agent_deployment.ipynb` in VS Code or Jupyter.
2. Select the project Python environment.
3. Restart the kernel after changing agent source files.
4. Run the cells from top to bottom.
5. Open the local Gradio URL printed by the final cell.

The first agent initialization may take time because it loads the tokenizer,
base model, PEFT adapter, sentence-transformer model, and Chroma collection.

## Example Interaction

```text
Customer:
I want to sell a wooden spatula that is not dishwasher safe.

Agent:
The suggested category is Home & Garden.
```

For an ambiguous product, the specialist and RAG models may disagree. The
classifying agent can then invoke the manager-notification tool and explain that
the product requires manual review.

If the customer disagrees with a successful classification, the customer-facing
agent can call `ping_manager` to simulate an additional review request.

## Evaluation

The repository includes notebooks and scripts for:

- Fine-tuned model evaluation
- Frontier-model evaluation
- RAG evaluation
- Batch testing
- Ensemble experiments

Reported results:

| Approach | Test accuracy |
|---|---:|
| Fine-tuned Llama 3.2 3B with QLoRA | 77.4% |
| GPT-5.1 with medium reasoning effort | 73.9% |
| Retrieval-augmented generation | 83.9% |

These figures come from the existing project experiments and may depend on the
specific curated split, prompts, model revisions, and evaluation procedure.

## Current Limitations

- The active workflow primarily classifies text descriptions; image
  classification remains part of the broader project goal.
- The agent currently supports only the classification step of a larger product
  registration process.
- Ensemble confidence is based on exact agreement between two model outputs.
- Manager notification is simulated.
- Model output normalization is limited, so formatting differences can appear
  as disagreement.
- Local model startup is resource intensive.
- The Gradio application is currently notebook-based rather than packaged as a
  standalone service.

## Potential Next Steps

- Expand the agent from category selection into an end-to-end product
  registration assistant.
- Add image-based product understanding.
- Replace simulated manager notifications with email, ticketing, or queue
  integrations.
- Integrate the classifier with a complete product-registration form and
  production database.
- Normalize category labels before ensemble comparison.
- Add confidence scoring and richer disagreement resolution.
- Move the notebook workflow into a tested application module or API.
