from typing import List, Dict
from sentence_transformers import SentenceTransformer
from customized_agents.agent import Agent
import os
import re
import math
from tqdm import tqdm
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
from peft import PeftModel


# hf_token = os.environ['HF_TOKEN']
# if hf_token:
#     login(hf_token, add_to_git_credential=True)
#     print("HuggingFace token found and set as environment variable.")
# else:
#     raise ValueError("HF_TOKEN environment variable not found. Please set it to your HuggingFace token.")

class SpecialistAgent(Agent):
    """
    An Agent that runs our fine-tuned LLM
    """
    
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.log("Specialist Agent is initializing")
        # Load the Tokenizer and the Model
        BASE_MODEL = "meta-llama/Llama-3.2-3B"
        PROJECT_NAME = "categorize_products_no_cate"
        HF_USER = "leearum95" 

        LITE_MODE = False

        if LITE_MODE:
            # RUN_NAME = "2026-05-16_15.51.46-lite"
            REVISION = None
        else:
            RUN_NAME = "2026-05-08_18.44.24"
            REVISION = None

        QUANT_4_BIT = True

        PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
        HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"
        

        # Hyper-parameters - QLoRA

        self.device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
        )

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if self.device == "cuda":
            capability = torch.cuda.get_device_capability()
            use_bf16 = capability[0] >= 8

            if QUANT_4_BIT:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                )

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=quant_config,
                device_map="auto",
            )

        else:
            # Mac M-series / CPU path: no bitsandbytes quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            )
            base_model.to(self.device)

        base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Load the fine-tuned model with PEFT
        if REVISION:
            self.predictor = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME, revision=REVISION)
        else:
            self.predictor = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME)
    

    def categorize(self, description: str):
        """
        Make a call to the fine-tuned model to categorize the product described in the input text
        """
        PREFIX = "The category is"

        QUESTION = "What is the category of the following product judging by its title and description?"
        description_new = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        self.log("Specialist Agent is calling remote fine-tuned model")
        inputs = self.tokenizer(description_new,return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.predictor.generate(**inputs,min_new_tokens = 2, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:-1]
        result = self.tokenizer.decode(generated_ids)
        self.log(f"Specialist Agent completed - predicting {result}")
        print("this is the raw response from the Specialist agent", result)
        return result
