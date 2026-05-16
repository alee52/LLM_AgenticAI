import os
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import accumulate
import math
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from huggingface_hub import login



WORKERS = 5
DEFAULT_SIZE = 3000
model_name = "meta-llama/Llama-3.2-3B"

hf_token = os.environ['HF_TOKEN']
if hf_token:
    print("HuggingFace token found.")
else:
    print("No HuggingFace token found.")

login(hf_token, add_to_git_credential=True)

class Tester:
    

    def __init__(self, data, size=DEFAULT_SIZE, workers=WORKERS):
        self.data = data
        self.size = size
        self.guesses = []
        self.truths = []
        self.is_correct = []
        self.workers = workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def post_process(self, value):
        if isinstance(value, str):
            tokens = self.tokenizer.encode(value, add_special_tokens=False)
            if tokens and tokens[-1] == 128001:
                tokens = tokens[:-1]
            return tokens
        else:
            raise ValueError("Expected a string output from the predictor, got: " + str(value))
        
        

    def run_datapoint(self, i):
        datapoint = self.data[i]
        value = datapoint.completion
        guess = self.post_process(value)
        truth = self.tokenizer.encode(datapoint.category, add_special_tokens=False)
        is_correct = (guess == truth)
        

        return guess, truth, is_correct


    def report(self):
        accuracy = sum(self.is_correct) / len(self.is_correct) * 100 if self.is_correct else 0
        title = f"<br><b>Accuracy:</b> {accuracy:.1f}%"
        print(title)

    def run(self):
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for guess, truth, is_correct in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size
            ):
                self.guesses.append(guess)
                self.truths.append(truth)
                self.is_correct.append(is_correct)
        self.report()


def evaluate(data, size=DEFAULT_SIZE, workers=WORKERS):
    Tester(data, size=size, workers=workers).run()
