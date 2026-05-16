import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from tqdm.notebook import tqdm

load_dotenv(override=True)
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-5.1"
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")
top_level_list = {'Bundles', 'Food, Beverages & Tobacco', 'Product Add-Ons', 'Gift Cards', 'Hardware', 'Home & Garden', 'Sporting Goods', 'Electronics', 'Baby & Toddler', 'Uncategorized', 'Apparel & Accessories', 'Furniture', 'Media', 'Toys & Games', 'Religious & Ceremonial', 'Luggage & Bags', 'Cameras & Optics', 'Arts & Entertainment', 'Software', 'Office Supplies', 'Animals & Pet Supplies', 'Vehicles & Parts', 'Health & Beauty', 'Business & Industrial', 'Services'}
SYSTEM_PROMPT = """Predict the category of the product from the given product description. Your answer must be one of the following: {categories}""".format(categories=", ".join(top_level_list))


class Batch:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite = False):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None
        self.batch_id = None
        self.output_file_id = None
        self.done = False
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        body = {
            "model": MODEL,
            "reasoning_effort": "medium",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.summary},
            ]
        }
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)


    def make_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("rb") as f:
            response = openai.files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        response = openai.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self):
        response = openai.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
        return status == "completed"

    def fetch_output(self):
        output_file = str(self.output / self.filename)
        response = openai.files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self):
        output_file = str(self.output / self.filename)
        with open(output_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                completion = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[id].completion = completion
        self.done = True

    @classmethod
    def create(cls, items, lite= False):
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
