from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")
DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "gpt-4.1-nano")
DEFAULT_REASONING_EFFORT = "low" if "gpt-oss" in DEFAULT_MODEL_NAME else None

SYSTEM_PROMPT = """Create a 1 sentence concise description of a product with key features, purpose or use cases. Do not include part numbers."""


class Preprocessor:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        base_url=None,
    ):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        if "ollama" in model_name and not base_url:
            self.base_url = "http://localhost:11434"

    def messages_for(self, text: str) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    def preprocess(self, text: str) -> str:
        messages = self.messages_for(text)
        response = completion(
            messages=messages,
            model=self.model_name,
            reasoning_effort=self.reasoning_effort,
            api_base=self.base_url,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.total_cost += response._hidden_params["response_cost"]
        return response.choices[0].message.content
