import re
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class RAGAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    def __init__(self, collection):
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing RAG Agent")
        self.client = OpenAI()
        self.MODEL = "gpt-5.1"
        self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("RAG Agent is ready")

    def make_context(self, similars: List[str], categories: List[str]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param categories: categories of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you listed.\n\n"
        for similar, category in zip(similars, categories):
            message += f"Potentially related product:\n{similar}\nCategory is {category}\n\n"
        return message

    def messages_for(
        self, description: str, similars: List[str], categories: List[str]
    ) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param categories: categories of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        message = f"Classfify the following product. Respond with the category, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, categories)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results["documents"][0][:]
        categories = [m["category"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, categories

    def categorize(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        documents, categories = self.find_similars(description)
        self.log(
            f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products"
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, categories),
            seed=42,
            reasoning_effort="none",
        )
        reply = response.choices[0].message.content
        self.log(f"Frontier Agent completed - predicting {reply}")
        return reply
