from customized_agents.agent import Agent
from customized_agents.specialist_agent import SpecialistAgent
from customized_agents.RAG_agent import RAGAgent
from customized_agents.preprocessor import Preprocessor


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.rag = RAGAgent(collection)
        self.preprocessor = Preprocessor()
        self.log("Ensemble Agent is ready")

    def categorize(self, description: str) -> str:
        """
        Run this ensemble model
        """
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        specialist = self.specialist.categorize(rewrite)
        rag = self.rag.categorize(rewrite)

        if specialist == rag:
            combined = specialist
            self.log(f"Ensemble Agent complete - returning {combined}")
            return combined
        else:
            self.log(f"Ensemble Agent - disagreement between models, more information needed to resolve")
            print(specialist)
            print(rag)
            return "Unable to categorize the product with high confidence."