from typing import Optional, List, Dict

import openai
from agents.agent import Agent


from agents.ensemble_agent import EnsembleAgent
# from agents.manager_agent import ManagerAgent
# from agents.customer_agent import CustomerAgent
from openai import OpenAI
import json


class ClassifyingAgent(Agent):
    name = "ProductRegistrationAgent"
    color = Agent.GREEN
    MODEL = "gpt-5.1"

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Product Registration Agent is initializing")
        self.ensemble = EnsembleAgent(collection)
        self.openai = OpenAI()
        self.log("Product Registration Agent is ready")


    def categorize(self, description: str) -> str:
        """
        Run the tool to categorize an item
        """
        self.log("Autonomous Sorting Agent is categorizing item")
        category = self.ensemble.categorize(description)
        print("this is the category returned from the ensemble", category)
        return f"The category of {description} is {category}"

    def fetch_manager(self, description: str):
        """
        Submit a request to managers for product reviews.
        """
        # ACTION: send email to manager with product details and request for categorization
        # this need id as well to track the request and link it back to the product in question
        self.log("Autonomous Planning agent has escalated a product to the manager")

        return "I am not sure how to categorize your product. I have sent a notification to the store manager to review this product and categorize it appropriately."

    

    def get_tools(self):
        """
        Return the json for the tools to be used
        """
        return [
            {"type": "function", "function": self.categorize_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        """
        Actually call the tools associated with this message
        """
        mapping = {
            "categorize_function": self.categorize,
            "notify_function": self.fetch_manager,
        }

        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    

    def classify(self, product_description: str):
        self.log("Classifying Agent is kicking off a run")
        categorize_function = {
            "name": "categorize_item",
            "description": "Given the description of an item, categorize it into a relevant category",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The description of the item to be categorized",
                    },
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        }

        notify_function = {
            "name": "fetch_manager",
            "description": "Send the user a push notification to the manager's about an item with uncertained category that they should categorize.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The description of the item to be categorized",
                    }
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        }

        tools = [{"type": "function", "function": categorize_function},{"type": "function", "function": notify_function}]

        system_message = "You are an categorization agent who helps bussiness owners upload their products onto the platform. Suggest the correct category for products the bussiness owner is trying to upload. Do not make up the category if you are unsure. Do not suggest subcategories. Return only the highest-level categories. If you come across a product that is difficult to categorize, fetch a store manager."
        user_message = f"""Use your tool to categorize the product into a relevant category. Here is the product description: {product_description}.
        """          
       
        messages = [{"role": "system", "content": system_message}] +  [{"role": "user", "content": user_message}]
        response = openai.chat.completions.create(model=self.MODEL, messages=messages, tools=tools)

        if response.choices[0].finish_reason=="tool_calls":
            message = response.choices[0].message
            response = self.handle_tool_call(message)
            messages.append(message.model_dump(exclude_none=True))

            messages.extend(response)
            response = openai.chat.completions.create(model=self.MODEL, messages=messages)

        
        reply = response.choices[0].message.content
        self.log(f"Product Registration Agent completed with: {reply}")
        # print("this is the final response from the classify function", reply)
        return reply


