from typing import Optional, List, Dict
from customized_agents.agent import Agent


from customized_agents.ensemble_agent import EnsembleAgent
# from customized_agents.manager_agent import ManagerAgent
# from customized_agents.customer_agent import CustomerAgent
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
        print(f"Notification sent to manager for product: {description}")

        return "I am not sure how to categorize your product. I have sent a notification to the store manager to review this product and categorize it appropriately."

    

    def get_tools(self):
        """
        Return the json for the tools to be used
        """
        categorize_function = {
            "name": "categorize_function",
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
            "name": "notify_function",
            "description": "Notify a store manager about an item with an uncertain category that needs manual categorization.",
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
        
        return [
            {"type": "function", "function": categorize_function},
            {"type": "function", "function": notify_function},
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
            print("DEBUG: ClassifyingAgent LLM called tool:", tool_name)
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    

    def classify(self, product_description: str):
        self.log("Classifying Agent is kicking off a run")
        

        # tools = [{"type": "function", "function": categorize_function},{"type": "function", "function": notify_function}]
        tools = self.get_tools()

        system_message = "You are a categorization agent who helps business owners upload their products onto the platform. Use categorize_function first to categorize the product. Do not make up the category if you are unsure. Do not suggest subcategories. Return only the highest-level categories. If categorize_function returns that it is unable to categorize the product with high confidence, you must call notify_function to fetch a store manager instead of answering directly."
        user_message = f"""Use your tool to categorize the product into a relevant category. Here is the product description: {product_description}.
        """          
       
        messages = [{"role": "system", "content": system_message}] +  [{"role": "user", "content": user_message}]
        response = self.openai.chat.completions.create(model=self.MODEL, messages=messages, tools=tools)

        tool_call_rounds = 0
        while response.choices[0].finish_reason == "tool_calls" and tool_call_rounds < 3:
            tool_call_rounds += 1
            message = response.choices[0].message
            tool_responses = self.handle_tool_call(message)
            messages.append(message.model_dump(exclude_none=True))
            messages.extend(tool_responses)
            response = self.openai.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=tools,
            )

        
        reply = response.choices[0].message.content
        self.log(f"Product Registration Agent completed with: {reply}")
        # print("this is the final response from the classify function", reply)
        return reply
