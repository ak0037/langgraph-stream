# from typing import List
# from langchain_core.prompts import MessagesPlaceholder
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain.agents import AgentExecutor
# from langchain.agents import initialize_agent, AgentType
# from langchain_core.tools import Tool, tool
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk import trace as trace_sdk
# from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
# from opentelemetry.sdk.resources import Resource
# from opentelemetry import trace as trace_api
# from openinference.instrumentation.langchain import LangChainInstrumentor
# from openinference.semconv.resource import ResourceAttributes
# import os

# # Configure OpenTelemetry with project name
# project_name = "lanningdd_loe"  # You can change this to your project name
# resource = Resource.create({ResourceAttributes.PROJECT_NAME: project_name})
# tracer_provider = trace_sdk.TracerProvider(resource=resource)

# # Configure OTLP exporter

# # Replace the OTLP exporter configuration with:
# endpoint = "http://localhost:6006/v1/traces"
# PHOENIX_API_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.ZDiiPZcNka_bxPxfxPeY5SQ0rf5gOCZLOAM7hG-ypjU'
# headers = {"authorization": f"Bearer {PHOENIX_API_KEY}"} if PHOENIX_API_KEY else {}

# span_exporter = OTLPSpanExporter(
#     endpoint=endpoint,
#     headers=headers  # Add this line to pass the authentication headers
# )
# span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
# tracer_provider.add_span_processor(span_processor)

# # Add console exporter for debugging (optional)
# tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# # Set the tracer provider
# trace_api.set_tracer_provider(tracer_provider)

# # Initialize LangChain instrumentation
# LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# # Set environment variable for Groq API key
# os.environ["GROQ_API_KEY"] = "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"
https://abhin-m2ifqtz5-eastus2.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview
4f46e8f30eac4a3abedeb41c04b7ab52
# # Define tools
# @tool
# def search_recipes(query: str) -> str:
#     """Search for recipes based on ingredients or cuisine type."""
#     # Clean up the query by removing quotes and extra whitespace
#     query = query.strip().strip("'\"").lower()
    
#     # Expanded recipe database with more options and variations
#     recipes = {
#         "pasta": "Classic Spaghetti: pasta, tomato sauce, garlic, basil",
#         "vegetarian": "Vegetable Stir Fry: tofu, broccoli, carrots, soy sauce, ginger, garlic",
#         "vegetarian dinner": "Vegetable Stir Fry: tofu, broccoli, carrots, soy sauce, ginger, garlic",
#         "quick": "15-minute Quesadillas: tortillas, cheese, beans, salsa",
#         "spicy": "Spicy Thai Curry: coconut milk, curry paste, tofu, vegetables, rice",
#         "healthy": "Quinoa Bowl: quinoa, chickpeas, sweet potato, kale, tahini dressing",
#         "dinner": "Grilled Chicken: chicken breast, herbs, lemon, olive oil, vegetables",
#         "breakfast": "Breakfast Bowl: oatmeal, banana, honey, nuts, berries"
#     }
    
#     # Search for exact matches first
#     if query in recipes:
#         return recipes[query]
    
#     # If no exact match, search for partial matches
#     for key, value in recipes.items():
#         if query in key or any(term in value.lower() for term in query.split()):
#             return value
            
#     return f"No recipes found for '{query}'. Available categories: vegetarian, quick, spicy, healthy, breakfast, dinner, pasta"

# @tool
# def analyze_nutrition(recipe: str) -> str:
#     """Analyze the nutritional content of a recipe."""
#     # Extract recipe name and create more specific nutritional information
#     recipe_name = recipe.split(":")[0].strip().lower()
    
#     nutrition_database = {
#         "vegetable stir fry": "Nutritional analysis: 300 calories per serving, 15g protein, 35g carbs, 12g fat, high in vitamins A and C",
#         "classic spaghetti": "Nutritional analysis: 450 calories per serving, 12g protein, 65g carbs, 15g fat, good source of lycopene",
#         "15-minute quesadillas": "Nutritional analysis: 400 calories per serving, 18g protein, 45g carbs, 20g fat, good source of calcium",
#         "spicy thai curry": "Nutritional analysis: 380 calories per serving, 14g protein, 40g carbs, 18g fat, high in vitamin C",
#         "quinoa bowl": "Nutritional analysis: 350 calories per serving, 13g protein, 50g carbs, 10g fat, high in fiber",
#         "grilled chicken": "Nutritional analysis: 320 calories per serving, 35g protein, 10g carbs, 15g fat, high in protein",
#         "breakfast bowl": "Nutritional analysis: 280 calories per serving, 8g protein, 45g carbs, 8g fat, high in fiber"
#     }
    
#     return nutrition_database.get(
#         recipe_name,
#         f"Generic nutritional analysis for {recipe_name}: Approximately 400 calories per serving, 15g protein, 45g carbs, 20g fat"
#     )

# @tool
# def generate_shopping_list(recipe: str) -> List[str]:
#     """Generate a shopping list from a recipe."""
#     try:
#         # Extract ingredients from the recipe string
#         ingredients = recipe.split(": ")[1].split(", ")
        
#         # Add quantities (this would come from a real database in a production system)
#         quantities = {
#             "pasta": "1 pound",
#             "tomato sauce": "24 oz",
#             "garlic": "3 cloves",
#             "basil": "1 bunch",
#             "tofu": "14 oz block",
#             "broccoli": "2 cups",
#             "carrots": "2 medium",
#             "soy sauce": "1/4 cup",
#             "ginger": "1 inch piece",
#             "tortillas": "8 count",
#             "cheese": "2 cups shredded",
#             "beans": "15 oz can",
#             "salsa": "16 oz jar"
#         }
        
#         # Create formatted shopping list with quantities
#         shopping_list = []
#         for ingredient in ingredients:
#             quantity = quantities.get(ingredient, "as needed")
#             shopping_list.append(f"{ingredient} ({quantity})")
            
#         return shopping_list
        
#     except Exception as e:
#         return [f"Could not parse recipe format. Please provide recipe in format 'Recipe Name: ingredient1, ingredient2, ...'"]

# # Create the Groq LLM
# llm = ChatGroq(
#     model_name="mixtral-8x7b-32768",
#     temperature=0.7
# )

# # Create tools list
# tools = [
#     search_recipes,
#     analyze_nutrition,
#     generate_shopping_list
# ]

# # Initialize the agent with ZERO_SHOT_REACT_DESCRIPTION type
# agent_executor = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=5  # Increased from 3 to 5 to allow for more complex reasoning
# )

# class MealPlanningAssistant:
#     def __init__(self):
#         self.chat_history = []
        
#     async def process_request(self, user_input: str) -> str:
#         """Process a user request through the agent."""
#         # Add system context to the input
#         enhanced_input = f"""Help plan a meal by following these steps:
# 1. Search for appropriate recipes using the search_recipes tool
# 2. Once you find a recipe, analyze its nutrition using the analyze_nutrition tool
# 3. Finally, generate a shopping list using the generate_shopping_list tool

# Current request: {user_input}

# Previous conversation context:
# {' '.join([msg.content for msg in self.chat_history[-4:] if msg.content])}"""

#         try:
#             response = await agent_executor.ainvoke(
#                 {
#                     "input": enhanced_input
#                 }
#             )
            
#             # Update chat history
#             self.chat_history.extend([
#                 HumanMessage(content=user_input),
#                 AIMessage(content=response["output"])
#             ])
            
#             return response["output"]
#         except Exception as e:
#             error_msg = f"I encountered an error: {str(e)}. Let me try to provide a direct response:\n"
#             # Attempt direct tool usage as fallback
#             try:
#                 recipe_result = search_recipes(user_input)
#                 nutrition = analyze_nutrition(recipe_result)
#                 shopping = generate_shopping_list(recipe_result)
                
#                 fallback_response = f"{error_msg}\nHere's what I found:\n\nRecipe: {recipe_result}\n\n{nutrition}\n\nShopping List:\n" + "\n".join(shopping)
#                 return fallback_response
#             except:
#                 return f"{error_msg}I apologize, but I'm having trouble processing your request. Could you please rephrase it or specify what kind of meal you're interested in?"

# # Example usage
# if __name__ == "__main__":
#     import asyncio
    
#     async def main():
#         assistant = MealPlanningAssistant()
        
#         print("Starting conversation...")
#         response = await assistant.process_request(
#             "I want to cook something vegetarian for dinner. Can you help me plan?"
#         )
#         print("\nResponse:", response)
        
#         print("\nAsking follow-up question...")
#         response = await assistant.process_request(
#             "What if I want to make it spicier?"
#         )
#         print("\nFollow-up response:", response)

#     asyncio.run(main())


from typing import List
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool, tool
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace as trace_api
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes

# Import RAG-specific and PDF-specific components
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import os

# Configure OpenTelemetry
project_name = "rag_pdf_assistant"
resource = Resource.create({ResourceAttributes.PROJECT_NAME: project_name})
tracer_provider = trace_sdk.TracerProvider(resource=resource)

endpoint = "http://localhost:6006/v1/traces"
PHOENIX_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6NCJ9.ZEb_RGQYLiYufKs0pnkW4ks3iLDV0Xby9GRIeVoWGYE'
headers = {"authorization": f"Bearer {PHOENIX_API_KEY}"} if PHOENIX_API_KEY else {}

# Configure only the OTLP exporter
span_exporter = OTLPSpanExporter(endpoint=endpoint)
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor)

# Set the tracer provider and instrument LangChain
trace_api.set_tracer_provider(tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Set API keys
os.environ["GROQ_API_KEY"] = "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"

class PDFProcessor:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # Adjusted chunk size for PDFs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_single_pdf(self, pdf_path: str):
        """Load a single PDF file."""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            print(f"Loaded {len(pages)} pages from {pdf_path}")
            return pages
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {str(e)}")
            return []
    
    def process_pdfs(self):
        """Process all PDFs in the directory."""
        try:
            if not os.path.exists(self.docs_dir):
                os.makedirs(self.docs_dir)
                print(f"Created documents directory at {self.docs_dir}")
                raise ValueError(f"No PDFs found in {self.docs_dir}. Please add some PDF files.")
            
            # Get all PDF files in directory
            pdf_files = [
                os.path.join(self.docs_dir, f) 
                for f in os.listdir(self.docs_dir) 
                if f.lower().endswith('.pdf')
            ]
            
            if not pdf_files:
                raise ValueError(f"No PDF files found in {self.docs_dir}")
            
            print(f"Found {len(pdf_files)} PDF files")
            
            # Load all PDFs
            all_pages = []
            for pdf_file in pdf_files:
                pages = self.load_single_pdf(pdf_file)
                all_pages.extend(pages)
            
            if not all_pages:
                raise ValueError("No content could be extracted from PDFs")
            
            # Split documents
            print("Splitting documents into chunks...")
            splits = self.text_splitter.split_documents(all_pages)
            print(f"Created {len(splits)} text chunks")
            
            # Create vector store
            print("Creating vector store...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            print("Vector store created and persisted")
            
            return vectorstore
            
        except Exception as e:
            print(f"Error processing PDFs: {str(e)}")
            raise



class RAGAssistant:
    def __init__(self, docs_dir: str = "./documents"):
        """Initialize the RAG assistant with PDF directory."""
        self.chat_history = []
        self.tracer = trace_api.get_tracer(__name__)
        
        try:
            # Initialize PDF processor
            print("\nInitializing PDF processor...")
            pdf_processor = PDFProcessor(docs_dir)
            
            print("\nProcessing PDFs...")
            self.vectorstore = pdf_processor.process_pdfs()
            
            # Initialize LLM
            print("\nInitializing LLM...")
            self.llm = ChatGroq(
                model_name="mixtral-8x7b-32768",
                temperature=0.7
            )
            
            # Create memory with explicit output key
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"  # Explicitly specify the output key
            )
            
            # Create retrieval chain with custom prompt
            print("\nCreating retrieval chain...")
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={
                        "k": 3,
                    }
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            print("\nRAG Assistant initialization complete!")
            
        except Exception as e:
            print(f"\nError initializing RAG Assistant: {str(e)}")
            raise
    
    async def process_request(self, user_input: str) -> str:
        """Process a user request using RAG."""
        with self.tracer.start_as_current_span("process_rag_request") as span:
            span.set_attribute("user_input", user_input)
            
            try:
                # Get response from chain
                response = await self.chain.ainvoke({
                    "question": user_input
                })
                
                # Format response with sources
                formatted_response = response["answer"]
                
                # Add source citations with page numbers
                if "source_documents" in response:
                    sources = []
                    seen_sources = set()  # Track unique sources
                    for doc in response["source_documents"]:
                        page_num = doc.metadata.get('page', 'Unknown page')
                        file_name = os.path.basename(doc.metadata.get('source', 'Unknown file'))
                        source_key = f"{file_name}-{page_num}"
                        if source_key not in seen_sources:
                            sources.append(f"\n- {file_name} (Page {page_num})")
                            seen_sources.add(source_key)
                
                # Update chat history
                self.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=formatted_response)
                ])
                
                span.set_attribute("success", True)
                return formatted_response
                
            except Exception as e:
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                return f"I encountered an error processing your request: {str(e)}"

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        try:
            print("Starting RAG Assistant...")
            
            # Specify the documents directory
            docs_dir = "./documents"
            
            # Create assistant
            assistant = RAGAssistant(docs_dir)
            
            print("\nStarting conversation...")
            response = await assistant.process_request(
                "What are the main topics discussed in these PDFs?"
            )
            print("\nResponse:", response)
            
            print("\nAsking follow-up question...")
            response = await assistant.process_request(
                "Can you provide more details about one of these topics?"
            )
            print("\nFollow-up response:", response)
            
        except Exception as e:
            print(f"\nAn error occurred in the main program: {str(e)}")
            raise

    asyncio.run(main())
