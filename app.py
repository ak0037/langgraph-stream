import streamlit as st
from PIL import Image
import os
from langchain.embeddings.openai import OpenAIEmbeddings
st.set_page_config(layout="wide")

from langchain_openai import AzureOpenAIEmbeddings
import os

# Configure Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://abhin-m2ifqtz5-eastus2.openai.azure.com",
    deployment="text-embedding-ada-002",  # Changed from azure_deployment to deployment
    api_key="4f46e8f30eac4a3abedeb41c04b7ab52",
    api_version="2023-05-15"
)

def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  
            file_path = os.path.join(directory_path, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

documents = load_documents('/Workspace/Users/abhinav.katiyar@spaceinventive.com/data/')

# Create text splitter for smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

# Process documents into chunks
texts = text_splitter.split_documents(documents)  # Changed to split_documents instead of manual splitting

# Build vector store
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    location=":memory:",
    collection_name="my_documents",
)

# Configure retriever
retriever = qdrant.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 3
    }
)

st.markdown("""
<style>
    /* Fixed size button container */
    .stButton {
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        height: 100px !important;
        width: 100% !important;
        white-space: normal !important;
        text-align: center !important;
        padding: 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        word-wrap: break-word !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
</style>
""", unsafe_allow_html=True)



# def resize_image(image_path, max_width):
#     image = Image.open(image_path)
#     width_percent = (max_width / float(image.size[0]))
#     new_height = int((float(image.size[1]) * float(width_percent)))
#     resized_image = image.resize((max_width, new_height), Image.LANCZOS)
#     return resized_image

# # Add logo
# logo_path = "images.jpg"
# max_logo_width = 200

# if os.path.exists(logo_path):
#     try:
#         logo = resize_image(logo_path, max_logo_width)
#         st.sidebar.image(logo, use_column_width=True)
#     except Exception as e:
#         st.sidebar.warning(f"Error loading logo: {str(e)}")
# else:
#     st.sidebar.warning("Logo file not found. Please add 'images.jpg' to the same directory as this script.")

# Initialize session state
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""


st.sidebar.title("Filters")
st.sidebar.write("")

# Brand selection
brand = st.sidebar.radio("Select a brand:", ["Cosentyx CEP studies", "Kisqali ATU Studies"])
st.sidebar.write("")

# Country selection based on brand
country_options = {
    "Cosentyx CEP studies": ["Germany", "Japan"],
    "Kisqali ATU Studies": ["Germany", "UK"]
}
st.sidebar.write("")
selected_country = st.sidebar.selectbox("Select Country:", country_options[brand])

# Time period selection based on brand
time_options = {
    "Cosentyx CEP studies": ["Q1 24", "Q2 24"],
    "Kisqali ATU Studies": ["Q3 24", "Q4 24"]
}
st.sidebar.write("")
selected_time = st.sidebar.selectbox("Select Time Period:", time_options[brand])


st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")


# About and Help sections with hover tooltips
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("**About** ℹ️", help=f"This is a Q&A system that answers questions about {brand}.")
with col2:
    st.markdown("**Help** 💡", help="Click on predefined questions or type your own question to get answers. Select filters from sidebar to narrow down results.")


st.title(f"{brand} Q&A System")
st.markdown(f"Selected Country: **{selected_country}** | Time Period: **{selected_time}**")

def get_rag_answer(question: str, brand: str, country: str, time_period: str) -> str:
    with st.spinner("Getting answer..."):
        # This is a placeholder - will be replaced wih RAG implementation
        return f"This is a placeholder response. The RAG system will provide actual answers for:\nQuestion: {question}\nBrand: {brand}\nCountry: {country}\nTime Period: {time_period}"


predefined_questions = {
    "Cosentyx CEP studies": [
        "What is Cosentyx?",
        "What are the main indications for Cosentyx?",
        "What are the common side effects of Cosentyx?",
        "How is Cosentyx administered?"
    ],
    "Kisqali ATU Studies": [
        "What is Kisqali?",
        "What type of breast cancer is Kisqali used to treat?",
        "What are the potential side effects of Kisqali?",
        "How is Kisqali taken?"
    ]
}

# Function to handle question click
def handle_question_click(question):
    st.session_state.current_question = question
    st.session_state.question_input = question
    st.session_state.current_answer = get_rag_answer(
        question, 
        brand, 
        selected_country, 
        selected_time
    )

# Function to truncate text
def truncate_text(text, max_length=50):
    return text if len(text) <= max_length else text[:max_length] + "..."


st.write('')

st.subheader("Predefined Questions")
col1, col2 = st.columns(2)
cols = [col1, col2]


for idx, question in enumerate(predefined_questions[brand]):
    with cols[idx // 2]:
    
        truncated_question = truncate_text(question)
        if st.button(
            truncated_question,
            key=f"q_{idx}",
            help=question,
            use_container_width=True
        ):
            handle_question_click(question)











# Question input and answer section
st.markdown("---")
question_section = st.container()

with question_section:
    st.subheader("Question & Answer")
    

    with st.form(key='question_form'):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            if 'question_input' not in st.session_state:
                st.session_state.question_input = ""
                
            question_input = st.text_input(
                "Enter or select a question:",
                value=st.session_state.question_input,
                key="question_box",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "Get Answer",
                use_container_width=True,
                type="primary"
            )
            
    
    if submit_button:
        if question_input:
            st.session_state.current_question = question_input
            st.session_state.current_answer = get_rag_answer(
                question_input,
                brand,
                selected_country,
                selected_time
            )
        else:
            st.warning("Please enter a question.")

    # Display answer
    if st.session_state.current_answer:
        st.markdown("### Answer")
        st.markdown(st.session_state.current_answer)



# Advanced options

st.markdown("---")
with st.expander("Advanced Options"):
  
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download as CSV", use_container_width=True,
                    help="Download the Q&A results in CSV format"):
          
            st.info("CSV download functionality will be implemented")
    
    with col2:
        if st.button("Download as Word", use_container_width=True,
                    help="Download the Q&A results in Word format"):
            st.info("Word download functionality will be implemented")
    
    st.markdown("---") 
    

    show_steps = st.checkbox("Show Interim Steps", 
                           help="Display intermediate steps in the analysis process")
    if show_steps:
        st.info("Interim steps functionality will be implemented with the RAG system")
      
        st.markdown("```\nInterim steps will appear here\n```")
    
    show_source = st.checkbox("Show Source", 
                            help="Display the source of information used in the answer")
    if show_source:
        st.info("Source information will be available with RAG implementation")
      
        st.markdown("```\nSource information will appear here\n```")



from transformers import AutoModelForCausalLM, AutoTokenizer
import mlflow
import torch

# Start MLflow run
with mlflow.start_run(run_name="llama2_registration"):
    
    # Choose model version
    model_id = "meta-llama/Llama-2-7b-chat-hf"  # or other Llama variants
    
    # Log model details as params
    mlflow.log_params({
        "model_id": model_id,
        "model_type": "chat"
    })
    
    # Download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # Save to Model Registry
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer
        },
        artifact_path="llama2-chat",
        registered_model_name="llama2-chat-model",  # This registers it in Model Registry
        pip_requirements=["torch", "transformers", "accelerate"]
    )

# Get the latest version
model_path = "models:/llama2-chat-model/latest"
