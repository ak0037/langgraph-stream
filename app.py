import streamlit as st
from PIL import Image
import os

st.set_page_config(layout="wide")

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

# Brand selection
brand = st.sidebar.radio("Select a brand:", ["Cosentyx CEP studies", "Kisqali ATU Studies"])

# Country selection based on brand
country_options = {
    "Cosentyx CEP studies": ["Germany", "Japan"],
    "Kisqali ATU Studies": ["Germany", "UK"]
}
selected_country = st.sidebar.selectbox("Select Country:", country_options[brand])

# Time period selection based on brand
time_options = {
    "Cosentyx CEP studies": ["Q1 24", "Q2 24"],
    "Kisqali ATU Studies": ["Q3 24", "Q4 24"]
}
selected_time = st.sidebar.selectbox("Select Time Period:", time_options[brand])

st.sidebar.header("About")
st.sidebar.info(
    f"This is a Q&A system that answers questions about {brand}."
)

# Main content
st.title(f"{brand} Q&A System")
st.markdown(f"Selected Country: **{selected_country}** | Time Period: **{selected_time}**")


def get_rag_answer(question: str, brand: str, country: str, time_period: str) -> str:
    with st.spinner("Getting answer..."):
        # This is a placeholder - will be repaced with actual RAG implementation or custom fnction
        return f"This is a placeholder response. The RAG system will provide actual answers for:\nQuestion: {question}\nBrand: {brand}\nCountry: {country}\nTime Period: {time_period}"

# Predefined questions for each brand
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

st.subheader("Predefined Questions")
col1, col2 = st.columns(2)
cols = [col1, col2]

for idx, question in enumerate(predefined_questions[brand]):
    with cols[idx // 2]:
        if st.button(question, key=f"q_{idx}", use_container_width=True,
                    help="Click to get answer"):
            handle_question_click(question)

# Question input and answer section
st.markdown("---")
question_section = st.container()

with question_section:
    st.subheader("Question & Answer")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize the question input session state if it doesn't exist
        if 'question_input' not in st.session_state:
            st.session_state.question_input = ""
            
        question_input = st.text_input(
            "Enter or select a question:",
            value=st.session_state.question_input,
            key="question_box"
        )
    
    with col2:
        if st.button("Get Answer", key="submit", use_container_width=True):
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
# st.markdown("---")
# with st.expander("Advanced Options"):
#     if st.checkbox("Show Information"):
#         st.info("This is a demo version.")