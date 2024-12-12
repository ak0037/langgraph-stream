import os
import json
import tempfile
from getpass import getpass
from urllib.request import urlretrieve

import nest_asyncio
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from langchain.chains import RetrievalQA, LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate as LangChainCorePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings


import phoenix as px
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace as trace_api
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes
from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    LiteLLMModel,
    QAEvaluator,
    run_evals,
    RelevanceEvaluator,
    ToxicityEvaluator,
    llm_classify,
    llm_generate,
    PromptTemplate
)
from phoenix.session.evaluation import get_retrieved_documents,get_qa_with_reference
from phoenix.trace import DocumentEvaluations, SpanEvaluations

#Defining configs 

project_name = "Phoenix_Capabilities_Testing"  
resource = Resource.create({ResourceAttributes.PROJECT_NAME: project_name})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
endpoint = "http://localhost:6006/v1/traces"
PHOENIX_API_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.ZDiiPZcNka_bxPxfxPeY5SQ0rf5gOCZLOAM7hG-ypjU'
headers = {"authorization": f"Bearer {PHOENIX_API_KEY}"} if PHOENIX_API_KEY else {}
span_exporter = OTLPSpanExporter(
    endpoint=endpoint,
    headers=headers  
)
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor)
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
client = px.Client(
    endpoint="http://localhost:6006", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6NCJ9.ZEb_RGQYLiYufKs0pnkW4ks3iLDV0Xby9GRIeVoWGYE"
)



try:
    groq_api_key = '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'
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["PHOENIX_PROJECT_NAME"] = "Phoenix_Capabilities_Testing"
    os.environ["AZURE_API_KEY"] = "38a6b22e0e4f43828877d844399faf4d"
    os.environ["AZURE_API_BASE"] = "https://ai-abhinavkatiyarai793972137108.openai.azure.com" 
    os.environ["AZURE_API_VERSION"] = "2024-08-01-preview"


except Exception as e:
    print("Error accessing Groq API key from secrets. Please add it to the 'llm-keys' scope with key 'groq-api-key'")
    raise e

model ="azure/gpt-35-turbo"

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://abhin-m2ifqtz5-eastus2.openai.azure.com",
    deployment="text-embedding-ada-002",  # Changed from azure_deployment to deployment
    api_key="4f46e8f30eac4a3abedeb41c04b7ab52",
    api_version="2023-05-15"
)

#Loading chuncking storing data in vector store---------------------------------------------------------
def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  
            file_path = os.path.join(directory_path, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

documents = load_documents('documents')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)
texts = text_splitter.split_documents(documents) 
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    location=":memory:",
    collection_name="my_documents",
)
retriever = qdrant.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 3
    }
)

#------------------------------------------------------------------------------------------------------------

# Defining llms, prompts & chains-------------------------------------------------------------------

question_llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.0,
    max_tokens=512,
    streaming=False
)

qa_llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.1,
    max_tokens=1024,
    streaming=False
)

generate_questions_template = """\
Context information is below.

---------------------
{text}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
3 questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."

Output the questions in JSON format with the keys question_1, question_2, question_3.
"""

qa_prompt_template = """Answer the following question based on the given context. Be concise.

Context: {context}

Question: {question}

Answer:"""

question_chain = LLMChain(
    llm=question_llm,
    prompt=LangChainCorePromptTemplate(
        template=generate_questions_template,
        input_variables=["text"]
    )
)

qa_chain = RetrievalQA.from_chain_type(
    llm=qa_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": LangChainCorePromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )
    }
)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Dry run------------------------------------------------------------------------------------------------
question = "What information is available in the context?"
response = qa_chain({"query": question})

#------------------------------------------------------------------------------------------------------------------------



#Getting all spans
spans_df = client.get_spans_dataframe(project_name=project_name)

# print(spans_df[["name", "span_kind", "attributes.input.value", "attributes.retrieval.documents","attributes.llm.output_messages"]].head(1))

#------------------------------------------------------------------------------------------------------------------------


# Generating Question based on chuncks (Synthetic Data Generation)----------------------------------------------------------------------------------------------------------------------

sampled_chunks = pd.DataFrame({"text": texts})
sample_size = min(2, len(sampled_chunks))
sampled_chunks = sampled_chunks.sample(n=sample_size, random_state=42)

def output_parser(response: str, index: int):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        return {"__error__": str(e)}
    

questions_df = llm_generate(
    dataframe=sampled_chunks,
    template=generate_questions_template,
    model=LiteLLMModel(model=model),
    output_parser=output_parser,
    concurrency=20,
)

questions_with_document_chunk_df = pd.concat([questions_df, sampled_chunks], axis=1)
questions_with_document_chunk_df = questions_with_document_chunk_df.melt(
    id_vars=["text"], value_name="question"
).drop("variable", axis=1)
questions_with_document_chunk_df = questions_with_document_chunk_df[
    questions_with_document_chunk_df["question"].notnull()
]

#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
#Now Whatever questions are generated will use llm answer those based on the conetxt retrievd--------------------------------------------------------------------------------------------------------------

def generate_qa_pairs(questions_df, retriever, qa_chain, max_context_length=1000, docs_per_question=2):

    qa_pairs = []
    total_questions = len(questions_df)
    
    if questions_df.empty:
        print("No questions to process!")
        return pd.DataFrame()
    
    for idx, row in questions_df.iterrows():
        try:
            question = row['question']
            # print(f"\nProcessing question {idx + 1}/{total_questions}:")
            # print(f"Question: {question}")

            relevant_docs = retriever.get_relevant_documents(question)
            context = " ".join([doc.page_content for doc in relevant_docs[:docs_per_question]])
            
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            response = qa_chain({
                "query": question
            })

            qa_pair = {
                "text": row['text'],
                "question": question,
                "answer": response["result"],
                "context": context,
                "context_length": len(context)
            }
            
            # print(f"Answer: {qa_pair['answer'][:100]}...")  
            qa_pairs.append(qa_pair)
            
        except Exception as e:
            print(f"Error processing question {idx + 1}: {str(e)}")
        
            qa_pairs.append({
                "text": row['text'],
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "context": "",
                "context_length": 0
            })
            continue
        
        #progres track
        if (idx + 1) % 5 == 0:
            print(f"\nCompleted {idx + 1}/{total_questions} questions")
    

    qa_df = pd.DataFrame(qa_pairs)

    qa_df['answer_length'] = qa_df['answer'].str.len()
    qa_df['question_length'] = qa_df['question'].str.len()
    
    print(f"\nProcessing complete! Generated {len(qa_df)} QA pairs")
    
    return qa_df

# Generate QA pairs
qa_df = generate_qa_pairs(
    questions_df=questions_with_document_chunk_df,
    retriever=retriever,
    qa_chain=qa_chain,
    max_context_length=1000,
    docs_per_question=2
)

#------------------------------------------------------------------------------------------------------------------------





#RAG Evaluation on retrieved context or docs------------------------------------------------------------------------------------------------------------------
# returns relevant or irrelevant

retrieved_documents_df = get_retrieved_documents(client, project_name=project_name)
nest_asyncio.apply()

relevance_evaluator = RelevanceEvaluator(LiteLLMModel(
    model= model
))

retrieved_documents_relevance_df = run_evals(
    evaluators=[relevance_evaluator],
    dataframe=retrieved_documents_df,
    provide_explanation=True,
    concurrency=5
)[0]

documents_with_relevance_df = pd.concat(
    [retrieved_documents_df, retrieved_documents_relevance_df.add_prefix("eval_")], axis=1
)

print(documents_with_relevance_df.shape)

precision_at_2 = pd.DataFrame(
    {
        "score": documents_with_relevance_df.groupby("context.span_id").apply(
            lambda x: x.eval_score[:2].sum(skipna=False) / 2
        )
    }
)

hit = pd.DataFrame(
    {
        "hit": documents_with_relevance_df.groupby("context.span_id").apply(
            lambda x: x.eval_score[:2].sum(skipna=False) > 0
        )
    }
)

retrievals_df = client.get_spans_dataframe(
    "span_kind == 'RETRIEVER' and input.value is not None", project_name= project_name
)
rag_evaluation_dataframe = pd.concat(
    [
        retrievals_df["attributes.input.value"],
        precision_at_2.add_prefix("precision@2_"),
        hit,
    ],
    axis=1,
)

results = rag_evaluation_dataframe.mean(numeric_only=True)
print(results)

#------------------------------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------------------------------
# Get question answer with refernce data
qa_with_reference_df = get_qa_with_reference(client, project_name=project_name)

# Evaluation on above data 

toxicity_evaluator = ToxicityEvaluator(LiteLLMModel(model=model))
qa_evaluator = QAEvaluator(LiteLLMModel(model=model))
hallucination_evaluator = HallucinationEvaluator(LiteLLMModel(model=model))

qa_correctness_eval_df, hallucination_eval_df,toxicity_eval_df = run_evals(
    evaluators=[qa_evaluator, hallucination_evaluator,toxicity_evaluator],
    dataframe=qa_with_reference_df,
    provide_explanation=True,
    concurrency=20,
)

toxicity_eval_df.mean(numeric_only=True)
qa_correctness_eval_df.mean(numeric_only=True)
hallucination_eval_df.mean(numeric_only=True)

# Custom classification using llm_classify
# Example: Evaluate answer completeness
completeness_template = PromptTemplate(
    """
    Question: {input}
    Answer: {output}
    Reference: {reference}
    
    Evaluate if the answer is complete relative to the available reference information.
    """
)

# completeness_eval = llm_classify(
#     dataframe=qa_with_reference_df,
#     model=LiteLLMModel(model=model),
#     template=completeness_template,
#     rails=["complete", "partially_complete", "incomplete"],
#     provide_explanation=True
# )

CLARITY_SCORE_TEMPLATE = """
You are evaluating the clarity of an AI response on a scale of 1-10.
1 = completely unclear, confusing
10 = crystal clear, perfectly understandable

[BEGIN DATA]
AI Response: {output}
[END DATA]

Please return the clarity score in format: "the score is: X"
Return only this score, no other text.
"""

def clarity_score_parser(output, row_index):
    import re
    pattern = r"score is.*?([0-9]|10)"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        return {"clarity_score": float(match.group(1))}
    return {"clarity_score": None}

clarity_scores = llm_generate(
    dataframe=qa_with_reference_df,
    template=CLARITY_SCORE_TEMPLATE,
    model=LiteLLMModel(model=model),
    output_parser=clarity_score_parser
)

clarity_scores.rename(columns={"clarity_score": "score"}, inplace=True)


#------------------------------------------------------------------------------------------------------------------------

#Logging all evaluation to be seen in UI


client.log_evaluations(
    SpanEvaluations(dataframe=precision_at_2, eval_name="precision@2"),
    DocumentEvaluations(dataframe=retrieved_documents_relevance_df, eval_name="relevance"),
    SpanEvaluations(dataframe=qa_correctness_eval_df, eval_name="Q&A Correctness"),
    SpanEvaluations(dataframe=hallucination_eval_df, eval_name="Hallucination"),
    SpanEvaluations(dataframe=toxicity_eval_df, eval_name="Toxicity"),
    # SpanEvaluations(dataframe=completeness_eval, eval_name="Completeness"),
    SpanEvaluations(dataframe=clarity_scores, eval_name="Clarity"),
)


#------------------------------------------------------------------------------------------------------------------------


#Store Dataframe for further processing----------------------------------------------------------------------------------------------------------------

os.makedirs(f"data/{model}/dataframes", exist_ok=True)
#question  & answer pairs
qa_df.to_csv(f"data/{model}/dataframes/qa_pairs.csv")

# Gives multiple refrence docs retrived per question
retrieved_documents_df.to_csv(f"data/{model}/dataframes/retrieved_documents.csv")

documents_with_relevance_df.to_csv(f"data/dataframes/documents_with_relevance.csv")

# stores input , precision, hit for each input
rag_evaluation_dataframe.to_csv(f"data/{model}/dataframes/rag_evaluation.csv")
precision_at_2.to_csv(f"data/{model}/dataframes/precision_at_2.csv")
qa_correctness_eval_df.to_csv(f"data/{model}/dataframes/qa_correctness.csv")
hallucination_eval_df.to_csv(f"data/{model}/dataframes/hallucination.csv")
toxicity_eval_df.to_csv(f"data/{model}/dataframes/toxicity.csv")
clarity_scores.to_csv(f"data/{model}/dataframes/clarity_scores.csv")