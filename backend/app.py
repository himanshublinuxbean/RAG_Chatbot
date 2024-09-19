
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pinecone
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from pinecone import ServerlessSpec
from flask_cors import CORS
import traceback
import langchain
from langchain.cache import InMemoryCache

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

os.environ["PINECONE_API_KEY"] = "Enter your pinecone api key here"
os.environ["OPENAI_API_KEY"] = "enter your openai api key here"

# Enable caching
langchain.llm_cache = InMemoryCache()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large")

sample_embedding = embedding_model.embed_documents(["test"])[0]
embedding_dimension = len(sample_embedding)
print(embedding_dimension)

# Initialize Pinecone
pinecone_instance = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment='us-west4-gcp-free')
index_name = "pdf-query"

if index_name not in pinecone_instance.list_indexes().names():
    pinecone_instance.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pinecone_instance.Index(index_name)

def process_pdf(pdf_path: str) -> List[str]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings_and_store(texts: List[str]):
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embedding_model, index_name=index_name)
    return docsearch

def simple_qa(vectorstore, query: str):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    relevant_docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Based on the following context, please answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.predict(prompt)
    return response, context

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    try:
        texts = process_pdf(file_path)
        docsearch = create_embeddings_and_store(texts)
        return jsonify({'message': 'File processed and embeddings created successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query_pdf():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        docsearch = Pinecone.from_existing_index(index_name, embedding_model)
        answer, context = simple_qa(docsearch, query)
        
        return jsonify({
            'answer': answer,
            'context': context
        }), 200
    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"Error processing query: {str(e)}\n{error_trace}")
        return jsonify({'error': f'Error processing query: {str(e)}', 'traceback': error_trace}), 500

if __name__ == "__main__":
    app.run(debug=True)
