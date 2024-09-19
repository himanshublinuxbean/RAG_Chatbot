# PDF Query with Pinecone and OpenAI

## Project Overview

This project provides a web service for querying PDF files by embedding their content in a vector database (Pinecone) and utilizing OpenAI's GPT models to answer user queries based on the content of the PDF. The service includes two main functionalities: 
1. Uploading and processing a PDF to generate embeddings of the text.
2. Querying the processed PDF data using a natural language question, where the service will return an answer based on the PDF's content.

## Approach

The approach taken in this project follows these key steps:
1. **File Upload**: The user uploads a PDF file, which is saved securely on the server.
2. **PDF Processing**: The PDF is loaded, and its content is split into manageable chunks using the `RecursiveCharacterTextSplitter` to maintain context between sections.
3. **Text Embedding**: The text chunks are embedded into high-dimensional vectors using OpenAI's `text-embedding-3-large` model. These embeddings are then stored in a Pinecone index, allowing efficient similarity searches.
4. **Querying**: When the user submits a natural language query, the system performs a similarity search in Pinecone to find the most relevant chunks of the document. These chunks are then fed into an OpenAI GPT model to generate a response based on the context.
5. **Answer Generation**: The final response, along with the context, is returned to the user as a JSON output.

## Tools and Libraries Used

- **Flask**: A lightweight web framework used to create the REST API for file uploads and querying.
- **Pinecone**: A vector database for efficiently storing and retrieving high-dimensional embeddings using similarity searches.
  - **Indexing**: Pinecone stores vectorized representations of the PDF content.
  - **Similarity Search**: Performs k-nearest neighbor search to find the most relevant text chunks based on the user query.
- **OpenAI API**: Provides the language model for embedding text (via `text-embedding-3-large`) and for generating natural language responses to the user queries.
- **Langchain**: A framework to facilitate document loading, text splitting, and integration with large language models.
  - **PyPDFLoader**: Loads and parses the PDF files.
  - **RecursiveCharacterTextSplitter**: Splits large documents into smaller, manageable chunks while preserving context.
- **Werkzeug**: Used for securely handling file uploads and filenames.
- **Flask-CORS**: Enables cross-origin resource sharing to allow requests from different domains.
- **Environment Variables**: OpenAI and Pinecone API keys are managed securely using environment variables.

## Assumptions

- **Predefined Embedding Model**: The project assumes that the OpenAI `text-embedding-3-large` model is appropriate for embedding PDF content and that it will work well for most text-heavy PDFs.
- **Single Index for All PDFs**: The code uses a single Pinecone index (`index_name = "pdf-query"`) for storing the embeddings of all uploaded PDFs. This assumes that all PDFs can be stored and queried from the same index. If handling multiple users or different projects, separate indices may be required.
- **File Format**: The system is designed to handle PDFs. Other document formats (e.g., Word, Excel) are not supported by the current implementation.
- **Similarity Search for Context**: The system retrieves relevant document sections based on vector similarity search. It assumes that relevant sections for answering a query will be close in vector space, though this might not always yield perfect results.

## Limitations

- **Model Limitations**: The accuracy of the responses depends on the quality of the embeddings from the `text-embedding-3-large` model and the response generation capabilities of the GPT model. Sometimes, the model may generate incorrect or incomplete answers.
- **PDF Structure**: PDFs with complex layouts (e.g., tables, images, or non-standard formatting) may not be processed accurately, as the text extraction mechanism may not handle such structures well.
- **Performance**: The current Flask implementation is not optimized for high concurrency or large-scale usage. For production, a more robust server (e.g., `gunicorn`) and additional optimizations, such as batching and better memory management, would be necessary.
- **Caching**: The project uses `InMemoryCache` to cache responses, but this is limited by the available memory on the server. For larger-scale systems, a more persistent caching solution might be required.
- **No Cleanup of Uploaded Files**: The uploaded PDF files are saved to the `uploads` directory but are not automatically deleted after processing. Over time, this can lead to storage issues if many PDFs are uploaded.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Pip package manager
- API keys for Pinecone and OpenAI

