import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from swarm import Swarm, Agent
import concurrent.futures

# Set up environment variables for API keys
os.environ["OPENAI_API_KEY"] = 'your_openai_api_key'
os.environ['COHERE_API_KEY'] = 'your_cohere_api_key'

def setup_embeddings():
    """
    Initialize and return the embedding model.
    """
    return OpenAIEmbeddings(model="text-embedding-3-large")

def load_documents(urls):
    """
    Load documents from given URLs.

    Args:
        urls (list): A list of URLs to load documents from.

    Returns:
        list: A list of loaded documents.
    """
    # Load documents from URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    # Flatten the list of documents
    return [item for sublist in docs for item in sublist]

def split_documents(docs):
    """
    Split documents into chunks for embedding.

    Args:
        docs (list): A list of documents to split.

    Returns:
        list: A list of document chunks.
    """
    # Initialize a text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=0)
    # Split documents into chunks
    return text_splitter.split_documents(docs)

def generate_context(chunk, whole_document):
    """
    Generate context for a document chunk.

    Args:
        chunk (str): The document chunk.
        whole_document (str): The whole document for context.

    Returns:
        str: Contextual information for the chunk.
    """
    return f"<document> {whole_document} </document> <chunk> {chunk} </chunk>"

def process_chunk(i, doc, docs, client, chunk_agent):
    """
    Process a single document chunk to add context.

    Args:
        i (int): Index of the document.
        doc (str): Document chunk.
        docs (list): List of original documents.
        client (Swarm): Client for agent processing.
        chunk_agent (Agent): Agent to process the chunk.

    Returns:
        Document: Processed document chunk with added context.
    """
    context_variables = {
        'whole_document': docs,
        'chunk': doc
    }
    messages = [{
        "role": "user",
        "content": '''Please give a short succinct context 
                      to situate this chunk within the overall document for the purposes of 
                      improving search retrieval of the chunk.'''
    }]
    response = client.run(agent=chunk_agent, messages=messages, context_variables=context_variables)
    doc.page_content = response.messages[-1]["content"] + '\n' + doc.page_content
    return doc

def main():
    """
    Main function to execute the Contextual RAG process.
    """
    # Initialize embedding model
    embedding = setup_embeddings()

    # URLs to load documents from
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]

    # Load and split documents
    docs = load_documents(urls)
    doc_splits = split_documents(docs)

    # Initialize Swarm client and agent
    client = Swarm()
    chunk_agent = Agent(
        name="Chunk Agent",
        instructions=generate_context,
        model='gpt-4o-mini'
    )

    # Process chunks with concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_chunk, i, doc, docs, client, chunk_agent)
                   for i, doc in enumerate(doc_splits)]
        processed_docs = [future.result() for future in concurrent.futures.as_completed(futures)]

    print("Processed document chunks with added context.")

if __name__ == "__main__":
    main()