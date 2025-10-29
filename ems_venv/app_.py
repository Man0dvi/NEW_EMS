# semantic_search.py

from typing import List, Dict, Any
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import PGVector
from langchain.docstore.document import Document
import os
import logging

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, pgvector_conn_str: str):
        # Setup embeddings model
        self.embeddings = OpenAIEmbeddings()
        # Setup vectorstore connection to PGVector (Postgres + pgvector extension)
        self.vectorstore = PGVector(connection_string=pgvector_conn_str, embedding=self.embeddings)
    
    def add_code_chunks(self, code_chunks: List[Dict[str, Any]]):
        """
        Add a list of code chunk dicts to the vector store.
        Each chunk dict should contain keys:
        - content (text)
        - metadata dict (e.g. file path, line numbers)
        """
        docs = []
        for chunk in code_chunks:
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            docs.append(Document(page_content=content, metadata=metadata))
        self.vectorstore.add_documents(docs)
        logger.info(f"Added {len(docs)} code chunks to vector store.")
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Perform semantic similarity search on vector store for query,
        returning top-k matching Documents.
        """
        results = self.vectorstore.similarity_search(query, k=top_k)
        logger.info(f"Found {len(results)} nearest neighbors for query: {query}")
        return results

    def clear_index(self):
        """
        Clear the vector index (e.g., before reindexing).
        """
        self.vectorstore.delete_collection()
        logger.info("Vector store collection cleared.")
