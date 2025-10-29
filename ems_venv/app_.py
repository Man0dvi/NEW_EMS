class CodeDiscoveryAgent:
    def __init__(self, llm_api_key: str = None, semantic_search: SemanticSearch = None, websocket_broadcaster=None):
        self.llm = LLMService(api_key=llm_api_key)
        self.semantic_search = semantic_search
        self.websocket_broadcaster = websocket_broadcaster

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Running Code Discovery Agent")

        project_path = state.get("project_path")
        file_contents = state.get("file_contents", {})

        if not project_path or not file_contents:
            state['code_discovery_error'] = "Missing project path or file contents."
            return state

        if self.websocket_broadcaster:
            await self.websocket_broadcaster.broadcast({ "event": "chunking_started", "message": f"Starting to chunk {len(file_contents)} files." })

        all_chunks = []
        for filename, content in file_contents.items():
            chunks = await chunk_code_semantically(filename, content, self.llm)
            enriched_chunks = await enrich_chunks_with_metadata(chunks, self.llm)
            all_chunks.extend(enriched_chunks)

            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({ "event": "file_chunked", "message": f"Chunked and enriched file: {filename}" })

        if self.semantic_search and all_chunks:
            documents = [
                Document(page_content=chunk.get("content", ""), metadata={
                    "file_path": chunk.get("file", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                })
                for chunk in all_chunks
            ]
            self.semantic_search.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} code chunks to vector store.")

            if self.websocket_broadcaster:
                await self.websocket_broadcaster.broadcast({
                    "event": "indexing_completed",
                    "message": f"Indexed {len(documents)} code chunks for semantic search."
                })

        state["semantic_chunks"] = all_chunks
        state["code_discovery_error"] = None

        logger.info("Finished Code Discovery Agent")
        return state
