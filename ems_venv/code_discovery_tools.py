import asyncio
from typing import Optional


# Older helper: Split code into chunks (from your original code)
def chunk_code(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into chunks with given size and overlap.
    Uses simple sliding window.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - overlap  # overlap for context continuity
    return chunks


# New async enriched summarization integrating config/verbosity
async def summarize_technical_details(text: str, verbosity: str = "standard") -> str:
    # Emulate enriching and summarizing code text using verbosity
    chunk_size = 1000 if verbosity == "high" else 500
    chunks = chunk_code(text, chunk_size=chunk_size)
    summaries = []
    for chunk in chunks:
        # Replace below with call to your actual LLM or summarizer
        summary = f"Tech summary [{verbosity}]: {chunk[:100]}..."
        summaries.append(summary)
        await asyncio.sleep(0.001)  # Simulate async call latency
    return "\n\n".join(summaries)


async def summarize_business_logic(text: str, verbosity: str = "standard") -> str:
    # Simulate summary tailored for PM/business person with verbosity choices
    chunk_size = 800 if verbosity == "high" else 400
    chunks = chunk_code(text, chunk_size=chunk_size)
    summaries = []
    for chunk in chunks:
        summary = f"Business summary [{verbosity}]: {chunk[:100]}..."
        summaries.append(summary)
        await asyncio.sleep(0.001)  # Simulate async call latency
    return "\n\n".join(summaries)

# Example enrichment adding metadata (legacy style)
def enrich_code_with_metadata(code: str, filename: str) -> str:
    metadata = f"// File: {filename}\n"
    # Add more metadata extraction/enrichment here if needed
    enriched_code = metadata + code
    return enriched_code
