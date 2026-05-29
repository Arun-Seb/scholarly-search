"""
query.py
--------
Query engine for the PhD RAG Assistant.

This module wraps LlamaIndex's query engine to provide
clean, cited answers from your indexed academic documents.

How a query works:
1. Your question is embedded into a vector (same model as indexing)
2. Cosine similarity finds the top-K most relevant text chunks
3. Those chunks + your question are sent to the LLM (Groq/Llama 3.1)
4. The LLM synthesises an answer grounded strictly in those chunks
5. Source metadata (filename, relevance score) is returned alongside

Response modes explained:
- 'tree_summarize': Best for complex questions spanning multiple docs.
                    Recursively summarises chunks before final answer.
- 'compact':        Fast, fits as many chunks as possible in one LLM call.
- 'simple_summarize': Truncates chunks to fit context. Fast but lossy.
- 'refine':         Iterates over each chunk, refining the answer.
                    Most thorough but slowest.
"""

from llama_index.core import VectorStoreIndex


def get_query_engine(index: VectorStoreIndex, top_k: int = 8, response_mode: str = "tree_summarize"):
    """
    Create a query engine from the index.

    Args:
        index (VectorStoreIndex): Built or loaded index.
        top_k (int): Number of chunks to retrieve per query.
                     Higher = more context, slower response.
                     Recommended: 5-10 for most questions.
        response_mode (str): How the LLM synthesises chunks into an answer.
                             'tree_summarize' recommended for academic text.

    Returns:
        QueryEngine: Ready to call .query() on.

    Example:
        >>> engine = get_query_engine(index, top_k=8)
        >>> response = engine.query("What is the main finding?")
    """
    return index.as_query_engine(
        similarity_top_k=top_k,
        response_mode=response_mode,
    )


def ask(index: VectorStoreIndex, question: str, top_k: int = 8, verbose: bool = True):
    """
    Ask a question and get a cited answer from your documents.

    This is the main function for querying the RAG system.
    Returns both the answer text and the source chunks used.

    Args:
        index (VectorStoreIndex): Built or loaded index.
        question (str): Natural language question about your research.
        top_k (int): Number of source chunks to retrieve. Default 8.
        verbose (bool): Print formatted answer + sources. Default True.

    Returns:
        response: LlamaIndex Response object with:
            - response.response (str): The answer text
            - response.source_nodes (list): Chunks used, with scores

    Example:
        >>> response = ask(index, "What methodology was used?")
        >>> print(response.response)
    """
    engine = get_query_engine(index, top_k=top_k)
    response = engine.query(question)

    if verbose:
        print(f"\n{'='*65}")
        print(f"Q: {question}")
        print(f"{'='*65}")
        print(f"\n💡 Answer:\n{response.response}")
        print(f"\n📚 Sources ({len(response.source_nodes)} chunks retrieved):")
        for i, node in enumerate(response.source_nodes):
            filename = node.metadata.get("filename", "unknown")
            score = node.score if node.score else 0.0
            preview = node.text[:150].replace("\n", " ")
            print(f"\n  [{i+1}] {filename}  (relevance: {score:.3f})")
            print(f"      \"{preview}...\"")

    return response


def ask_batch(index: VectorStoreIndex, questions: list[str], top_k: int = 8) -> list:
    """
    Ask multiple questions at once and print all answers.

    Useful for running a standard set of research questions
    against your documents in one go.

    Args:
        index (VectorStoreIndex): Built or loaded index.
        questions (list[str]): List of questions to ask.
        top_k (int): Chunks to retrieve per question. Default 8.

    Returns:
        list: List of response objects in same order as questions.

    Example:
        >>> questions = [
        ...     "What is the main contribution?",
        ...     "What methodology was used?",
        ...     "What are the key findings?",
        ... ]
        >>> responses = ask_batch(index, questions)
    """
    responses = []
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}]", end="")
        response = ask(index, question, top_k=top_k, verbose=True)
        responses.append(response)
    return responses


def interactive_qa(index: VectorStoreIndex, top_k: int = 8):
    """
    Start an interactive Q&A loop in the notebook.

    Type questions and get answers in real time.
    Type 'quit', 'exit', or 'q' to stop.

    Args:
        index (VectorStoreIndex): Built or loaded index.
        top_k (int): Chunks to retrieve per query. Default 8.

    Example:
        >>> interactive_qa(index)
        🎓 PhD Research Assistant — type 'quit' to stop
        Your question: What are the main findings?
        ...
    """
    print("🎓 PhD Research Assistant — type 'quit' to stop\n")
    print("Tips:")
    print("  - Be specific: 'What EEG features were used?' > 'What features?'")
    print("  - Mention specific papers: 'In the EMBC 2020 paper, what...'")
    print("  - Ask about methodology, findings, limitations, future work\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Q&A session.")
            break

        if question.lower() in ["quit", "exit", "q", ""]:
            print("Session ended.")
            break

        ask(index, question, top_k=top_k, verbose=True)
        print()


# ── Preset question sets for common research review tasks ──────────────────

THESIS_OVERVIEW_QUESTIONS = [
    "What is the main research problem or gap this thesis addresses?",
    "What are the key contributions and novelty of this research?",
    "What methodology and research design was used?",
    "What are the main findings and results?",
    "What are the limitations of this study?",
    "What future research directions are recommended?",
]

LITERATURE_REVIEW_QUESTIONS = [
    "What existing work does this research build upon?",
    "What are the most cited authors or papers in this field?",
    "How does this work compare to existing approaches?",
    "What gap in the literature is identified?",
]

METHODS_QUESTIONS = [
    "What data was collected and how?",
    "How many participants were in the study?",
    "What signal processing or analysis techniques were applied?",
    "How were results validated or evaluated?",
]
