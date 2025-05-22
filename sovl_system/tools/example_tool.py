# Per-tool activation phrases for the auto_tldr tool
ACTIVATION_PHRASES = [
    "summarize this",
    "give me a tldr",
    "short summary",
    "auto tldr"
]

def auto_tldr(text: str, max_sentences: int = 3) -> str:
    """
    AutoTLDR Tool: Generates a short summary (TL;DR) from a block of text.

    Parameters:
        text (str): The text to summarize.
        max_sentences (int): Maximum number of sentences in the summary (default: 3).

    Returns:
        str: A TL;DR summary consisting of the first N sentences, or an error message if not enough content.

    Example:
        >>> auto_tldr('Python is a popular programming language. It is used for AI and web development. It is easy to learn. Many people use it for data science.', 2)
        'TL;DR: Python is a popular programming language. It is used for AI and web development.'
    """
    import re
    # Split text into sentences (simple regex, not perfect for all cases)
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if not sentences or not sentences[0]:
        return "Error: No content to summarize."
    summary = ' '.join(sentences[:max_sentences])
    return f"TL;DR: {summary}"