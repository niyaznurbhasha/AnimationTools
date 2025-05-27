# readme_generator.py
from llm_interface import query_llm

def generate_readme(repo_name: str, code_summaries: str) -> str:
    """
    Generate a comprehensive README for the repository.
    It should include an introduction, installation instructions, usage examples,
    and a summary of code changes.
    
    Keywords: **LLM**, **RAG**, **documentation generation**, **prompt engineering**
    """
    prompt = (
        f"Generate a detailed README for a GitHub repository named '{repo_name}'. "
        "Include sections for introduction, installation, usage, and a summary of code changes. "
        f"The following are the key changes:\n\n{code_summaries}\n\n"
        "README:"
    )
    return query_llm(prompt, max_length=1024)
