# code_refactor.py
from llm_interface import query_llm

def refactor_code(code: str) -> str:
    """
    Refactor the given functional-style code into an object-oriented design.
    Add detailed inline comments explaining the changes.
    
    Keywords: **LLM**, **refactoring**, **object-oriented**, **prompt engineering**
    """
    prompt = (
        "Refactor the following functional code into a more object-oriented style. "
        "Add detailed inline comments to explain the transformation:\n\n"
        f"{code}\n\n"
        "Refactored code:"
    )
    return query_llm(prompt, max_length=1024)

def generate_docstrings(code: str) -> str:
    """
    Enhance the provided code by inserting comprehensive docstrings and inline comments.
    
    Keywords: **LLM**, **docstrings**, **code documentation**, **prompt engineering**
    """
    prompt = (
        "Enhance the following code by adding detailed docstrings for each function and class, "
        "and insert inline comments explaining the logic:\n\n"
        f"{code}\n\n"
        "Enhanced code:"
    )
    return query_llm(prompt, max_length=1024)
