from pathlib import Path
from llama_cpp import Llama
from bertopic.representation import LlamaCPP

def build_llm_representation(
    model_path: str = "models/llm/zephyr-7b-beta.Q4_K_M.gguf",
    n_ctx: int = 2048,
    temperature: float = 0.1,
    max_tokens: int = 32,
):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        stop=["\nQ:"],
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=False,
    )
    return LlamaCPP(llm)
