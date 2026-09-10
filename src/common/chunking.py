"""Paragraph-aware token-based chunking for corpus documents.

Uses tiktoken's cl100k_base purely as an approximate token counter (Gemini
doesn't expose a local tokenizer) -- good enough for sizing chunks
consistently; exact Gemini token counts are logged separately per-call by
src/common/llm.py at generation/embedding time.
"""
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

TARGET_CHUNK_TOKENS = 550
OVERLAP_TOKENS = 80


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def chunk_document(text: str, target_tokens: int = TARGET_CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph boundaries
    where possible so entities/relations aren't split mid-sentence."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks = []
    current_paras: list[str] = []
    current_tokens = 0

    def flush():
        if current_paras:
            chunks.append("\n".join(current_paras))

    for para in paragraphs:
        p_tokens = n_tokens(para)
        if current_tokens + p_tokens > target_tokens and current_paras:
            flush()
            # start next chunk with overlap: keep trailing paragraphs
            # totalling roughly overlap_tokens from the chunk just flushed
            overlap_paras = []
            overlap_count = 0
            for p in reversed(current_paras):
                pt = n_tokens(p)
                if overlap_count + pt > overlap_tokens:
                    break
                overlap_paras.insert(0, p)
                overlap_count += pt
            current_paras = overlap_paras
            current_tokens = overlap_count

        current_paras.append(para)
        current_tokens += p_tokens

        # a single paragraph larger than target on its own: hard-split it
        if p_tokens > target_tokens * 1.5:
            flush()
            current_paras, current_tokens = [], 0

    flush()
    return chunks if chunks else [text]


if __name__ == "__main__":
    sample = "Para one is short.\n" * 3 + ("Para long. " * 200) + "\nPara two is also short."
    cs = chunk_document(sample)
    for i, c in enumerate(cs):
        print(f"chunk {i}: {n_tokens(c)} tokens")
