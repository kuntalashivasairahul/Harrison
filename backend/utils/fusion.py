# fusion.py
import re

def clean_text(text: str) -> str:
    text = re.sub(r"■ ■.*", "", text)
    text = re.sub(r"\(Reproduced.*?\)", "", text)
    text = re.sub(r"FIGURE\s*\d+-\d+.*", "", text)
    text = re.sub(r"TABLE\s*\d+-\d+.*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fuse_chunks(chunks):
    cleaned = []
    for ch in chunks:
        text = ch.get("text") if isinstance(ch, dict) else None
        if not text:
            continue
        txt = clean_text(text)
        if len(txt.split()) < 5:
            continue
        page = ch.get("page")
        chunk_id = ch.get("chunk_id")
        if page is None or chunk_id is None:
            continue
        cleaned.append(f"- {txt} [p:{page}|c:{chunk_id}]")
    return "\n".join(cleaned)

# ✅ ALIAS TO MATCH main.py
def fuse_context(chunks):
    return fuse_chunks(chunks)
