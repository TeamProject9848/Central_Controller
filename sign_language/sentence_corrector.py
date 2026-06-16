import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download
from spellchecker import SpellChecker

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPELL = SpellChecker()

GRAMMAR_REPO = "prithivida/grammar_error_correcter_v1"
GRAMMAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "grammar_corrector")

CONTEXT_REPO = "google/flan-t5-base"
CONTEXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "context_fixer")

# =============================
# UTILS
# =============================
def download_if_needed(repo, local_dir):
    if not os.path.exists(local_dir):
        print(f"Downloading {repo} ...")
        snapshot_download(
            repo_id=repo,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    else:
        print(f"Using local model: {local_dir}")

# =============================
# STAGE 1 — SPELL CORRECTION
# =============================
def spell_correct(text: str) -> str:
    words = text.split()
    return " ".join([SPELL.correction(w) or w for w in words])

# =============================
# LOAD GRAMMAR MODEL
# =============================
download_if_needed(GRAMMAR_REPO, GRAMMAR_DIR)

grammar_tokenizer = AutoTokenizer.from_pretrained(
    GRAMMAR_DIR, local_files_only=True
)
grammar_model = AutoModelForSeq2SeqLM.from_pretrained(
    GRAMMAR_DIR, local_files_only=True
).to(DEVICE)
grammar_model.eval()

# =============================
# LOAD CONTEXT MODEL
# =============================
download_if_needed(CONTEXT_REPO, CONTEXT_DIR)

context_tokenizer = AutoTokenizer.from_pretrained(
    CONTEXT_DIR, local_files_only=True
)
context_model = AutoModelForSeq2SeqLM.from_pretrained(
    CONTEXT_DIR, local_files_only=True
).to(DEVICE)
context_model.eval()

print("All NLP models loaded on:", DEVICE)

# =============================
# MAIN PIPELINE
# =============================
def correct_sentence(text: str) -> str:
    if not text.strip():
        return text

    # ---------- Stage 1: Spell ----------
    text = spell_correct(text)

    # ---------- Stage 2: Grammar ----------
    grammar_prompt = f"gec: {text}"
    g_inputs = grammar_tokenizer(
        grammar_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(DEVICE)

    with torch.no_grad():
        g_outputs = grammar_model.generate(
            **g_inputs,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

    text = grammar_tokenizer.decode(
        g_outputs[0],
        skip_special_tokens=True
    )

    # ---------- Stage 3: Context / Intent ----------
    context_prompt = (
        "Rewrite the sentence to be natural, concise, "
        "and contextually correct:\n"
        f"{text}"
    )

    c_inputs = context_tokenizer(
        context_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(DEVICE)

    with torch.no_grad():
        c_outputs = context_model.generate(
            **c_inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    final_text = context_tokenizer.decode(
        c_outputs[0],
        skip_special_tokens=True
    )

    return final_text

# =============================
# TEST MODE
# =============================
if __name__ == "__main__":
    print("\nContext-Aware Sentence Corrector Ready")
    print("Type a sentence (q to quit)\n")

    while True:
        raw = input("Raw sentence: ")
        if raw.lower() == "q":
            break

        print("Corrected   :", correct_sentence(raw))
        print("-" * 40)
