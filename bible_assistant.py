# @title Upload, Install, Import
# Install required packages
!pip install -q transformers accelerate
!pip install -q sentence-transformers faiss-cpu
!pip install -q gradio

# Imports
import re
import glob
import torch
import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from google.colab import files
uploaded = files.upload()

# Load Bible text file
bible_file = sorted(glob.glob("KJV_formatted.txt"))[0]

# @title Load and Parse Bible
# Read raw lines
with open(bible_file, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

# Define verse pattern: [1:1] In the beginning...
verse_pattern = re.compile(r'\[(\d+:\d+)\]\s*(.+)')

# Parse valid verse + reference pairs from lines
verses = []
current_book = None
for line in raw_lines:
    line = line.strip()
    if not line:
        continue

    # Detect book name
    if line.startswith("###"):
        current_book = line[3:].strip()
        continue

    # Match verse lines like [1:1] In the beginning...
    match = verse_pattern.match(line)
    if match and current_book:
        chapter_verse = match.group(1)
        verse_text = match.group(2).strip()
        reference = f"{current_book} {chapter_verse}"
        verses.append((verse_text, reference))

# Use verse texts for embedding
verse_texts = [v[0] for v in verses]

# @title Embedding Model, FAISS
# Load embedding model
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Create embeddings
verse_embeddings = embedder.encode(verse_texts, convert_to_numpy=True, show_progress_bar=True)
verse_embeddings = verse_embeddings.astype("float32")
verse_embeddings = verse_embeddings / np.maximum(np.linalg.norm(verse_embeddings, axis=1, keepdims=True), 1e-8)
np.save("bible_embeddings.npy", verse_embeddings)

# Create FAISS index
embedding_dim = verse_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(verse_embeddings)

# Sanity check: confirm FAISS + verses alignment
assert len(verse_embeddings) == len(verses), "Embedding count and verses mismatch!"

# @title Phi2
# Load Phi-2 model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

system_prompt = (
    "You are a faithful Christian teacher, guided by Scripture and led by the Holy Spirit. "
    "You respond with wisdom, clarity, and compassion, drawing only from the Bible. "
    "Speak gently but boldly, as if encouraging a fellow believer in their walk with Christ. "
    "You always answer with biblical truth, speaking with kindness, clarity, and reverence. "
    "Never generate fictional stories, logic puzzles, or invented characters. "
    "Do not include imaginary professions or hypothetical scenarios. "
    "Avoid repeating the same conclusions in multiple ways. "
    "Keep your answers clear, concise, and focused only on what the Bible says. "
    "Respond only with direct teaching, examples from the Bible, or practical advice grounded in Scripture. "

    "⚠️ Quality Standard: Scripture must never be used decoratively. "
    "Every verse must be integrated into the teaching itself — never quoted just for display or to sound spiritual. "
    "Only include verses that are meaningful, contextually relevant, and clearly explained. "
    "Whenever a verse is mentioned, help the reader understand its context and application. "

    "The tone should remain warm, devotional, and scripturally grounded. "
    "Avoid made-up scenarios or speculative reasoning. "
    "When answering, always include a direct Bible verse if it supports the message. "
    "Use this format when presenting verses, for example: “Love is patient, love is kind...” (1 Corinthians 13:4). "
    "If multiple verses apply, include up to three. "
    "Do not generate lists, analogies, or hypotheticals. Speak naturally and biblically, like a devotional. "
    "Remember: the people asking questions are not scholars, but everyday believers seeking clarity and encouragement."
)

# @title Semantic Search

def semantic_search(query, top_k=10, min_similarity=0.4):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype("float32")
    query_embedding = query_embedding / np.maximum(np.linalg.norm(query_embedding, axis=1, keepdims=True), 1e-8)

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Search in FAISS index (overfetch)
    scores, indices = index.search(query_embedding, top_k * 5)

    seen_refs = set()
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if score < min_similarity:
            continue

        text, ref = verses[idx]

        # Skip short or malformed verses
        if not ref or len(text.split()) <= 5:
            continue

        # Skip duplicates
        if ref in seen_refs:
            continue

        seen_refs.add(ref)
        results.append((score, idx))

    # Sort by semantic similarity score
    results.sort(key=lambda x: x[0], reverse=True)

    return [idx for _, idx in results[:top_k]]

# @title Extended Verse for Incomplete Sentences
def get_extended_verse(index, max_verses = 5):
    text, ref = verses[index]
    bookname, chapter_and_verse = ref.rsplit(" ", 1)
    chapter, verse = chapter_and_verse.rsplit(":", 1)
    chapter, verse = int(chapter), int(verse)
    if text.endswith((".", "!", "?")):
      return text + " (" + ref + ")"

    else:
      combined = text + " (" + ref + ")"
      for i in range(1, max_verses):
        current_index = index + i
        if current_index >= len(verses):
          break
        next_text, next_ref = verses[current_index]
        next_bookname, next_chapter_and_verse = next_ref.rsplit(" ", 1)
        if next_bookname != bookname:
          break
        combined += "\n" + next_text + " (" + next_ref + ")"
        if next_text.endswith((".", "!", "?")):
          return combined
      return combined

# @title Relevant Verses Check
def check_relevant_verses(output_text, verses):
    """
    output_text: str - full generated answer with a **Relevant Bible Verses:** section
    verses: list of (text, ref) tuples - your canonical verses from the Bible

    Returns: dict with keys:
      - 'all_correct': bool
      - 'mismatches': list of tuples (ref, canonical_text, output_text_excerpt)
    """

    # Find the Relevant Bible Verses section
    pattern = r"\*\*Relevant Bible Verses:\*\*(.*)$"
    match = re.search(pattern, output_text, re.DOTALL)
    if not match:
        return {"all_correct": False, "mismatches": [], "error": "No Relevant Bible Verses section found."}

    verses_section = match.group(1).strip()

    # Parse each verse line from output, e.g. "- Verse text (Book Chapter:Verse)"
    output_verses = re.findall(r"-\s*(.+)\s*\(([^)]+)\)", verses_section)

    if not output_verses:
        return {
          "all_correct": False,
          "mismatches": [],
          "error": "No verses found for the Relevant Bible Verses section."
        }

    # Build a lookup for canonical verses by reference for quick checking
    canonical_dict = {ref: text for text, ref in verses}

    mismatches = []

    for out_text, out_ref in output_verses:
        if out_ref not in canonical_dict:
            mismatches.append((out_ref, None, out_text))
            continue

        canonical_text = canonical_dict[out_ref].strip()
        out_text_clean = out_text.strip()

        # Compare texts loosely
        # Check if canonical text is contained inside output text (or vice versa)
        if canonical_text not in out_text_clean and out_text_clean not in canonical_text:
            mismatches.append((out_ref, canonical_text, out_text_clean))

    all_correct = len(mismatches) == 0

    return {"all_correct": all_correct, "mismatches": mismatches}

# @title Main AskPhi2 Function
def ask_phi2(user_input, max_new_tokens=300, temperature=0.2):
    # Step 1: Semantic filtering
    relevant_indices = semantic_search(user_input, top_k=15, min_similarity=0.45)

    if not relevant_indices:
        relevant_indices = semantic_search(user_input, top_k = 15, min_similarity = 0.35)

    # Step 2: Keep top 3 most relevant verses
    relevant_indices = relevant_indices[:3]

    # Step 3: insight_block instead of verse_block
    labels = ["First Verse:", "Second Verse:", "Third Verse:"]
    insight_lines = []

    for position, idx in enumerate(relevant_indices):
      extended_text = get_extended_verse(idx)
      insight_lines.append(f"{labels[position]} {extended_text}")

    insight_block = "\n\n".join(insight_lines)

    # Step 4: Build full prompt
    teaching_prompt = (
        f"{system_prompt}\n\n"
        f"### Question:\n{user_input}\n\n"
        f"### Verse Insights:\n{insight_block}\n\n"
        f"Write a biblically grounded answer to the question, using only the teachings in the verses below.\n"
        f"Only state claims that can be directly supported by the supplied verse text.\n"
        f"Structure your response like this:\n"
        f"1. Begin with a brief summary of the biblical answer.\n"
        f"2. Then, for each verse, explain what it teaches and how it contributes to the answer.\n"
        f"3. Do not include any stories, theology, or interpretations not present in the verses themselves.\n"
        f"Do not quote verses directly—paraphrase instead.\n"
        f"Strictly avoid fictional scenarios, imaginary characters, use cases, logical reasoning tasks, or quizzes.\n"
        f"Do not include anything labeled 'Use Case', 'Quiz', 'Logical Reasoning', 'Question:', or 'Answer:'.\n"
        f"You MUST base your entire answer ONLY on the teachings in the three verses above. "
        f"Do NOT mention, quote, or imply any verses, chapters, books, or passages other than these three. "
        f"If the question cannot be fully answered by these verses alone, respond with: "
        f"'These verses do not directly answer the question, but they reveal this truth: ...'\n"
        f"Base your entire answer strictly on these three verses. Do not add stories, people, or events that are not directly described in these verses. "
        f"Conclude with exactly these three references: {', '.join(verses[i][1] for i in relevant_indices)}.\n\n"
        f"Again: ONLY use the teachings in the verses listed above. Do not introduce any other verse.\n\n"
        f"### Response:"
    )

    # Step 5: Run model
    inputs = tokenizer(teaching_prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Step 6: Ensure final punctuation
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if sentences and not re.search(r'[.!?]$', sentences[-1]):
        sentences = sentences[:-1]
    answer = ' '.join(sentences)
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = re.sub(r'(\s[.!?]){2,}', r'\1', answer)

    # Step 7: Final formatted return
    formatted = "**Biblical Teaching:**\n" + answer
    formatted += "\n\n**Relevant Bible Verses:**\n" + "\n".join(
        f"- {get_extended_verse(i)}" for i in relevant_indices
    )

    # --- Verse correctness check ---
    check_result = check_relevant_verses(formatted, verses)
    if not check_result["all_correct"]:
        print("Verse mismatch detected in Relevant Bible Verses section:")
        for ref, canon, out in check_result["mismatches"]:
            print(f"Reference: {ref}")
            print(f"Canonical: {canon}")
            print(f"Output: {out}")
            print("---")

    return formatted
