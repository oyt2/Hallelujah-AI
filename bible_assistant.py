# Install required packages
!pip install -q transformers accelerate
!pip install -q sentence-transformers faiss-cpu
!pip install -q scikit-learn

# Imports
import re
import glob
import torch
import os
import faiss
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from google.colab import files
uploaded = files.upload()

# Load Bible text file
bible_file = sorted(glob.glob("KJV*.txt"))[0]

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

# Final texts and references
verse_texts = [v[0] for v in verses]
verse_refs = [v[1] for v in verses]

# Parse verse into text and reference, not necessary since a formatted Bible is being used
# def parse_verse(line):
#     patterns = [
#         r'(.*?)\s*\(([^)]+)\)$',
#         r'(.*?)\s*—\s*([A-Za-z]+ \d+:\d+)',
#         r'([^¶]+)¶\s*([A-Za-z]+ \d+:\d+)'
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, line)
#         if match:
#             return match.group(1).strip(), match.group(2).strip()
#     return line.strip(), None

# Make sure verses is clean and deduplicated
seen_refs = set()
unique_verses = []
for text, ref in verses:
    if ref not in seen_refs:
        unique_verses.append((text, ref))
        seen_refs.add(ref)
verses = unique_verses

# Use only these texts for embedding
verse_texts = [v[0] for v in verses]

chat_history = []

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

# Load Phi-2 model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
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
    "Respond only with direct teaching, examples from the Bible, or practical advice grounded in scripture. "
    "Avoid made-up scenarios or speculative reasoning. "
    "When answering, always include a direct Bible verse if it supports the message. "
    "Use this format when presenting verses, for example: “Love is patient, love is kind...” (1 Corinthians 13:4). "
    "If multiple verses apply, include up to three. "
    "Do not generate lists, analogies, or hypotheticals. Speak naturally and biblically, like a devotional. "
    "Remember: the people asking questions are not scholars, but everyday believers seeking clarity and encouragement."
)

def semantic_search(query, top_k=10, min_similarity=0.4, min_relevance=2):
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

        # Require a minimum keyword overlap with query
        if relevance_score(text, query) < min_relevance:
            continue

        # Skip duplicates
        if ref in seen_refs:
            continue

        seen_refs.add(ref)
        results.append((score, idx))

    # Sort by semantic similarity score
    results.sort(key=lambda x: x[0], reverse=True)

    return [idx for _, idx in results[:top_k]]

def relevance_score(text, query):
    query_keywords = set(re.findall(r'\w+', query.lower()))
    verse_keywords = set(re.findall(r'\w+', text.lower()))
    return len(query_keywords & verse_keywords)

def search_bible_advanced(query):
    # fallback keyword-based search
    keywords = ["money", "gold", "silver", "riches", "wealth", "poor", "give", "greed", "offering"]
    matches = [i for i, (text, _) in enumerate(verses) if any(k in text.lower() for k in keywords)]
    return matches[:10]

def clean_answer(text):
    # Define strong hallucination triggers (regex-friendly)
    triggers = [
        r'\b(use ?case|logical reasoning|quiz|scenario|choose (the )?(correct|right)|question:?|answer:?|true or false)\b',
        r'##\s*(logical reasoning|questions|use ?cases?)',
        r'\b(imagine you are|let\'s say|suppose|consider this|assume|picture yourself|maybe|possibly|hypothetically|for instance)\b'
    ]

    cutoff_index = len(text)
    for pattern in triggers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() < cutoff_index:
            cutoff_index = match.start()

    cleaned = text[:cutoff_index].strip()

    # Trim any trailing incomplete sentences
    if cleaned and not re.search(r'[.!?]$', cleaned):
        last_punct = max(cleaned.rfind('.'), cleaned.rfind('!'), cleaned.rfind('?'))
        if last_punct != -1:
            cleaned = cleaned[:last_punct + 1].strip()

    # Fallback to original if cleaning strips too much
    return cleaned if cleaned else text.strip()

def ends_with_continuation(s):
    s = s.rstrip()
    return s.endswith((':', ';', ','))

def get_extended_verse(index, max_verses=3):
    sentences_with_refs = []
    current_index = index
    verses_added = 0

    while verses_added < max_verses and current_index < len(verses):
        text, ref = verses[current_index]
        text = text.strip()

        # Split verse text into sentences by period (keep periods)
        sentences = re.findall(r'[^.]+(?:\.|$)', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentences_with_refs.append(f"{sentence} ({ref})")

            # If sentence ends with a period, consider this verse fully processed
            if sentence.endswith('.'):
                # We only return here if this is the last sentence of the last verse chunk
                # If you want to continue accumulating sentences for current verse,
                # just break here and continue to next verse
                return " ".join(sentences_with_refs)

        # Prepare to check next verse consecutiveness
        if current_index + 1 >= len(verses):
            break

        try:
            current_book, current_chap_verse = ref.rsplit(" ", 1)
            next_book, next_chap_verse = verses[current_index + 1][1].rsplit(" ", 1)
            current_chap, current_verse = map(int, current_chap_verse.split(":"))
            next_chap, next_verse = map(int, next_chap_verse.split(":"))
        except Exception:
            break

        if not (current_book == next_book and current_chap == next_chap and next_verse == current_verse + 1):
            break

        current_index += 1
        verses_added += 1

    return " ".join(sentences_with_refs)

# Example testing code snippet:
# indices_to_test = []
# for i, (_, ref) in enumerate(verses):
#     if ref == "Romans 3:21" or ref == "II Peter 2:5":
#         indices_to_test.append(i)

# print(f"Testing get_extended_verse on indices: {indices_to_test}")

# for idx in indices_to_test:
#     print(f"Index {idx} ({verses[idx][1]}):")
#     extended = get_extended_verse(idx)
#     print(extended)
#     print("---")

# print("Ends with comma:", ends_with_continuation("Hello, "))
# print("Ends with comma:", ends_with_continuation("Hello,"))
# print("Ends with semicolon:", ends_with_continuation("Hello; "))
# print("Ends with colon:", ends_with_continuation("Hello: "))
# print("Ends with period:", ends_with_continuation("Hello."))  # Should be False here

def get_verse_insight(index):
    text, ref = verses[index]
    text = text.strip()

    sentence_match = re.match(r'^(.+?[.!?])(?:\s|$)', text)
    if sentence_match:
        summary = sentence_match.group(1).strip()
    else:
        summary = text.strip()

    # Remove this line if you no longer want the flag printed
    # needs_extension = not summary.endswith(('.', '!', '?'))
    # if needs_extension:
    #     summary += " [extended verse on]"

    insight = f"{ref} teaches: {summary}"
    return insight

def get_verse_insight_extended(index, label):
    extended_text = get_extended_verse(index)
    return f"{label} {extended_text}"

def warm_biblical_teaching(response_text):
    import re

    teaching_match = re.search(r"\*\*Biblical Teaching:\*\*\n(.*?)(?=\n\*\*Relevant Bible Verses:\*\*)", response_text, re.DOTALL)
    verses_match = re.search(r"\*\*Relevant Bible Verses:\*\*\n(.*)", response_text, re.DOTALL)

    if not teaching_match or not verses_match:
        return response_text  # fallback if structure is off

    teaching = teaching_match.group(1).strip()
    verses = verses_match.group(1).strip()

    # Warm rephrasing logic
    # You could swap this for GPT-powered rewriting if needed
    warmer_teaching = (
        "The Old Testament offers a beautiful reminder of God’s faithful love and mercy. "
        "We see how He lovingly rescues those who trust Him. Noah, a man who walked with God, "
        "was saved along with his family. Lot was spared from destruction because his heart longed "
        "for righteousness in a sinful place. Over and over, we witness how God responds with compassion "
        "when His people cry out.\n\n"
        "Salvation wasn’t based on perfection — it was always about trusting in God, turning from sin, "
        "and depending on His grace. These stories remind us that the heart of God has always been to save, "
        "redeem, and restore those who seek Him."
    )

    return f"**Biblical Teaching:**\n{warmer_teaching}\n\n**Relevant Bible Verses:**\n{verses}"

def ask_phi2(user_input, max_new_tokens=300, temperature=0.3):
    # Step 1: Semantic + relevance filtering
    relevant_indices = semantic_search(user_input, top_k=15, min_similarity=0.45)
    min_relevance = 2
    relevant_indices = [
        i for i in relevant_indices
        if relevance_score(verses[i][0], user_input) >= min_relevance
    ]

    if not relevant_indices:
        relevant_indices = search_bible_advanced(user_input)

    # Step 2: Sort by keyword overlap relevance
    relevant_indices = sorted(
        relevant_indices,
        key=lambda i: relevance_score(verses[i][0], user_input),
        reverse=True
    )[:3]  # Limit to top 3 directly here

    # Step 3: Prepare extended verse context
    # verse_block = "\n".join([get_extended_verse(i) for i in relevant_indices])

    # Step 3.5: insight_block instead of verse_block
    labels = ["First Verse:", "Second Verse:", "Third Verse:"]
    insight_lines = []
    verse_count = 0
    current_index = 0

    while verse_count < 3 and current_index < len(relevant_indices):
        idx = relevant_indices[current_index]
        extended_text = get_extended_verse(idx)
        last_sentence = re.findall(r'[^.]+(?:\.|$)', extended_text.strip())[-1].strip()

        insight_lines.append(f"{labels[verse_count]} {extended_text}")

        if last_sentence.endswith('.'):
            verse_count += 1

        current_index += 1

    insight_block = "\n\n".join(insight_lines)

    # DEBUG: print verses used
    print("Selected verses for prompt:")
    for i in relevant_indices:
        print(f"{verses[i][1]}: {get_extended_verse(i)}")

    # Step 4: Build full prompt
    teaching_prompt = (
        f"{system_prompt}\n\n"
        f"### Question:\n{user_input}\n\n"
        # f"### Verses:\n{verse_block}\n\n" # if verse_block is used, uncomment again
        f"### Verse Insights:\n{insight_block}\n\n"
        f"Write a clear, biblically grounded explanation answering the question.\n"
        f"Start with a summary of the biblical teaching, then explain the message behind the verses.\n"
        f"Do not quote verses directly—paraphrase instead.\n"
        f"Strictly avoid fictional scenarios, imaginary characters, use cases, logical reasoning tasks, or quizzes.\n"
        f"Do not include anything labeled 'Use Case', 'Quiz', 'Logical Reasoning', 'Question:', or 'Answer:'.\n"
        f"You MUST base your entire answer ONLY on the teachings in the three verses above. "
        f"Do NOT mention, quote, or imply any verses, chapters, books, or passages other than these three. "
        f"If the question cannot be fully answered by these verses alone, respond with: "
        f"'These verses do not directly answer the question, but they reveal this truth: ...'\n"
        f"Base your entire answer strictly on these three verses. Do not add stories, people, or events that are not directly described in these verses. "
        f"Conclude with exactly these three references: {', '.join(verses[i][1] for i in relevant_indices)}.\n\n"
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

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result[len(teaching_prompt):].strip()

    # Step 6: Clean hallucinations/speculations (if you want to enable)
    # answer = clean_answer(answer)

    # Step 7: Ensure final punctuation
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if sentences and not re.search(r'[.!?]$', sentences[-1]):
        sentences = sentences[:-1]
    answer = ' '.join(sentences)
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = re.sub(r'(\s[.!?]){2,}', r'\1', answer)

    # Step 8: Format reference list (not full verse text)
    references_only = sorted(
        set(verses[i][1] for i in relevant_indices)
    )

    # Step 9: Final formatted return
    formatted = "**Biblical Teaching:**\n" + answer
    formatted += "\n\n**Relevant Bible Verses:**\n" + "\n".join(
        f"- {get_extended_verse(i)}" for i in relevant_indices
    )

    return formatted
