# Bible AI Assistant

A retrieval-augmented AI assistant that answers questions about
Christianity using relevant Bible passages as its source material.

**Core Technologies:** Python · Phi-2 · FAISS · SentenceTransformers · PyTorch


**Problem**

There are many AI systems capable of holding conversations, answering questions, and providing sources when necessary. However, while these systems perform well across many general-purpose topics, they may not consistently answer questions about Christianity using the desired theological source, tone, or format.

**Solution**

The goal of this project is to develop an AI assistant familiar with Christian terminology that answers questions using the Bible as its sole source of information.

**Development Process**
**Initial Approach**

I initially considered using GPT or similar large language models. However, I ultimately chose Microsoft's Phi-2, a smaller 2.7 billion parameter causal language model that could be run locally and gave me greater control over the system's behavior.

Rather than relying entirely on the language model's existing knowledge, I built the system around retrieving relevant passages from Scripture and supplying those passages to Phi-2 as context.

**Reducing Hallucinations**

During early testing, Phi-2 frequently produced irrelevant hypothetical scenarios when answering religious questions. For example, when asked what the Bible says about love, the model could begin with a relevant response but eventually drift into unrelated hypothetical examples.

To reduce this behavior, I experimented with two mechanisms.

The first was a detailed system_prompt containing instructions that control the model's behavior and response style. These instructions require the model to remain focused on Scripture, avoid fictional scenarios and hypothetical examples, provide concise responses, and ground its answers in retrieved Biblical passages.

The second was a clean_answer() function designed to detect phrases commonly associated with unwanted hypothetical or speculative responses and truncate the generated text before the hallucinated section. This function remains in the project as an experimental safeguard, although it is currently disabled in the main response pipeline.

**Bible Parsing and Preprocessing**

The next step was providing the model with a structured Biblical source.

Early versions used Bible files containing additional formatting, commentary, footnotes, and special characters. These complicated parsing and retrieval, so I moved to a simpler format in which each book is identified by a heading and each verse follows a consistent structure:

Genesis

[1:1] In the beginning God created the heavens and the earth.

[1:2] Now the earth was formless and empty,...

The program parses this structure into individual verse-text/reference pairs, such as the verse text alongside Genesis 1:1. Duplicate references are removed before the verses are passed into the retrieval system.

**From Keyword Search to Semantic Search**

An early version of the project relied on predefined keywords such as money, love, sin, hate, and anger. The program searched for these words in the user's question and attempted to find related passages.

This approach was too rigid. Questions outside the predefined vocabulary could produce poor retrieval results, and manually accounting for every possible theological topic was neither scalable nor practical.

The project therefore transitioned toward semantic search, where the meaning of the user's complete question is compared against the meaning of Biblical passages rather than relying exclusively on exact keyword matches.

The current implementation still uses keyword overlap as an additional relevance filter and fallback mechanism, but semantic similarity performs the primary retrieval.

**Phi-2**

Phi-2 is a 2.7 billion parameter causal language model developed by Microsoft Research.

As a causal language model, Phi-2 generates text by predicting subsequent tokens based on the tokens that came before them. Its relatively small size also makes local inference significantly more practical than many much larger language models.

In this project, however, Phi-2 is not responsible for finding the relevant Bible verses.

Its primary responsibility is response generation.

The retrieval system first identifies passages relevant to the user's question. Those passages are then incorporated into a tightly constrained prompt instructing Phi-2 to construct its answer using only the retrieved Biblical context.

This separation between retrieval and generation is the foundation of the project's RAG-style architecture.

**FAISS (Facebook AI Similarity Search)**

Instead of maintaining enormous lists of keywords for every possible question, the project uses FAISS to perform semantic similarity searches across the Bible.

Each Bible verse is converted into a numerical vector known as an embedding. The user's question is converted into the same type of representation.

Semantically related sentences should occupy nearby positions within this vector space. For example, questions such as:

"What does the Bible say about marriage?"

and

"Is it a sin for someone to separate from their spouse?"

may contain substantially different words while still discussing closely related concepts.

The embeddings are normalized, stored in a FAISS IndexFlatIP index, and compared using their vector similarity.

**When the user submits a question, the system:**
1. Generates an embedding for the question.
2. Searches the FAISS index for semantically similar Bible verses.
3. Applies a minimum similarity threshold.
4. Applies an additional keyword-overlap relevance filter.
5. Removes duplicate references.
6. Ranks the remaining passages by similarity.
7. Supplies the strongest results to Phi-2 as context.

This hybrid approach combines semantic similarity with lexical relevance filtering to reduce unrelated retrieval results.

**SentenceTransformer / MPNet Embeddings**

The current version uses:

sentence-transformers/paraphrase-multilingual-mpnet-base-v2

to generate embeddings for both Bible verses and user questions.

The model converts sentences into multidimensional vector representations designed to capture semantic meaning. This allows differently worded sentences discussing similar concepts to be positioned closer together in the embedding space.

Its multilingual capabilities also create an interesting direction for future development: allowing questions in multiple languages while retrieving semantically related Biblical passages.

The resulting embeddings are normalized and passed to FAISS for similarity search.

**Verse Completion — get_extended_verse()**

One issue encountered during retrieval was that an individual Bible verse does not always contain a complete grammatical sentence. Some sentences span multiple consecutive verses, meaning that retrieving only the highest-ranked verse can result in incomplete context.

For example:

For even if there are so-called gods, whether in heaven or on earth (as indeed there are many “gods” and many “lords”),
(I Corinthians 8:5)

yet for us there is but one God, the Father, from whom all things came and for whom we live;...
(I Corinthians 8:6)

Returning only I Corinthians 8:5 leaves the thought incomplete.

To address this, I created get_extended_verse(). The function examines the retrieved verse and, when necessary, continues into consecutive verses to preserve the surrounding sentence and context. It verifies that subsequent passages belong to the same book and chapter and that their verse numbers are consecutive.

The function also limits the amount of additional text that can be retrieved to prevent a single result from becoming excessively long.

For example:

Question:
"Are there many gods?"

Initial retrieval:
I Corinthians 8:5

Extended context:
I Corinthians 8:5–6

This allows the Relevant Bible Verses section to preserve more of a passage's intended context rather than presenting an isolated or incomplete fragment.

**Response Generation**

The final pipeline can be summarized as:

**User Question** → **SentenceTransformer** → **Question Embedding** → **FAISS Semantic Search** → **Similarity + Relevance Filtering** → **Top Relevant Bible Verses** → **Extended Verse Context** → **Phi-2** → **Biblically Grounded Response**


The generated response is then formatted into two sections:

Biblical Teaching:
[Generated explanation]

Relevant Bible Verses:
[Retrieved passages and references]

**Response Tone — warm_biblical_teaching()**

During development, I also experimented with making Phi-2's responses warmer and more conversational.

The model could retrieve appropriate passages while still producing answers that sounded overly mechanical or impersonal. I created warm_biblical_teaching() as an experiment in modifying this tone.

The current function is not part of the main generation pipeline, and its present implementation contains a hard-coded example of warmer Biblical teaching rather than dynamically rewriting arbitrary responses.

For that reason, I currently rely primarily on the system_prompt and generation instructions to control tone.

Future versions could approach this more systematically through improved prompt engineering, conversational history, user feedback, or fine-tuning.
