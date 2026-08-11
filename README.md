# AI-project

This project answers questions about Christianity using the Bible as its sole source. It combines the Phi-2 language model for response generation with a FAISS index and the e5-large-v2 embedding model for semantic similarity search. The system converts the user's question into a numerical vector representation, compares it against Bible verse embeddings, and retrieves the three verses with the closest semantic meaning. Phi-2 then uses these verses as context to generate a response grounded in Scripture.
