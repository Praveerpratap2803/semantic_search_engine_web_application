import streamlit as st
import numpy as np
import re
import os
from embedding import save_model, load_model, build_vocab
from preprocessing import preprocess_text
from embedding import train_cbow, train_skipgram, get_sentence_vector
from similarity import compute_similarity

st.title("Semantic Based Document Search Engine ")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# FILE UPLOAD

uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

documents = []
doc_names = []

if uploaded_files:
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        documents.append(text)
        doc_names.append(file.name)

    # st.success("Documents processed successfully!")


# QUERY INPUT

query = st.text_input("Enter your query")

embedding_choice = st.selectbox(
    "Choose Embedding Technique",
    ["One-Hot", "Word2Vec CBOW", "Word2Vec Skip-Gram"]
)


# SEARCH

if st.button("Search"):


    if query and documents:
        if embedding_choice == "Word2Vec CBOW":
            MODEL_PATH = os.path.join(MODEL_DIR, "cbow_model.npz")
            VOCAB_PATH = os.path.join(MODEL_DIR, "cbow_vocab.json")

        elif embedding_choice == "Word2Vec Skip-Gram":
            MODEL_PATH = os.path.join(MODEL_DIR, "skipgram_model.npz")
            VOCAB_PATH = os.path.join(MODEL_DIR, "skipgram_vocab.json")

        processed_query = preprocess_text(query)

        # Flatten query tokens
        query_tokens = []
        for sent in processed_query:
            for word in sent:
                query_tokens.append(word)

        
        # CREATE PARAGRAPHS
        
        paragraphs = []
        paragraph_doc_map = []
        paragraph_ids = []
        raw_paragraphs = []
        seen_paragraphs = set()

        for doc_id, doc_text in enumerate(documents):

            para_list = re.split(r'\n\s*\n', doc_text)
            para_counter = 1

            for para_text in para_list:
                para_text = para_text.strip()

                if para_text and para_text not in seen_paragraphs:

                    seen_paragraphs.add(para_text)

                    raw_paragraphs.append(para_text)

                    processed_para = preprocess_text(para_text)

                    tokens = []
                    for sent in processed_para:
                        tokens.extend(sent)

                    if tokens:
                        paragraphs.append(tokens)
                        paragraph_doc_map.append(doc_id)
                        paragraph_ids.append(para_counter)

                        para_counter += 1
        
        # ONE HOT IMPLEMENTATION
        
        if embedding_choice == "One-Hot":

            vocab = build_vocab(paragraphs)
            # One-hot vectors (Bag-of-Words style)
            def one_hot_vector(tokens):
                vec = np.zeros(len(vocab))
                for word in tokens:
                    if word in vocab:
                        vec[vocab[word]] += 1
                return vec
            # def one_hot_vector(tokens):
            #     vectors = []
            #
            #     for word in tokens:
            #         if word in vocab:
            #             vec = np.zeros(len(vocab))
            #             vec[vocab[word]] = 1
            #             vectors.append(vec)
            #
            #     if not vectors:
            #         return np.zeros(len(vocab))
            #
            #     return np.mean(vectors, axis=0)

            para_vectors = [one_hot_vector(para) for para in paragraphs]
            query_vec = one_hot_vector(query_tokens)

        
        # WORD2VEC CBOW
        
        elif embedding_choice == "Word2Vec CBOW":

            # W1, vocab = train_cbow(paragraphs)

            
            # LOAD OR TRAIN MODEL
            
            if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):

                st.write(f"✅ Loading saved {embedding_choice} model...")
                W1, vocab = load_model(MODEL_PATH, VOCAB_PATH)

            else:
                st.write(f"⚙️ Training CBOW model...")
                W1, vocab = train_cbow(paragraphs,3,50, 50, 0.01)

                save_model(W1, vocab, MODEL_PATH, VOCAB_PATH)
                st.write("💾 Model saved!")

            para_vectors = [
                get_sentence_vector(para, W1, vocab)
                for para in paragraphs
            ]

            query_vec = get_sentence_vector(query_tokens, W1, vocab)

        
        # WORD2VEC SKIP-GRAM
        
        elif embedding_choice == "Word2Vec Skip-Gram":

            # W1, vocab = train_skipgram(paragraphs)

            
            # LOAD OR TRAIN MODEL
            
            if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):

                st.write(f"✅ Loading saved {embedding_choice} model...")
                W1, vocab = load_model(MODEL_PATH, VOCAB_PATH)

            else:
                st.write(f"⚙️ Training Skip-Gram model...")
                W1, vocab = train_skipgram(paragraphs,3,50, 50, 0.01)

                save_model(W1, vocab, MODEL_PATH, VOCAB_PATH)
                st.write("💾 Model saved!")

            para_vectors = [
                get_sentence_vector(para, W1, vocab)
                for para in paragraphs
            ]

            query_vec = get_sentence_vector(query_tokens, W1, vocab)

        
        # SIMILARITY
        
        similarities = compute_similarity(query_vec, para_vectors)

        
        # EDGE CASE 1: All similarities zero
        
        if max(similarities) == 0:
            st.warning("⚠️ No relevant paragraphs found.")
            st.stop()

        
        # RANKING
        
        indexed_results = list(enumerate(similarities))
        indexed_results.sort(key=lambda x: x[1], reverse=True)

        
        # EDGE CASE 2: Threshold filtering
        
        SIMILARITY_THRESHOLD = 0.4

        filtered_results = [
            (idx, score) for idx, score in indexed_results
            if score >= SIMILARITY_THRESHOLD
        ]

        # If nothing passes threshold → fallback to best results
        if not filtered_results:
            st.info("ℹ️ No strong matches found. Showing closest results.")
            top_results = indexed_results[:5]
        else:
            top_results = filtered_results[:5]
        
        # DISPLAY
        
        st.subheader("🔍 Top 5 Relevant Paragraphs")

        for idx, score in top_results:
            doc_id = paragraph_doc_map[idx]
            para_id = paragraph_ids[idx]
            doc_name = doc_names[doc_id]

            st.write(f"📄 Document: {doc_name}")
            st.write(f"📌 Paragraph ID: {para_id}")
            st.write(f"🔹 Paragraph:\n{raw_paragraphs[idx]}")
            st.write(f"⭐ Similarity Score: {score:.4f}")
            st.write("---")