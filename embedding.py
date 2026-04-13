import json
import numpy as np


def show_progress(epoch, total_epochs):
    percent = (epoch / total_epochs) * 100
    bar_length = 30

    filled_length = int(bar_length * epoch // total_epochs)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    print(f"\rEpoch [{epoch}/{total_epochs}] |{bar}| {percent:.2f}%", end="")
    if epoch == total_epochs:
        print()  # move to next line


# 1. BUILD VOCABULARY

def build_vocab(paragraphs):
    vocab = {}
    idx = 0
    for para in paragraphs:
        for word in para:
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    return vocab



# 2. ONE HOT ENCODING (MANUAL)

def one_hot_vector(word, vocab):
    vec = np.zeros(len(vocab))
    if word in vocab:
        vec[vocab[word]] = 1
    return vec


# 3. SOFTMAX FUNCTION

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()



# 4. INITIALIZE WEIGHTS

def initialize_weights(vocab_size, embedding_dim):
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)
    return W1, W2



# 5. CBOW TRAINING

def train_cbow(paragraphs, window_size, embedding_dim, epochs, lr):
    vocab = build_vocab(paragraphs)
    vocab_size = len(vocab)

    W1, W2 = initialize_weights(vocab_size, embedding_dim)

    for epoch in range(1,epochs+1):
        for sentence in paragraphs:
            for i, target_word in enumerate(sentence):

                context = []

                # Collect context words
                for j in range(i - window_size, i + window_size + 1):
                    if j != i and j >= 0 and j < len(sentence):
                        context.append(sentence[j])

                if not context:
                    continue

                # Input = average of context vectors
                x = np.zeros(vocab_size)
                for word in context:
                    x += one_hot_vector(word, vocab)
                x = x / len(context)

                # Forward pass
                h = np.dot(x, W1)
                u = np.dot(h, W2)
                y_pred = softmax(u)

                # True output
                y_true = one_hot_vector(target_word, vocab)

                # Error
                e = y_pred - y_true

                # Backprop
                dW2 = np.outer(h, e)
                dW1 = np.outer(x, np.dot(W2, e))

                # Update
                W1 -= lr * dW1
                W2 -= lr * dW2
        show_progress(epoch, epochs)
    return W1, vocab



# 6. SKIP-GRAM TRAINING

def train_skipgram(paragraphs, window_size, embedding_dim, epochs, lr):
    vocab = build_vocab(paragraphs)
    vocab_size = len(vocab)

    W1, W2 = initialize_weights(vocab_size, embedding_dim)

    for epoch in range(1,epochs+1):
        for sentence in paragraphs:
            for i, center_word in enumerate(sentence):

                x = one_hot_vector(center_word, vocab)

                for j in range(i - window_size, i + window_size + 1):
                    if j != i and j >= 0 and j < len(sentence):

                        context_word = sentence[j]

                        # Forward
                        h = np.dot(x, W1)
                        u = np.dot(h, W2)
                        y_pred = softmax(u)

                        y_true = one_hot_vector(context_word, vocab)

                        # Error
                        e = y_pred - y_true

                        # Backprop
                        dW2 = np.outer(h, e)
                        dW1 = np.outer(x, np.dot(W2, e))

                        # Update
                        W1 -= lr * dW1
                        W2 -= lr * dW2
        show_progress(epoch, epochs)
    return W1, vocab



# 7. GET WORD VECTOR

def get_word_vector(word, W1, vocab):
    if word in vocab:
        return W1[vocab[word]]
    else:
        return np.zeros(W1.shape[1])



# 8. SENTENCE VECTOR

def get_sentence_vector(sentence, W1, vocab):
    vectors = []

    for word in sentence:
        vec = get_word_vector(word, W1, vocab)
        if np.any(vec):
            vectors.append(vec)

    if not vectors:
        return np.zeros(W1.shape[1])

    return np.mean(vectors, axis=0)



# SAVE MODEL

def save_model(W1, vocab, model_path="model.npz", vocab_path="vocab.json"):
    np.savez(model_path, W1=W1)

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)



# LOAD MODEL

def load_model(model_path="model.npz", vocab_path="vocab.json"):
    data = np.load(model_path)
    W1 = data["W1"]

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    return W1, vocab