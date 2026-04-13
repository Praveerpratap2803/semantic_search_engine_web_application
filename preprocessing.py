from nltk.stem import WordNetLemmatizer
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()


# 1. TEXT CLEANING

def clean_text(text):
    cleaned = ""
    for ch in text:
        if ch.isalpha() or ch.isspace() or ch.isalnum() or ch in ".!?":
            cleaned += ch
    return cleaned



# 2. SENTENCE SEGMENTATION

def sentence_segmentation(text):
    sentences = []
    current = ""

    for ch in text:
        current += ch
        if ch in ".!?":
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences



# 3. TOKENIZATION

def tokenize(sentence):
    tokens = []
    word = ""

    for ch in sentence:
        if ch.isalpha():
            word += ch
        else:
            if word != "":
                tokens.append(word)
                word = ""

    if word != "":
        tokens.append(word)

    return tokens



# 4. LOWERCASE

def to_lower(tokens):
    return [word.lower() for word in tokens]



# 5. STOPWORD REMOVAL (CUSTOM)

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'did', 'do', 'does',
    'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
    'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
    'just', 'me', 'might', 'more', 'most', 'must', 'my', 'myself', 'no', 'nor',
    'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some',
    'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yours',
    'yourself', 'yourselves'
}

def remove_stopwords(tokens):
    filtered = []
    for word in tokens:
        if word not in STOPWORDS:
            filtered.append(word)
    return filtered


# 6. LEMMATIZATION (CORE CHANGE)

def apply_lemmatization(tokens):
    pos_tags = pos_tag(tokens)

    lemmatized = []
    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos_from_tag(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        lemmatized.append(lemma)

    return lemmatized


def get_wordnet_pos_from_tag(tag):
    tag = tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)



# FULL PIPELINE

def preprocess_text(text):
    text = clean_text(text)

    sentences = sentence_segmentation(text)

    processed_sentences = []

    for sentence in sentences:
        tokens = tokenize(sentence)
        tokens = to_lower(tokens)
        tokens = remove_stopwords(tokens)
        tokens = apply_lemmatization(tokens)

        if tokens:
            processed_sentences.append(tokens)

    return processed_sentences