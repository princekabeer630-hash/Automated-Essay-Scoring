import re
import numpy as np
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words as nltk_words

ENGLISH_VOCAB = set(w.lower() for w in nltk_words.words())


def extract_features(essay: str) -> dict:
    essay = str(essay).strip()

    sentences = sent_tokenize(essay) if essay else []
    tokens = word_tokenize(essay) if essay else []
    word_tokens = [t for t in tokens if re.match(r"^[A-Za-z']+$", t)]

    num_sentences = max(len(sentences), 1)
    num_words = max(len(word_tokens), 1)

    avg_word_len = np.mean([len(w) for w in word_tokens]) if word_tokens else 0
    avg_sentence_len = num_words / num_sentences

    unique_words = set(w.lower() for w in word_tokens)
    vocab_richness = len(unique_words) / num_words

    misspelled = [w for w in word_tokens if w.lower() not in ENGLISH_VOCAB]
    spelling_error_ratio = len(misspelled) / num_words

    punctuation_count = len(re.findall(r"[.,;:!?]", essay))

    long_word_ratio = sum(1 for w in word_tokens if len(w) >= 7) / num_words

    return {
        "num_words": len(word_tokens),
        "num_sentences": len(sentences),
        "avg_word_len": avg_word_len,
        "avg_sentence_len": avg_sentence_len,
        "vocab_richness": vocab_richness,
        "spelling_error_ratio": spelling_error_ratio,
        "punctuation_count": punctuation_count,
        "long_word_ratio": long_word_ratio,
    }


def extract_features_batch(essays):
    return [extract_features(e) for e in essays]
