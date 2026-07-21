import sys
import pickle
from scipy.sparse import hstack, csr_matrix
import pandas as pd

from features import extract_features_batch

MODEL_PATH = "model/aes_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def score_essay(essay: str, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    feat_df = pd.DataFrame(extract_features_batch([essay]))
    tfidf_vec = vectorizer.transform([essay])
    X = hstack([tfidf_vec, csr_matrix(feat_df.values)])

    pred = model.predict(X)[0]
    return round(float(pred), 2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        essay_text = " ".join(sys.argv[1:])
    else:
        essay_text = (
            "Technology has changed the way we live and communicate with each "
            "other. It has made life easier but also brought new challenges "
            "that society must deal with carefully."
        )
        print("(No essay argument diya gaya, sample essay use ho raha hai)\n")

    score = score_essay(essay_text)
    print(f"Essay: {essay_text[:100]}...")
    print(f"Predicted Score: {score}")
