import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import hstack, csr_matrix

from features import extract_features_batch

DATA_PATH = "data/essays.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "aes_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' nahi mila. Pehle 'python data/generate_sample_data.py' "
            f"chalayein ya apna essays.csv is path par rakh dein "
            f"(columns: essay, score)."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["essay", "score"])
    return df


def build_features(essays):
    feat_dicts = extract_features_batch(essays)
    feat_df = pd.DataFrame(feat_dicts)
    return feat_df


def main():
    print("[1/5] Loading data...")
    df = load_data()
    print(f"   -> {len(df)} essays loaded.")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["essay"].tolist(), df["score"].values, test_size=0.2, random_state=42
    )

    print("[2/5] Extracting handcrafted features...")
    train_feats = build_features(X_train_text)
    test_feats = build_features(X_test_text)

    print("[3/5] Building TF-IDF text features...")
    vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
    tfidf_train = vectorizer.fit_transform(X_train_text)
    tfidf_test = vectorizer.transform(X_test_text)

    X_train = hstack([tfidf_train, csr_matrix(train_feats.values)])
    X_test = hstack([tfidf_test, csr_matrix(test_feats.values)])

    print("[4/5] Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("[5/5] Evaluating on test set...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n--- Evaluation Results ---")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Vectorizer saved: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
