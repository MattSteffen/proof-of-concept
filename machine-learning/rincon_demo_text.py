"""
Text Classification Baseline: TF-IDF + Logistic Regression
===========================================================

This script demonstrates why TF-IDF + Logistic Regression is an excellent baseline for text:
1. TF-IDF captures word importance effectively
2. Logistic regression is fast and interpretable
3. Works well with high-dimensional sparse data
4. Easy to understand feature weights
5. Surprisingly competitive performance

Dataset: IMDB Movie Reviews Sentiment Classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
import os
import tarfile
from urllib.request import urlretrieve
from sklearn.datasets import load_files


warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def load_and_explore_data():
    """Load IMDB dataset and perform initial exploration"""
    print("=" * 60)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("=" * 60)

    # Download and extract IMDB dataset if not present
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_dir = "aclImdb"
    archive_path = "aclImdb_v1.tar.gz"

    # Check if data is extracted
    if not os.path.isdir(os.path.join(dataset_dir, "train")) or not os.path.isdir(
        os.path.join(dataset_dir, "test")
    ):
        # Check if archive exists
        if not os.path.exists(archive_path):
            print(f"Downloading IMDB dataset from {dataset_url}...")
            urlretrieve(dataset_url, archive_path)
            print("Download complete.")

        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall()
        print("Extraction complete.")

    # Load from disk
    print("Loading IMDB Movie Reviews dataset from disk...")

    # Load train and test sets. The 'neg' and 'pos' subfolders are the classes.
    reviews_train = load_files(
        os.path.join(dataset_dir, "train"),
        categories=["neg", "pos"],
        shuffle=True,
        random_state=42,
    )
    reviews_test = load_files(
        os.path.join(dataset_dir, "test"),
        categories=["neg", "pos"],
        shuffle=True,
        random_state=42,
    )

    # Combine train and test data
    # The data is in bytes, so decode to utf-8
    reviews_text = [doc.decode("utf-8") for doc in reviews_train.data + reviews_test.data]
    # Labels are 0 for 'neg' and 1 for 'pos' because of the order in `categories`
    labels = np.concatenate([reviews_train.target, reviews_test.target])

    df = pd.DataFrame({"review": reviews_text, "sentiment": labels})

    # Clean up <br /> tags from reviews
    df["review"] = df["review"].str.replace(r"<br\s*/?>", " ", regex=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nSample reviews:")
    # Print a positive and a negative review
    if 1 in df["sentiment"].values:
        print("Positive review example:")
        print(df[df["sentiment"] == 1]["review"].iloc[0])
    if 0 in df["sentiment"].values:
        print("\nNegative review example:")
        print(df[df["sentiment"] == 0]["review"].iloc[0])

    print("\nTarget distribution:")
    sentiment_counts = df["sentiment"].value_counts()
    print(sentiment_counts)
    print(f"Positive rate: {df['sentiment'].mean():.3f}")

    print(f"\nAverage review length: {df['review'].str.len().mean():.1f} characters")
    print(f"Average words per review: {df['review'].str.split().str.len().mean():.1f}")

    return df


def text_preprocessing_demo(texts):
    """Demonstrate text preprocessing steps"""
    print("\n" + "=" * 60)
    print("STEP 2: TEXT PREPROCESSING EXPLORATION")
    print("=" * 60)

    print("Text preprocessing considerations:")
    print("✓ Lowercase normalization")
    print("✓ Punctuation removal")
    print("✓ Stop word removal")
    print("✓ Stemming/Lemmatization")

    sample_text = texts.iloc[0]
    print(f"\nOriginal text: {sample_text}")

    # Lowercase
    processed = sample_text.lower()
    print(f"Lowercase: {processed}")

    # Remove punctuation
    processed = re.sub(r"[^a-zA-Z\s]", "", processed)
    print(f"Remove punctuation: {processed}")

    # Remove extra spaces
    processed = " ".join(processed.split())
    print(f"Clean spaces: {processed}")

    # Stop words removal
    stop_words = set(stopwords.words("english"))
    words = processed.split()
    words_no_stop = [word for word in words if word not in stop_words]
    processed_no_stop = " ".join(words_no_stop)
    print(f"Remove stop words: {processed_no_stop}")

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words_no_stop]
    processed_stemmed = " ".join(stemmed_words)
    print(f"Stemmed: {processed_stemmed}")

    print(f"\nOriginal length: {len(sample_text)} chars")
    print(f"Processed length: {len(processed_stemmed)} chars")

    return processed_stemmed


def simple_text_cleaner(text):
    """Simple text cleaning function"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def establish_text_baselines(y):
    """Establish naive baselines for text classification"""
    print("\n" + "=" * 60)
    print("STEP 3: NAIVE BASELINES")
    print("=" * 60)

    # Most frequent class baseline
    most_frequent_accuracy = max(np.bincount(y)) / len(y)
    print(f"Most Frequent Class Baseline: {most_frequent_accuracy:.3f}")

    # Random baseline
    random_accuracy = 0.5  # Binary classification
    print(f"Random Baseline: {random_accuracy:.3f}")

    print(f"\nOur goal: Beat {most_frequent_accuracy:.3f} significantly!")

    return most_frequent_accuracy


def train_tfidf_baseline(X_text, y):
    """Train TF-IDF + Logistic Regression baseline"""
    print("\n" + "=" * 60)
    print("STEP 4: TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 60)

    print("Why TF-IDF + Logistic Regression for text baseline?")
    print("✓ TF-IDF captures word importance (frequent in doc, rare in corpus)")
    print("✓ Logistic Regression handles high-dimensional sparse data well")
    print("✓ Fast to train and predict")
    print("✓ Interpretable coefficients")
    print("✓ No complex preprocessing required")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Create pipeline with text cleaning and TF-IDF
    print("\nCreating TF-IDF pipeline...")

    # Vectorizer parameters
    tfidf_params = {
        "max_features": 10000,  # Limit vocabulary size
        "lowercase": True,  # Convert to lowercase
        "stop_words": "english",  # Remove common English stop words
        "ngram_range": (1, 1),  # Only unigrams for baseline
        "min_df": 2,  # Ignore terms appearing in fewer than 2 documents
        "max_df": 0.95,  # Ignore terms appearing in more than 95% of documents
    }

    print("TF-IDF parameters:")
    for param, value in tfidf_params.items():
        print(f"  {param}: {value}")

    # Create pipeline
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )

    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train)

    # Analyze TF-IDF transformation
    tfidf = pipeline.named_steps["tfidf"]
    print(f"\nTF-IDF Vocabulary size: {len(tfidf.vocabulary_)}")

    # Transform to see sparsity
    X_train_tfidf = tfidf.transform(X_train)
    sparsity = 1.0 - (
        X_train_tfidf.nnz / float(X_train_tfidf.shape[0] * X_train_tfidf.shape[1])
    )
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Sparsity: {sparsity:.3f} (good for LogReg)")

    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Training and test accuracy
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)

    print(f"\nTraining accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Overfitting check: {train_accuracy - test_accuracy:.3f} difference")

    # Feature importance (top positive and negative words)
    feature_names = tfidf.get_feature_names_out()
    coef = pipeline.named_steps["classifier"].coef_[0]

    top_positive = np.argsort(coef)[-10:]
    top_negative = np.argsort(coef)[:10]

    print("\nMost positive words (indicating positive sentiment):")
    for idx in reversed(top_positive):
        print(f"  {feature_names[idx]}: {coef[idx]:.3f}")

    print("\nMost negative words (indicating negative sentiment):")
    for idx in top_negative:
        print(f"  {feature_names[idx]}: {coef[idx]:.3f}")

    # Detailed evaluation
    y_pred = pipeline.predict(X_test)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return pipeline, X_train, X_test, y_train, y_test, test_accuracy


def improve_with_ngrams(X_train, X_test, y_train, y_test):
    """Improve baseline by adding n-grams"""
    print("\n" + "=" * 60)
    print("STEP 5: IMPROVING WITH N-GRAMS")
    print("=" * 60)

    print("N-grams can capture phrase-level sentiment:")
    print("✓ 'not good' vs 'good' - bigrams help!")
    print("✓ 'very bad' - captures intensity")
    print("✓ More context for sentiment")

    # Try different n-gram combinations
    ngram_configs = [
        ((1, 1), "Unigrams only"),
        ((1, 2), "Unigrams + Bigrams"),
        ((1, 3), "Unigrams + Bigrams + Trigrams"),
    ]

    results = {}

    for ngram_range, description in ngram_configs:
        print(f"\nTrying {description}: {ngram_range}")

        pipeline_ngram = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        lowercase=True,
                        stop_words="english",
                        ngram_range=ngram_range,
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline_ngram.fit(X_train, y_train)

        test_accuracy = pipeline_ngram.score(X_test, y_test)
        cv_scores = cross_val_score(
            pipeline_ngram, X_train, y_train, cv=3, scoring="accuracy"
        )

        results[description] = test_accuracy
        print(f"  Test accuracy: {test_accuracy:.3f}")
        print(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Show vocabulary size
        vocab_size = len(pipeline_ngram.named_steps["tfidf"].vocabulary_)
        print(f"  Vocabulary size: {vocab_size}")

    best_config = max(results.items(), key=lambda x: x[1])
    print(
        f"\nBest n-gram configuration: {best_config[0]} with {best_config[1]:.3f} accuracy"
    )

    return best_config[1]


def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """Comprehensive hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("STEP 6: HYPERPARAMETER TUNING")
    print("=" * 60)

    print("Tuning key hyperparameters:")
    print("✓ TF-IDF max_features (vocabulary size)")
    print("✓ TF-IDF n-gram range")
    print("✓ Logistic Regression C (regularization)")

    # Parameter grid
    param_grid = {
        "tfidf__max_features": [5000, 10000, 15000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.1, 1.0, 10.0],
    }

    print("\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Create pipeline for grid search
    pipeline_tuned = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True, stop_words="english", min_df=2, max_df=0.95
                ),
            ),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )

    print("\nPerforming Grid Search (this may take a moment)...")
    grid_search = GridSearchCV(
        pipeline_tuned, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

    # Evaluate tuned model
    tuned_train_accuracy = grid_search.score(X_train, y_train)
    tuned_test_accuracy = grid_search.score(X_test, y_test)

    print(f"Tuned model - Training accuracy: {tuned_train_accuracy:.3f}")
    print(f"Tuned model - Test accuracy: {tuned_test_accuracy:.3f}")

    return grid_search.best_estimator_, tuned_test_accuracy


def compare_results(baseline_accuracy, ngram_accuracy, tuned_accuracy, naive_accuracy):
    """Compare all results"""
    print("\n" + "=" * 60)
    print("STEP 7: RESULTS COMPARISON")
    print("=" * 60)

    print("Performance Summary:")
    print(f"Naive Baseline (most frequent): {naive_accuracy:.3f}")
    print(f"TF-IDF + LogReg Baseline:       {baseline_accuracy:.3f}")
    print(f"Best N-gram Configuration:      {ngram_accuracy:.3f}")
    print(f"Fully Tuned Model:              {tuned_accuracy:.3f}")

    improvement_from_naive = baseline_accuracy - naive_accuracy
    improvement_from_ngram = ngram_accuracy - baseline_accuracy
    improvement_from_tuning = tuned_accuracy - ngram_accuracy

    print(f"\nImprovements:")
    print(
        f"Baseline vs Naive:    +{improvement_from_naive:.3f} ({improvement_from_naive/naive_accuracy*100:.1f}%)"
    )
    print(
        f"N-grams vs Baseline:  +{improvement_from_ngram:.3f} ({improvement_from_ngram/baseline_accuracy*100:.1f}%)"
    )
    print(
        f"Tuning vs N-grams:    +{improvement_from_tuning:.3f} ({improvement_from_tuning/ngram_accuracy*100:.1f}%)"
    )

    total_improvement = tuned_accuracy - naive_accuracy
    print(
        f"Total improvement:    +{total_improvement:.3f} ({total_improvement/naive_accuracy*100:.1f}%)"
    )

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("✓ TF-IDF + LogReg provides strong baseline for text")
    print("✓ Simple preprocessing often sufficient")
    print("✓ N-grams can capture important phrase-level patterns")
    print("✓ Feature interpretability helps understand model")
    print("✓ Hyperparameter tuning provides consistent improvements")
    print("✓ Fast training makes experimentation easy")


def demonstrate_predictions(model, sample_texts):
    """Demonstrate model predictions on sample texts"""
    print("\n" + "=" * 60)
    print("STEP 8: PREDICTION DEMONSTRATION")
    print("=" * 60)

    test_reviews = [
        "This movie was absolutely fantastic! Great acting and amazing story.",
        "Terrible film, complete waste of time. Very disappointing.",
        "Not bad, but not great either. Average movie.",
        "Outstanding performance by the lead actor. Highly recommend!",
        "Boring and predictable. Nothing special about this film.",
    ]

    print("Testing model predictions on new reviews:")
    print("-" * 50)

    for i, review in enumerate(test_reviews, 1):
        prediction = model.predict([review])[0]
        probability = model.predict_proba([review])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability)

        print(f"\nReview {i}: {review}")
        print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        print(
            f"Probabilities: Negative={probability[0]:.3f}, Positive={probability[1]:.3f}"
        )


def main():
    """Main execution function"""
    print("TF-IDF + LOGISTIC REGRESSION BASELINE FOR TEXT CLASSIFICATION")
    print("IMDB Movie Reviews: Sentiment Analysis")

    # Step 1: Load and explore data
    df = load_and_explore_data()

    # Step 2: Text preprocessing demo
    sample_processed = text_preprocessing_demo(df["review"])

    # Step 3: Establish naive baselines
    naive_accuracy = establish_text_baselines(df["sentiment"].values)

    # Step 4: Train TF-IDF baseline
    model, X_train, X_test, y_train, y_test, baseline_accuracy = train_tfidf_baseline(
        df["review"], df["sentiment"]
    )

    # Step 5: Improve with n-grams
    ngram_accuracy = improve_with_ngrams(X_train, X_test, y_train, y_test)

    # Step 6: Hyperparameter tuning
    tuned_model, tuned_accuracy = hyperparameter_tuning(
        X_train, X_test, y_train, y_test
    )

    # Step 7: Compare results
    compare_results(baseline_accuracy, ngram_accuracy, tuned_accuracy, naive_accuracy)

    # Step 8: Demonstrate predictions
    demonstrate_predictions(tuned_model, df["review"].head())


if __name__ == "__main__":
    main()
