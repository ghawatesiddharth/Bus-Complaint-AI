from pathlib import Path
import re
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
RAW_FILE = DATA_DIR / "reviews.csv"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "reviews_clean.csv"


def clean_text(text: str) -> str:
    text = str(text or "").lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Replace email addresses
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", text)
    # Replace email addresses
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\b\d{10}\b", " ", text)

    # Remove common ticket / PNR identifiers
    text = re.sub(
        r"\b(?:PNR|Ticket|TKT|TT)[\s:#-]*[A-Z0-9-]{5,}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Keep letters/numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def rating_to_sentiment(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    # Keep only fields needed for NLP.
    clean_df = df[["rating", "review_text", "date"]].copy()

    clean_df["review_text"] = clean_df["review_text"].fillna("").astype(str)
    clean_df["cleaned_text"] = clean_df["review_text"].map(clean_text)

    clean_df["sentiment"] = clean_df["rating"].astype(int).map(
        rating_to_sentiment
    )

    # Remove empty reviews.
    clean_df = clean_df[clean_df["cleaned_text"].str.len() > 0]

    # Remove exact duplicates after normalization.
    clean_df = clean_df.drop_duplicates(
        subset=["cleaned_text"]
    ).reset_index(drop=True)

    clean_df.to_csv(OUTPUT_FILE, index=False)

    print("Cleaning completed.")
    print(f"Input records : {len(df)}")
    print(f"Output records: {len(clean_df)}")
    print(f"Output file   : {OUTPUT_FILE}")
    print()
    print("Sentiment distribution:")
    print(clean_df["sentiment"].value_counts())
    print()
    print("Columns:")
    print(clean_df.columns.tolist())


if __name__ == "__main__":
    main()