import os
import pandas as pd
from utils.preprocess import clean_email

def safe_read_csv(path):
    try:
        # Try UTF-8 first, skip bad lines, use Python engine for flexibility
        return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    except Exception as e_utf8:
        print(f"⚠️ UTF-8 failed for {path}: {e_utf8}")
        try:
            # Fallback to latin1 encoding
            return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip")
        except Exception as e_latin:
            print(f"❌ Failed to read {path} with latin1: {e_latin}")
            raise

def load_enron(path):
    df = safe_read_csv(path)
    df["Label"] = "Legitimate"
    print(f"[enron] Loaded {len(df)} rows")
    return df

def load_spamassassin(path):
    df = safe_read_csv(path)
    df["Label"] = "Spam"
    print(f"[spam] Loaded {len(df)} rows")
    return df

def load_phishing(path):
    df = safe_read_csv(path)
    df["Label"] = "Phishing"
    print(f"[phishing] Loaded {len(df)} rows")
    return df

def load_and_preprocess(enron_path, spam_path, phishing_path):
    print("📁 Working directory:", os.getcwd())

    # Load datasets
    df_enron = load_enron(enron_path)
    df_spam = load_spamassassin(spam_path)
    df_phish = load_phishing(phishing_path)

    # Combine them
    df = pd.concat([df_enron, df_spam, df_phish], ignore_index=True)
    print(f"[combined] Total rows: {len(df)}")

    # Detect and clean text column
    for col in ["body", "text", "content", "message"]:
        if col in df.columns:
            df["clean_body"] = df[col].fillna("").astype(str).apply(clean_email)
            print(f"✅ Cleaned text using column: '{col}'")
            break
    else:
        raise KeyError("❌ No valid text column found (expected 'body', 'text', 'content', or 'message').")

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save to absolute path
    save_path = r"C:\Users\KARTHIK D\Desktop\MinI Project\email_spam_project\data\combined_dataset.csv"
    try:
        df.to_csv(save_path, index=False)
        print(f"✅ Saved labeled dataset to: {save_path}")
    except Exception as e:
        print("❌ Failed to save CSV:", e)
        raise

    # Preview first few rows
    print(df.head())

    return df

# Run the pipeline
if __name__ == "__main__":
    load_and_preprocess(
        enron_path="data/enron.csv",
        spam_path="data/spam.csv",
        phishing_path="data/phishing.csv"
    )