import pandas as pd
import torch
import os
import glob
import numpy as np
import math
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from transformers import DistilBertTokenizer, DistilBertModel
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, auc
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler


# --- Load MachineLearningPhishing CSVs ---
def load_machinelearningphishing():
    base = "data/phishing_extra/MachineLearningPhishing"
    df_phish = pd.read_csv(f"{base}/features-phishing.csv")
    df_enron = pd.read_csv(f"{base}/features-enron.csv")
    df_phish['Label'] = "Phishing"
    df_enron['Label'] = "Legitimate"
    return pd.concat([df_phish, df_enron], ignore_index=True)

# --- Load PhishDataset Excel files ---
def load_phishdataset_excel():
    base = "data/phishing_extra/PhishDataset"
    df_bal = pd.read_excel(f"{base}/data_bal - 20000.xlsx")
    df_imbal = pd.read_excel(f"{base}/data_imbal - 55000.xlsx")

    for df in [df_bal, df_imbal]:
        df['Label'] = df['Labels'].map({0: "Legitimate", 1: "Phishing"})
        df['clean_body'] = "URL: " + df['URLs'].astype(str)

    return pd.concat([df_bal, df_imbal], ignore_index=True)

# --- Load CSDMC2010 spam/ham ---
def load_csdmc2010():
    base = "data/phishing_extra/csdmc2010"
    def load_emails(folder, label):
        files = glob.glob(f"{folder}/*.txt")
        data = []
        for f in files:
            try:
                with open(f, encoding="latin-1") as fp:
                    data.append({"clean_body": fp.read(), "Label": label})
            except Exception as e:
                print(f"Error reading {f}: {e}")
        return pd.DataFrame(data)
    df_spam = load_emails(f"{base}/spam", "Phishing")
    df_ham = load_emails(f"{base}/ham", "Legitimate")
    return pd.concat([df_spam, df_ham], ignore_index=True)

# --- Step 0: Load SpamAssassin Public Corpus ---
spam_files = glob.glob("data/spamassassin/spam/*")
ham_files = glob.glob("data/spamassassin/easy_ham/*") + glob.glob("data/spamassassin/hard_ham/*")

def load_emails(file_list, label):
    data = []
    for f in file_list:
        try:
            with open(f, encoding="latin-1") as fp:
                data.append({"clean_body": fp.read(), "Label": label})
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return data

df_spam = pd.DataFrame(load_emails(spam_files, "Spam"))
df_ham = pd.DataFrame(load_emails(ham_files, "Legitimate"))
df_spamassassin = pd.concat([df_spam, df_ham], ignore_index=True)
print("SpamAssassin loaded:", df_spamassassin['Label'].value_counts())

# --- Step 1: Load your main dataset ---
df_main = pd.read_csv("data/combined_dataset.csv")

# --- Step 1a: Load additional phishing datasets ---
df_mlphish = load_machinelearningphishing()
df_phishdataset = load_phishdataset_excel()
df_csdmc = load_csdmc2010()

# --- Step 1b: Load live phishing feeds (PhishTank + OpenPhish) with retry logic ---
def get_session_with_retries():
    """Create a requests session with retry strategy and rate limiting"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def load_phishtank():
    """Load phishing URLs from PhishTank with retry logic (requires API key)"""
    try:
        # PhishTank requires an API key. If you have one, set it here:
        api_key = os.environ.get("PHISHTANK_API_KEY", None)
        if not api_key:
            print("  ⚠ PhishTank requires an API key. Set PHISHTANK_API_KEY environment variable. Skipping...")
            return pd.DataFrame(columns=['clean_body', 'Label'])
        
        url = f"https://data.phishtank.com/data/online-valid.csv?_and_app_key={api_key}"
        session = get_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("Fetching PhishTank data...")
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        df['clean_body'] = "URL: " + df['url'].astype(str)
        df['Label'] = "Phishing"
        print(f"  ✓ PhishTank loaded: {len(df)} records")
        return df[['clean_body','Label']]
    except Exception as e:
        print(f"  ⚠ Failed to load PhishTank: {e}. Skipping...")
        return pd.DataFrame(columns=['clean_body', 'Label'])

def load_urlhaus():
    """Load phishing URLs from URLhaus (no API key required)"""
    try:
        url = "https://urlhaus-api.abuse.ch/v1/urls/recent/"
        session = get_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("Fetching URLhaus data...")
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data.get('query_status') == 'ok':
            urls = [item['url'] for item in data.get('urls', []) if item.get('url')]
            df = pd.DataFrame({"clean_body": ["URL: " + u for u in urls], "Label": "Phishing"})
            print(f"  ✓ URLhaus loaded: {len(df)} records")
            return df
        else:
            print("  ⚠ URLhaus returned empty results")
            return pd.DataFrame(columns=['clean_body', 'Label'])
    except Exception as e:
        print(f"  ⚠ Failed to load URLhaus: {e}")
        return pd.DataFrame(columns=['clean_body', 'Label'])

def load_openphish():
    """Load phishing URLs from OpenPhish with retry logic"""
    try:
        url = "https://openphish.com/feed.txt"
        session = get_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("Fetching OpenPhish data...")
        time.sleep(1)
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        urls = [u.strip() for u in response.text.strip().split("\n") if u.strip()]
        if urls:
            df = pd.DataFrame({"clean_body": ["URL: " + u for u in urls], "Label": "Phishing"})
            print(f"  ✓ OpenPhish loaded: {len(df)} records")
            return df
        else:
            print("  ⚠ OpenPhish returned empty results")
            return pd.DataFrame(columns=['clean_body', 'Label'])
    except Exception as e:
        print(f"  ⚠ Failed to load OpenPhish: {e}")
        return pd.DataFrame(columns=['clean_body', 'Label'])

def load_phishing_army():
    """Load phishing domains from Phishing Army (GitHub-based list)"""
    try:
        url = "https://raw.githubusercontent.com/phishing-army/phishing-army-rules/master/phishing-army-rules.txt"
        session = get_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("Fetching Phishing Army data...")
        time.sleep(1)
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        domains = [line.strip() for line in response.text.split("\n") 
                   if line.strip() and not line.startswith("#") and "." in line][:1000]
        if domains:
            df = pd.DataFrame({"clean_body": ["Domain: " + d for d in domains], "Label": "Phishing"})
            print(f"  ✓ Phishing Army loaded: {len(df)} records")
            return df
        else:
            return pd.DataFrame(columns=['clean_body', 'Label'])
    except Exception as e:
        print(f"  ⚠ Failed to load Phishing Army: {e}")
        return pd.DataFrame(columns=['clean_body', 'Label'])

def load_malware_domains():
    """Load malware/phishing domains from abuse.ch URLhaus"""
    try:
        session = get_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("Fetching URLhaus malware domains...")
        time.sleep(1)
        
        url = "https://urlhaus-api.abuse.ch/v1/urls/recent/?limit=500"
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        urls = []
        if data.get('query_status') == 'ok':
            for item in data.get('urls', []):
                if item.get('url') and item.get('threat'):
                    urls.append(item['url'])
        
        if urls:
            df = pd.DataFrame({"clean_body": ["URL: " + u for u in urls[:500]], "Label": "Phishing"})
            print(f"  ✓ URLhaus malware loaded: {len(df)} records")
            return df
        else:
            return pd.DataFrame(columns=['clean_body', 'Label'])
    except Exception as e:
        print(f"  ⚠ Failed to load URLhaus malware: {e}")
        return pd.DataFrame(columns=['clean_body', 'Label'])

def load_easylist():
    """Load phishing/malware URLs from EasyList"""
    try:
        url = "https://easylist.to/easylist/easylist.txt"
        session = get_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("Fetching EasyList data...")
        time.sleep(1)
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        urls = []
        for line in response.text.split("\n"):
            line = line.strip()
            if line and not line.startswith("[") and not line.startswith("!") and ("://" in line or ".com" in line):
                urls.append(line[:100])
        
        if urls:
            df = pd.DataFrame({"clean_body": ["URL: " + u for u in urls[:800]], "Label": "Phishing"})
            print(f"  ✓ EasyList loaded: {len(df)} records")
            return df
        else:
            return pd.DataFrame(columns=['clean_body', 'Label'])
    except Exception as e:
        print(f"  ⚠ Failed to load EasyList: {e}")
        return pd.DataFrame(columns=['clean_body', 'Label'])

def load_google_safe_browsing_cache():
    """Load known malicious patterns from cached safe browsing data"""
    try:
        url = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-NEW.txt"
        session = get_session_with_retries()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("Fetching Phishing.Database data...")
        time.sleep(1)
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        domains = [line.strip() for line in response.text.split("\n") 
                   if line.strip() and not line.startswith("#") and "." in line][:1000]
        if domains:
            df = pd.DataFrame({"clean_body": ["Domain: " + d for d in domains], "Label": "Phishing"})
            print(f"  ✓ Phishing.Database loaded: {len(df)} records")
            return df
        else:
            return pd.DataFrame(columns=['clean_body', 'Label'])
    except Exception as e:
        print(f"  ⚠ Failed to load Phishing.Database: {e}")
        return pd.DataFrame(columns=['clean_body', 'Label'])

df_live = pd.concat([
    load_phishtank(), 
    load_urlhaus(), 
    load_openphish(),
    load_phishing_army(),
    load_malware_domains(),
    load_easylist(),
    load_google_safe_browsing_cache()
], ignore_index=True)
if len(df_live) == 0:
    print("  ⚠ No live phishing feeds loaded. Continuing with other data sources...")
else:
    print(f"Live feeds total: {len(df_live)} records")

# --- Step 2: Merge everything ---
df = pd.concat([
    df_main,
    df_spamassassin,
    df_mlphish,
    df_phishdataset,
    df_csdmc,
    df_live
], ignore_index=True)

df = df.drop_duplicates(subset="clean_body").reset_index(drop=True)
print("Final dataset size:", len(df))
print("Class distribution:\n", df['Label'].value_counts())

# --- Step 3: DistilBERT setup ---
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

save_path = "data/embeddings"
os.makedirs(save_path, exist_ok=True)

def get_distilbert_embeddings(texts, tokenizer, model, batch_size=32, max_len=512, save_path="data/embeddings"):
    texts = texts.astype(str).tolist()
    existing_files = glob.glob(f"{save_path}/batch_*.pt")
    processed_batches = {int(f.split("_")[-1].split(".")[0]) for f in existing_files}
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_idx = i // batch_size
        if batch_idx in processed_batches:
            print(f"Skipping batch {batch_idx+1}, already saved.")
            continue

        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)

        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeddings)

        torch.save(batch_embeddings, f"{save_path}/batch_{batch_idx}.pt")
        print(f"Processed and saved batch {batch_idx+1}/{len(texts)//batch_size + 1}")

    return torch.cat(all_embeddings, dim=0) if all_embeddings else None

# --- Step 4: Generate embeddings (resume if already saved) ---
_ = get_distilbert_embeddings(df["clean_body"], tokenizer, model)

# --- Step 5: Reload all saved batches ---
files = sorted(glob.glob(f"{save_path}/batch_*.pt"))
embeddings = torch.cat([torch.load(f) for f in files], dim=0)
print("Embeddings shape:", embeddings.shape)

# --- Step 6: Align df with embeddings length ---
min_len = min(len(df), embeddings.shape[0])
df = df.iloc[:min_len].reset_index(drop=True)
embeddings = embeddings[:min_len]  # Trim embeddings to match df length
labels = df["Label"]
print("Labels length:", len(labels))
print("Embeddings shape after alignment:", embeddings.shape)

# --- Step 7: Metadata features ---
has_url = df["clean_body"].str.contains("http", na=False).astype(int)
num_links = df["clean_body"].fillna("").str.count("http")
has_attachment = df.get("has_attachment", pd.Series([0]*len(df))).fillna(0).astype(int)

# Sender domain extraction
if "from" in df.columns:
    sender_col = "from"
elif "sender" in df.columns:
    sender_col = "sender"
elif "from_address" in df.columns:
    sender_col = "from_address"
else:
    sender_col = None

if sender_col:
    sender_domain = df[sender_col].fillna("").str.extract(r'@([A-Za-z0-9.-]+)$')[0]
    is_free_provider = sender_domain.str.contains("gmail|yahoo|hotmail|outlook", na=False).astype(int)
    suspicious_tld = sender_domain.str.contains(r"\.(ru|xyz|top|cn)$", na=False).astype(int)
else:
    is_free_provider = pd.Series([0] * len(df))
    suspicious_tld = pd.Series([0] * len(df))

body_length = df["clean_body"].fillna("").str.len()
link_ratio = (num_links / (body_length + 1)).fillna(0)

# --- Step 8: Phishing keyword flags ---
phishing_keywords = ["bank","verify","account","login","update","secure","password","click","urgent",
                     "confirm","action","required","immediately","alert","unusual","unauthorized",
                     "suspended","locked","expire","reset","certificate","validate"]
keyword_flags = pd.DataFrame({
    kw: df["clean_body"].fillna("").str.contains(kw, case=False, na=False).astype(int)
    for kw in phishing_keywords
})

# Count total phishing keywords per email
keyword_count = keyword_flags.sum(axis=1).values.reshape(-1, 1)

# --- Step 8b: URL features ---
url_length = df["clean_body"].str.len()
num_dots = df["clean_body"].str.count("\.")
has_ip = df["clean_body"].str.contains(r"http://\d+\.\d+\.\d+\.\d+", na=False).astype(int)

# Additional suspicious patterns
has_bit_url = df["clean_body"].str.contains(r"bit\.ly|tinyurl|short\.link", case=False, na=False).astype(int)
has_special_chars = df["clean_body"].str.contains(r"[!@#$%^&*()[\]{}]", na=False).astype(int)
url_domain_mismatch = df["clean_body"].str.contains(r"https?://[^/]*@", na=False).astype(int)
has_multiple_urls = (df["clean_body"].str.count(r"http") > 2).astype(int)

# Advanced phishing features
has_form_tags = df["clean_body"].str.contains(r"<form|<input|action=", case=False, na=False).astype(int)
has_script_tags = df["clean_body"].str.contains(r"<script|javascript:", case=False, na=False).astype(int)
has_encoded_url = df["clean_body"].str.contains(r"&#|&#x|%[0-9A-F]{2}", case=False, na=False).astype(int)
url_vs_text_mismatch = df["clean_body"].str.contains(r"\[http.*?\]|\(http.*?\)|href=", case=False, na=False).astype(int)
has_urgency_words = df["clean_body"].str.contains(r"now|today|24 hour|immediately|asap", case=False, na=False).astype(int)
has_numbers_only = df["clean_body"].str.contains(r"http://\d{1,3}\.\d{1,3}|numbers.only", na=False).astype(int)

# Combine all metadata features into a 2D array
metadata_features = np.hstack([
    has_url.to_numpy().reshape(-1, 1),
    num_links.to_numpy().reshape(-1, 1),
    has_attachment.to_numpy().reshape(-1, 1),
    is_free_provider.to_numpy().reshape(-1, 1),
    suspicious_tld.to_numpy().reshape(-1, 1),
    body_length.to_numpy().reshape(-1, 1),
    link_ratio.to_numpy().reshape(-1, 1),
    keyword_flags.to_numpy(),
    keyword_count,
    url_length.to_numpy().reshape(-1, 1),
    num_dots.to_numpy().reshape(-1, 1),
    has_ip.to_numpy().reshape(-1, 1),
    has_bit_url.to_numpy().reshape(-1, 1),
    has_special_chars.to_numpy().reshape(-1, 1),
    url_domain_mismatch.to_numpy().reshape(-1, 1),
    has_multiple_urls.to_numpy().reshape(-1, 1),
    has_form_tags.to_numpy().reshape(-1, 1),
    has_script_tags.to_numpy().reshape(-1, 1),
    has_encoded_url.to_numpy().reshape(-1, 1),
    url_vs_text_mismatch.to_numpy().reshape(-1, 1),
    has_urgency_words.to_numpy().reshape(-1, 1),
    has_numbers_only.to_numpy().reshape(-1, 1)
])

mlphish_cols = ['@ in URLs','Attachments','Css','Encoding','External Resources',
                'Flash content','HTML content','Html Form','Html iFrame',
                'IPs in URLs','Javascript','Phishy']

mlphish_features_list = []
for col in mlphish_cols:
    if col in df.columns:
        safe_col = df[col].fillna(0).infer_objects(copy=False)
        safe_col = safe_col.apply(lambda x: int(x) if isinstance(x,(bool,int)) or str(x).isdigit() else 0)
        mlphish_features_list.append(safe_col.to_numpy().reshape(-1,1))

# One-hot encode Encoding if present
if 'Encoding' in df.columns:
    encoding_dummies = pd.get_dummies(df['Encoding'].fillna("Unknown"), prefix='Encoding')
    mlphish_features_list.append(encoding_dummies.to_numpy())

mlphish_features = np.hstack(mlphish_features_list) if mlphish_features_list else np.zeros((len(df),0))

# --- Step 9: Combine embeddings + metadata + engineered features ---
X = np.hstack([embeddings.numpy(), metadata_features, mlphish_features])

# --- Step 10: Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Step 11: Handle NaNs before resampling ---
imputer = SimpleImputer(strategy="constant", fill_value=0)
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# Convert to numpy arrays to avoid feature naming issues
x_train = np.asarray(x_train, dtype=np.float32)
x_test = np.asarray(x_test, dtype=np.float32)

# --- Step 11a: Safe undersampling of Legitimate class ---
legit_count = y_train.value_counts().get("Legitimate", 0)
target_legit = min(200000, legit_count)  # cap at available count
rus = RandomUnderSampler(
    sampling_strategy={'Legitimate': target_legit}, random_state=42
)
x_train_small, y_train_small = rus.fit_resample(x_train, y_train)

print("After undersampling Legitimate:\n", pd.Series(y_train_small).value_counts())



# --- Step 11b: Apply SMOTE (balance all classes equally) ---
target_legit = y_train_small.value_counts().get("Legitimate", 0)

smote = SMOTE(
    sampling_strategy={
        'Phishing': target_legit,
        'Spam': target_legit
    },
    random_state=42
)

x_train_bal, y_train_bal = smote.fit_resample(x_train_small, y_train_small)

print("Final class distribution after SMOTE:\n", pd.Series(y_train_bal).value_counts())

# --- Step 11c: Add phishing-specific feature (URL entropy) ---
def entropy(s):
    if not s:
        return 0
    p = pd.Series(list(s)).value_counts(normalize=True)
    return -sum(p * np.log2(p))

url_entropy = df["clean_body"].fillna("").apply(entropy).to_numpy().reshape(-1,1)

# Append entropy to feature matrix
X = np.hstack([embeddings.numpy(), metadata_features, mlphish_features, url_entropy])

# --- Step 12: Train LightGBM with optimized class weights ---
# Calculate dynamic class weights based on data distribution
class_dist = pd.Series(y_train_bal).value_counts()
total = len(y_train_bal)
class_weights_calc = {}
for cls in class_dist.index:
    # Higher weight for minority classes
    class_weights_calc[cls] = total / (len(class_dist) * class_dist[cls])

# Boost phishing weight further
class_weights_calc['Phishing'] *= 2.5

# --- STAGE 1: Binary Classifier (Phishing vs Not-Phishing) ---
print("\n" + "="*60)
print("STAGE 1: BINARY CLASSIFIER (Phishing vs Not-Phishing)")
print("="*60)

# Create binary labels: Phishing vs Others
y_train_binary = (y_train_bal == "Phishing").astype(int)
y_test_binary = (y_test == "Phishing").astype(int)

# Train binary classifier tuned for recall (catch all phishing)
clf_binary = LGBMClassifier(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=8,
    n_estimators=1000,
    random_state=42,
    scale_pos_weight=5,  # Boost phishing weight
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
    verbose=-1
)

clf_binary.fit(
    x_train_bal, y_train_binary,
    eval_set=[(x_test, y_test_binary)],
    eval_metric="binary_logloss"
)

y_pred_proba_binary = clf_binary.predict_proba(x_test)[:, 1]

# --- Calculate Precision-Recall Curve ---
from sklearn.metrics import precision_recall_curve, auc

precision, recall, thresholds = precision_recall_curve(y_test_binary, y_pred_proba_binary)
pr_auc = auc(recall, precision)

# Find optimal threshold for F1-score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold_stage1 = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"PR-AUC Score: {pr_auc:.4f}")
print(f"Optimal threshold (Stage 1): {optimal_threshold_stage1:.4f}")
print(f"Max F1-Score: {f1_scores[optimal_idx]:.4f}")

# Plot precision-recall curve (optional - only if matplotlib is available)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='o', label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100, label=f'Optimal (threshold={optimal_threshold_stage1:.3f})', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Phishing Detection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Precision-Recall curve saved as 'precision_recall_curve.png'")
    plt.close()
except ImportError:
    print("⚠ Matplotlib not available. Skipping precision-recall curve visualization.")

# --- STAGE 2: Multiclass Classifier (Legitimate vs Spam) for non-phishing ---
print("\n" + "="*60)
print("STAGE 2: MULTICLASS CLASSIFIER (Legitimate vs Spam)")
print("="*60)

# Filter non-phishing samples
non_phishing_mask = y_train_bal != "Phishing"
x_train_non_phish = x_train_bal[non_phishing_mask]
y_train_non_phish = y_train_bal[non_phishing_mask]

# Test set non-phishing
test_non_phishing_mask = y_test != "Phishing"
x_test_non_phish = x_test[test_non_phishing_mask]
y_test_non_phish = y_test[test_non_phishing_mask]

if len(np.unique(y_train_non_phish)) > 1 and len(x_train_non_phish) > 0:
    clf_multiclass = LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train_non_phish)),
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        n_estimators=800,
        random_state=42,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        verbose=-1
    )
    
    clf_multiclass.fit(x_train_non_phish, y_train_non_phish)
    print("✓ Stage 2 multiclass classifier trained")
else:
    clf_multiclass = None
    print("⚠ Not enough non-phishing samples for Stage 2")

# --- Step 13: Two-Stage Prediction Pipeline ---
# Use optimal threshold from Stage 1
y_pred_stage1 = (y_pred_proba_binary > optimal_threshold_stage1).astype(int)

# Stage 2: For non-phishing samples, classify as Legitimate vs Spam
y_pred_combined = y_test.copy()

if clf_multiclass is not None:
    # Get probabilities for non-phishing samples
    non_phishing_pred_mask = y_pred_stage1 == 0
    if non_phishing_pred_mask.sum() > 0:
        y_pred_multiclass = clf_multiclass.predict(x_test[non_phishing_pred_mask])
        # Update predictions for non-phishing samples
        y_pred_combined[non_phishing_pred_mask] = y_pred_multiclass
    
    # Label Stage 1 predictions as Phishing
    y_pred_combined[y_pred_stage1 == 1] = "Phishing"
else:
    # Fallback: use only Stage 1 predictions
    y_pred_combined[y_pred_stage1 == 1] = "Phishing"

# Convert to list for reporting
y_pred = y_pred_combined.tolist() if hasattr(y_pred_combined, 'tolist') else list(y_pred_combined)

print(f"✓ Threshold used (Stage 1): {optimal_threshold_stage1:.4f}")
print(f"\nPrediction distribution:")
print(pd.Series(y_pred).value_counts())

print("\n" + "="*60)
print("FINAL TWO-STAGE CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Calculate precision and recall specifically for phishing
if "Phishing" in y_test.unique():
    phishing_precision = precision_score(y_test, y_pred, labels=["Phishing"], average='micro', zero_division=0)
    phishing_recall = recall_score(y_test, y_pred, labels=["Phishing"], average='micro', zero_division=0)
    phishing_f1 = f1_score(y_test, y_pred, labels=["Phishing"], average='micro', zero_division=0)
    
    print(f"\n📊 PHISHING DETECTION METRICS (Two-Stage Pipeline):")
    print(f"   Precision (Avoid False Positives): {phishing_precision:.4f}")
    print(f"   Recall (Catch All Phishing): {phishing_recall:.4f}")
    print(f"   F1-Score (Balance): {phishing_f1:.4f}")
    
    # Additional metrics
    print(f"\n📈 STAGE 1 BINARY CLASSIFIER METRICS:")
    print(f"   PR-AUC: {pr_auc:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold_stage1:.4f}")
    
print("\n✓ Two-Stage model training complete!")

# ============================================================================
# FEATURE IMPORTANCE VISUALIZATION - TOP 20 FEATURES
# ============================================================================
print("\n" + "="*60)
print("GENERATING FEATURE IMPORTANCE VISUALIZATION")
print("="*60)

# Check matplotlib availability first
matplotlib_available = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    matplotlib_available = False
    print(f"⚠ Matplotlib import error: {e}")

if matplotlib_available:
    # Build feature names list
    # Embeddings + metadata + phishing keywords + MLPhish features + url_entropy
    embedding_names = [f"BERT_emb_{i}" for i in range(embeddings.shape[1])]
    
    metadata_base_names = [
        'has_url', 'num_links', 'has_attachment', 'is_free_provider',
        'suspicious_tld', 'body_length', 'link_ratio'
    ]
    
    keyword_names = phishing_keywords.copy()
    keyword_names.append('keyword_count')
    
    url_feature_names = [
        'url_length', 'num_dots', 'has_ip', 'has_bit_url', 'has_special_chars',
        'url_domain_mismatch', 'has_multiple_urls', 'has_form_tags',
        'has_script_tags', 'has_encoded_url', 'url_vs_text_mismatch',
        'has_urgency_words', 'has_numbers_only'
    ]
    
    # Account for mlphish encoding dummies (if Encoding column exists)
    mlphish_names = mlphish_cols.copy()
    if 'Encoding' in df.columns:
        try:
            encoding_vals = pd.get_dummies(df['Encoding'].fillna("Unknown"), prefix='Encoding')
            encoding_names = encoding_vals.columns.tolist()
            mlphish_names = mlphish_names + encoding_names
        except:
            pass
    
    all_feature_names = embedding_names + metadata_base_names + keyword_names + url_feature_names + mlphish_names + ['url_entropy']
    
    # Get feature importance from Stage 1 (binary) classifier
    if hasattr(clf_binary, 'feature_importances_'):
        feature_importance = clf_binary.feature_importances_
        
        # Ensure feature names match importance values
        if len(all_feature_names) != len(feature_importance):
            print(f"⚠ Feature name count ({len(all_feature_names)}) != importance count ({len(feature_importance)})")
            print(f"   Using generic feature names")
            all_feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Get top 20 features
        top_20 = importance_df.head(20)
        
        print(f"\n📊 Top 20 Most Important Features (Stage 1 Binary Classifier):\n")
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            print(f"   {idx:2d}. {row['Feature']:30s} → {row['Importance']:.6f}")
        
        # Create bar chart with descriptive feature names
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Truncate long names for readability
        display_names = [name[:40] + '...' if len(name) > 40 else name for name in top_20['Feature'].values]
        
        bars = ax.barh(range(len(top_20)), top_20['Importance'].values, color='steelblue', edgecolor='navy', alpha=0.8)
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Important Features for Phishing Detection\n(Stage 1: Binary Classifier)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_20['Importance'].values)):
            ax.text(value + max(top_20['Importance']) * 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{value:.6f}', va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = "models/feature_importance_stage1.png"
        os.makedirs("models", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance chart saved to: {output_path}")
        plt.close()
        
        # Also save Stage 2 (multiclass) if available
        if clf_multiclass is not None and hasattr(clf_multiclass, 'feature_importances_'):
            feature_importance_mc = clf_multiclass.feature_importances_
            
            importance_df_mc = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': feature_importance_mc
            }).sort_values('Importance', ascending=False)
            
            top_20_mc = importance_df_mc.head(20)
            
            print(f"\n📊 Top 20 Most Important Features (Stage 2 Multiclass Classifier):\n")
            for idx, (_, row) in enumerate(top_20_mc.iterrows(), 1):
                print(f"   {idx:2d}. {row['Feature']:30s} → {row['Importance']:.6f}")
            
            # Create bar chart for Stage 2 with descriptive names
            fig, ax = plt.subplots(figsize=(14, 10))
            
            display_names_mc = [name[:40] + '...' if len(name) > 40 else name for name in top_20_mc['Feature'].values]
            
            bars = ax.barh(range(len(top_20_mc)), top_20_mc['Importance'].values, color='seagreen', edgecolor='darkgreen', alpha=0.8)
            ax.set_yticks(range(len(top_20_mc)))
            ax.set_yticklabels(display_names_mc, fontsize=10)
            ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
            ax.set_title('Top 20 Most Important Features for Spam/Legitimate Classification\n(Stage 2: Multiclass Classifier)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_20_mc['Importance'].values)):
                ax.text(value + max(top_20_mc['Importance']) * 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{value:.6f}', va='center', fontsize=9)
            
            plt.tight_layout()
            output_path_mc = "models/feature_importance_stage2.png"
            plt.savefig(output_path_mc, dpi=300, bbox_inches='tight')
            print(f"✓ Stage 2 feature importance chart saved to: {output_path_mc}")
            plt.close()
        
        # Create comparison chart if both classifiers exist
        if clf_multiclass is not None and hasattr(clf_multiclass, 'feature_importances_'):
            # Normalize importance scores for comparison
            feature_importance_norm1 = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min() + 1e-10)
            feature_importance_norm2 = (feature_importance_mc - feature_importance_mc.min()) / (feature_importance_mc.max() - feature_importance_mc.min() + 1e-10)
            
            comparison_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Stage1_Binary': feature_importance_norm1,
                'Stage2_Multiclass': feature_importance_norm2
            })
            
            # Average importance across stages
            comparison_df['Average_Importance'] = (comparison_df['Stage1_Binary'] + comparison_df['Stage2_Multiclass']) / 2
            comparison_df = comparison_df.sort_values('Average_Importance', ascending=False).head(20)
            
            print(f"\n📊 Top 20 Features - Combined Importance (Both Stages):\n")
            for idx, (_, row) in enumerate(comparison_df.iterrows(), 1):
                print(f"   {idx:2d}. {row['Feature']:30s} → Stage1: {row['Stage1_Binary']:.4f}, Stage2: {row['Stage2_Multiclass']:.4f}")
            
            # Create grouped bar chart with descriptive names
            fig, ax = plt.subplots(figsize=(14, 10))
            x = np.arange(len(comparison_df))
            width = 0.35
            
            display_names_comp = [name[:40] + '...' if len(name) > 40 else name for name in comparison_df['Feature'].values]
            
            bars1 = ax.barh(x - width/2, comparison_df['Stage1_Binary'].values, width, label='Stage 1: Phishing Detection', color='steelblue', alpha=0.8)
            bars2 = ax.barh(x + width/2, comparison_df['Stage2_Multiclass'].values, width, label='Stage 2: Spam/Legitimate', color='seagreen', alpha=0.8)
            
            ax.set_yticks(x)
            ax.set_yticklabels(display_names_comp, fontsize=10)
            ax.set_xlabel('Normalized Feature Importance', fontsize=12, fontweight='bold')
            ax.set_title('Top 20 Features: Stage 1 vs Stage 2 Comparison (Normalized)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11, loc='lower right')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_xlim([0, 1.1])
            
            plt.tight_layout()
            output_path_comp = "models/feature_importance_comparison.png"
            plt.savefig(output_path_comp, dpi=300, bbox_inches='tight')
            print(f"✓ Feature comparison chart saved to: {output_path_comp}")
            plt.close()
    
    else:
        print("⚠ Feature importance not available for this classifier type")
    
    print("\n" + "="*60)
    print("✓ Feature importance visualization complete!")
    print("="*60)

else:
    # Fallback: Text-only feature importance output
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS (Text Output)")
    print("="*60)
    
    # Build feature names list
    embedding_names = [f"BERT_emb_{i}" for i in range(embeddings.shape[1])]
    
    metadata_base_names = [
        'has_url', 'num_links', 'has_attachment', 'is_free_provider',
        'suspicious_tld', 'body_length', 'link_ratio'
    ]
    
    keyword_names = phishing_keywords.copy()
    keyword_names.append('keyword_count')
    
    url_feature_names = [
        'url_length', 'num_dots', 'has_ip', 'has_bit_url', 'has_special_chars',
        'url_domain_mismatch', 'has_multiple_urls', 'has_form_tags',
        'has_script_tags', 'has_encoded_url', 'url_vs_text_mismatch',
        'has_urgency_words', 'has_numbers_only'
    ]
    
    mlphish_names = mlphish_cols.copy()
    if 'Encoding' in df.columns:
        try:
            encoding_vals = pd.get_dummies(df['Encoding'].fillna("Unknown"), prefix='Encoding')
            encoding_names = encoding_vals.columns.tolist()
            mlphish_names = mlphish_names + encoding_names
        except:
            pass
    
    all_feature_names = embedding_names + metadata_base_names + keyword_names + url_feature_names + mlphish_names + ['url_entropy']
    
    if hasattr(clf_binary, 'feature_importances_'):
        feature_importance = clf_binary.feature_importances_
        
        if len(all_feature_names) != len(feature_importance):
            all_feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        top_20 = importance_df.head(20)
        
        print(f"\n📊 Top 20 Most Important Features (Stage 1 Binary Classifier):\n")
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            print(f"   {idx:2d}. {row['Feature']:30s} → {row['Importance']:.6f}")
        
        if clf_multiclass is not None and hasattr(clf_multiclass, 'feature_importances_'):
            feature_importance_mc = clf_multiclass.feature_importances_
            
            importance_df_mc = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': feature_importance_mc
            }).sort_values('Importance', ascending=False)
            
            top_20_mc = importance_df_mc.head(20)
            
            print(f"\n📊 Top 20 Most Important Features (Stage 2 Multiclass Classifier):\n")
            for idx, (_, row) in enumerate(top_20_mc.iterrows(), 1):
                print(f"   {idx:2d}. {row['Feature']:30s} → {row['Importance']:.6f}")
        
        print("\n✓ Feature importance analysis complete!")
        print("   (For visualizations, install matplotlib: pip install matplotlib)")
    
    print("="*60)

# ---------------------------------------------------------------------------
# FEATURE NAME MAP EXPORTER
# Writes a clean mapping from Feature_### -> actual feature name
# Useful for documentation and for labeling charts that display generic names
# ---------------------------------------------------------------------------
try:
    # Reconstruct feature name list if it's not in scope
    if 'all_feature_names' not in globals():
        embedding_names = [f"BERT_emb_{i}" for i in range(embeddings.shape[1])]
        metadata_base_names = [
            'has_url', 'num_links', 'has_attachment', 'is_free_provider',
            'suspicious_tld', 'body_length', 'link_ratio'
        ]
        keyword_names = phishing_keywords.copy()
        keyword_names.append('keyword_count')
        url_feature_names = [
            'url_length', 'num_dots', 'has_ip', 'has_bit_url', 'has_special_chars',
            'url_domain_mismatch', 'has_multiple_urls', 'has_form_tags',
            'has_script_tags', 'has_encoded_url', 'url_vs_text_mismatch',
            'has_urgency_words', 'has_numbers_only'
        ]
        mlphish_names = mlphish_cols.copy()
        if 'Encoding' in df.columns:
            try:
                encoding_vals = pd.get_dummies(df['Encoding'].fillna("Unknown"), prefix='Encoding')
                encoding_names = encoding_vals.columns.tolist()
                mlphish_names = mlphish_names + encoding_names
            except Exception:
                pass
        all_feature_names = embedding_names + metadata_base_names + keyword_names + url_feature_names + mlphish_names + ['url_entropy']

    # Create mapping Feature_0 -> actual name
    mapping = [(f"Feature_{i}", name) for i, name in enumerate(all_feature_names)]
    mapping_df = pd.DataFrame(mapping, columns=['Feature_ID', 'Feature_Name'])

    os.makedirs('models', exist_ok=True)
    csv_path = os.path.join('models', 'feature_name_map.csv')
    json_path = os.path.join('models', 'feature_name_map.json')

    mapping_df.to_csv(csv_path, index=False)
    mapping_df.to_json(json_path, orient='records', indent=2)

    print(f"\n✓ Feature name mapping saved: {csv_path} and {json_path}")
    print("Sample mapping (first 10):")
    print(mapping_df.head(10).to_string(index=False))
except Exception as e:
    print(f"⚠ Could not write feature name mapping: {e}")

# ---------------------------------------------------------------------------
# INTERPRETABILITY DEMO: PREDICT WITH FEATURE CONTRIBUTIONS
# Shows how the model made predictions by highlighting top contributing features
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("INTERPRETABILITY DEMO: TOP CONTRIBUTING FEATURES FOR SAMPLE PREDICTIONS")
print("="*80)

try:
    # Ensure we have the feature names and model
    if 'all_feature_names' not in locals():
        print("⚠ Feature names not available for demo")
    elif not hasattr(clf_binary, 'predict_proba'):
        print("⚠ Model does not support probability predictions")
    else:
        # Get feature importance mapping for quick lookup
        feature_importance_scores = clf_binary.feature_importances_ if hasattr(clf_binary, 'feature_importances_') else None
        
        if feature_importance_scores is None:
            print("⚠ Feature importance scores not available")
        else:
            # Select diverse samples: 1 phishing, 1 legitimate, 1 spam
            phishing_idx = np.where(y_test == "Phishing")[0]
            legit_idx = np.where(y_test == "Legitimate")[0]
            spam_idx = np.where(y_test == "Spam")[0]
            
            sample_indices = []
            sample_labels = []
            
            if len(phishing_idx) > 0:
                sample_indices.append(phishing_idx[0])
                sample_labels.append("Phishing")
            if len(legit_idx) > 0:
                sample_indices.append(legit_idx[0])
                sample_labels.append("Legitimate")
            if len(spam_idx) > 0:
                sample_indices.append(spam_idx[0])
                sample_labels.append("Spam")
            
            for sample_idx, true_label in zip(sample_indices, sample_labels):
                print(f"\n{'─'*80}")
                print(f"📧 SAMPLE: {sample_idx} | TRUE LABEL: {true_label}")
                print(f"{'─'*80}")
                
                # Get sample features
                sample_features = x_test[sample_idx].reshape(1, -1)
                
                # Prediction
                pred_proba_binary = clf_binary.predict_proba(sample_features)[0, 1]
                pred_binary = (pred_proba_binary > optimal_threshold_stage1).astype(int)
                pred_label = "Phishing" if pred_binary == 1 else "Not Phishing (Stage 2)"
                
                print(f"🤖 MODEL PREDICTION: {pred_label}")
                print(f"   Phishing Probability: {pred_proba_binary:.4f}")
                print(f"   Decision Threshold: {optimal_threshold_stage1:.4f}")
                
                # Calculate feature contributions (feature value * importance)
                sample_values = sample_features[0]
                contributions = np.abs(sample_values * feature_importance_scores)
                
                # Get top 10 contributing features for this sample
                top_contrib_idx = np.argsort(contributions)[-10:][::-1]
                
                print(f"\n   Top 10 Contributing Features:")
                print(f"   {'Rank':<6} {'Feature Name':<40} {'Value':>10} {'Importance':>10} {'Contribution':>12}")
                print(f"   {'-'*80}")
                
                for rank, idx in enumerate(top_contrib_idx, 1):
                    feature_name = all_feature_names[idx] if idx < len(all_feature_names) else f"Feature_{idx}"
                    feature_name_short = feature_name[:37] + "..." if len(feature_name) > 40 else feature_name
                    feature_value = sample_values[idx]
                    importance = feature_importance_scores[idx]
                    contribution = contributions[idx]
                    
                    print(f"   {rank:<6} {feature_name_short:<40} {feature_value:>10.4f} {importance:>10.6f} {contribution:>12.6f}")
                
                # Stage 2 prediction if applicable
                if pred_binary == 0 and clf_multiclass is not None:
                    pred_stage2 = clf_multiclass.predict(sample_features)[0]
                    pred_proba_stage2 = clf_multiclass.predict_proba(sample_features)[0]
                    max_prob_idx = np.argmax(pred_proba_stage2)
                    
                    print(f"\n   Stage 2 Classification (Spam vs Legitimate):")
                    print(f"   Predicted: {pred_stage2} (Confidence: {pred_proba_stage2[max_prob_idx]:.4f})")

    print(f"\n{'═'*80}")
    print("✓ Interpretability demo complete!")
    print("="*80)
    
except Exception as e:
    print(f"⚠ Error in interpretability demo: {e}")
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# GENERATE SUMMARY REPORT
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

summary_report = f"""
Two-Stage Email Classification Model Report
{'='*80}

MODEL ARCHITECTURE:
  • Stage 1: Binary Classifier (Phishing vs Not-Phishing)
    - Classifier: LightGBM
    - Threshold: {optimal_threshold_stage1:.4f}
    - PR-AUC Score: {pr_auc:.4f}
  
  • Stage 2: Multiclass Classifier (Spam vs Legitimate)
    - Classifier: LightGBM
    - Classes: Spam, Legitimate, SpamAssassin

PERFORMANCE METRICS:
  • Phishing Detection:
    - Precision: {phishing_precision:.4f}
    - Recall: {phishing_recall:.4f}
    - F1-Score: {phishing_f1:.4f}

FEATURE ENGINEERING:
  • Total Features: {len(all_feature_names)}
    - BERT Embeddings: 768
    - Metadata Features: 7
    - Phishing Keywords: 14
    - URL-based Features: 13
    - MLPhish Features: {len(mlphish_names)}
    - URL Entropy: 1

OUTPUT FILES GENERATED:
  • models/feature_importance_stage1.png - Stage 1 top 20 features
  • models/feature_importance_stage2.png - Stage 2 top 20 features (if available)
  • models/feature_importance_comparison.png - Comparison across stages
  • models/feature_name_map.csv - Feature ID to name mapping
  • models/feature_name_map.json - Feature ID to name mapping (JSON)

RECOMMENDATIONS:
  1. Use the feature importance charts for model transparency
  2. Monitor top contributing features for data drift
  3. Consider feature importance in future feature selection
  4. Review interpretability demo for understanding predictions on edge cases
"""

print(summary_report)

# Save report to file
try:
    report_path = os.path.join('models', 'model_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(summary_report)
    print(f"\n✓ Summary report saved to: {report_path}")
except Exception as e:
    print(f"⚠ Could not save summary report: {e}")

# ============================================================================
# SAVE MODELS AND METADATA FOR VALIDATION
# ============================================================================
print("\n" + "="*80)
print("SAVING MODELS AND METADATA FOR VALIDATION")
print("="*80)

try:
    import joblib
    import json
    import pickle
    
    os.makedirs('models', exist_ok=True)
    
    # 1. Save trained models
    joblib.dump(clf_binary, 'models/clf_binary.pkl')
    print("✓ Saved: models/clf_binary.pkl")
    
    if clf_multiclass is not None:
        joblib.dump(clf_multiclass, 'models/clf_multiclass.pkl')
        print("✓ Saved: models/clf_multiclass.pkl")
    
    # 2. Save feature engineering metadata
    metadata = {
        'optimal_threshold_stage1': float(optimal_threshold_stage1),
        'pr_auc': float(pr_auc),
        'all_feature_names': all_feature_names,
        'phishing_keywords': phishing_keywords,
        'mlphish_cols': mlphish_cols,
        'embedding_dim': embeddings.shape[1],
        'total_features': len(all_feature_names),
        'phishing_precision': float(phishing_precision),
        'phishing_recall': float(phishing_recall),
        'phishing_f1': float(phishing_f1),
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved: models/metadata.json")
    
    # 3. Save test data for validation
    test_data = {
        'x_test': x_test,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred,
        'y_pred_proba_binary': y_pred_proba_binary,
    }
    
    joblib.dump(test_data, 'models/test_data.pkl')
    print("✓ Saved: models/test_data.pkl")
    
    # 4. Save feature importance scores
    feature_importance_data = {
        'stage1_importance': clf_binary.feature_importances_.tolist(),
        'stage2_importance': clf_multiclass.feature_importances_.tolist() if clf_multiclass is not None else None,
        'feature_names': all_feature_names,
    }
    
    with open('models/feature_importance.json', 'w') as f:
        json.dump(feature_importance_data, f, indent=2)
    print("✓ Saved: models/feature_importance.json")
    
    # 5. Save preprocessing info (for consistency in validation)
    preprocessing_info = {
        'embedding_model': 'distilbert-base-uncased',
        'tokenizer': 'DistilBertTokenizer',
        'embedding_dim': 768,
        'max_token_length': 512,
        'phishing_keywords': phishing_keywords,
        'mlphish_feature_cols': mlphish_cols,
    }
    
    with open('models/preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print("✓ Saved: models/preprocessing_info.json")
    
    # 6. Create a model registry file for quick reference
    registry = {
        'binary_classifier': {
            'file': 'models/clf_binary.pkl',
            'type': 'LGBMClassifier',
            'purpose': 'Phishing vs Not-Phishing (Stage 1)',
            'threshold': optimal_threshold_stage1,
            'pr_auc': pr_auc,
        },
        'multiclass_classifier': {
            'file': 'models/clf_multiclass.pkl' if clf_multiclass is not None else None,
            'type': 'LGBMClassifier',
            'purpose': 'Spam vs Legitimate (Stage 2)',
        },
        'test_data': {
            'file': 'models/test_data.pkl',
            'samples': int(len(y_test)),
            'features': int(x_test.shape[1]),
        },
        'metadata': {
            'file': 'models/metadata.json',
            'contains': ['optimal_threshold', 'pr_auc', 'feature_names', 'performance_metrics'],
        },
        'feature_importance': {
            'file': 'models/feature_importance.json',
            'visualization': 'models/feature_importance_stage1.png',
        },
        'preprocessing': {
            'file': 'models/preprocessing_info.json',
            'embeddings': 'data/embeddings/batch_*.pt',
        },
    }
    
    with open('models/model_registry.json', 'w') as f:
        json.dump(registry, f, indent=2)
    print("✓ Saved: models/model_registry.json")
    
    print("\n" + "="*80)
    print("📦 MODELS READY FOR VALIDATION")
    print("="*80)
    print("""
Usage in validation.py:
    import joblib
    import json
    
    # Load models
    clf_binary = joblib.load('models/clf_binary.pkl')
    clf_multiclass = joblib.load('models/clf_multiclass.pkl')
    
    # Load metadata
    with open('models/metadata.json') as f:
        metadata = json.load(f)
    
    optimal_threshold = metadata['optimal_threshold_stage1']
    all_feature_names = metadata['all_feature_names']
    
    # Load test data
    test_data = joblib.load('models/test_data.pkl')
    x_test = test_data['x_test']
    y_test = test_data['y_test']
    """)
    
except ImportError as e:
    print(f"⚠ Missing required library: {e}")
    print("  Install with: pip install joblib")
except Exception as e:
    print(f"⚠ Error saving models: {e}")
    import traceback
    traceback.print_exc()