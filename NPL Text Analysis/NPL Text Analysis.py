import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
# For POS tagging and tokenization (Note: Execution environment may not support downloading necessary NLTK resources)
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag

# Load dataset
def load_data():
    df = pd.read_csv(file_path)
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    df.dropna(subset=['length'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y/%m')
    df['YearMonth'] = df['Date'].dt.to_period('M')
    return df

# EDA: Message Lengths and Messages Over Time
def exploratory_data_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['length'], bins=30, kde=True)
    plt.title('Distribution of Message Lengths')
    plt.xlabel('Length of message')
    plt.ylabel('Frequency')
    plt.show()

    messages_over_time = df.groupby('YearMonth').size()
    plt.figure(figsize=(12, 6))
    messages_over_time.plot(kind='line')
    plt.title('Number of Messages Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# TF-IDF Vectorization
def tfidf_vectorization(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Message'].values.astype('U'))  # 'U' for Unicode
    return tfidf_matrix

# Main function to run the program
def main(file_path):
    df = load_data(file_path)
    exploratory_data_analysis(df)
    tfidf_matrix = tfidf_vectorization(df)
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

# Example file path (adjust based on actual location)
file_path = 'path_to_your_file/clean_nus_sms.csv'
main(file_path)
