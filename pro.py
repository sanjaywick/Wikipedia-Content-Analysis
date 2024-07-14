import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.util import ngrams
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from textblob import TextBlob
from PIL import ImageTk, Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import CoherenceModel
import gensim.corpora as corpora


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to scrape content from Wikipedia page
def scrape_wikipedia_content(topic):
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content_paragraphs = soup.find_all('p')
    content_text = ' '.join([paragraph.text for paragraph in content_paragraphs])
    return content_text,url

# Function to generate word cloud from text content
def generate_word_cloud(text,frame):
    if text:
        wordcloud = WordCloud(width=1000, height=270, background_color='white').generate(text)
        plt.figure(figsize=(7, 2))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig('wordcloud.png')
        plt.close()
        img = tk.PhotoImage(file='wordcloud.png')
        wordcloud_label = tk.Label(frame, image=img)
        wordcloud_label.image = img
        wordcloud_label.grid(row=1, column=1)


# Function to calculate perplexity
def calculate_perplexity(lda_model, X):
    return lda_model.perplexity(X)

# Function to perform topic modeling
def perform_topic_modeling(texts, num_topics=5, max_features=None):
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    dt_matrix = lda.transform(X)
    tw_matrix = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    
    # Preprocess the texts
    tokenized_texts = [text.split() for text in texts]
    
    dictionary = corpora.Dictionary(tokenized_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Get the topics in the correct format
    topics = []
    for i in range(num_topics):
        topic_tokens = [feature_names[idx] for idx in np.argsort(-tw_matrix[i])[:10]]  # Get top 10 tokens for each topic
        topics.append(topic_tokens)
    
    coherence_model = CoherenceModel(topics=topics, texts=tokenized_texts, dictionary=dictionary, coherence='u_mass')
    coherence_score = coherence_model.get_coherence()
    
    perplexity = calculate_perplexity(lda, X)
    
    return lda, vectorizer.get_feature_names_out(), coherence_score, perplexity



# Function to extract common n-grams from text content
def content_mining(texts, n=2, top_n=10):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    all_ngrams = []
    for text in texts:
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalnum()]
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        text_ngrams = list(ngrams(words, n))
        all_ngrams.extend(text_ngrams)

    ngram_counts = Counter(all_ngrams)
    common_ngrams = ngram_counts.most_common(top_n)

    return common_ngrams

# Function to generate a summary from text content
def summarize_content(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=2)
    return " ".join([str(sentence) for sentence in summary])

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Function to search for keywords in the text content and get their positions
def search_keywords(text, keywords):
    keyword_results = {}
    for keyword in keywords:
        keyword_count = text.lower().count(keyword.lower())
        positions = [pos for pos, char in enumerate(text.lower()) if text.lower().find(keyword.lower(), pos) == pos]
        keyword_results[keyword] = {"count": keyword_count, "positions": positions}
    return keyword_results


def display_results():
    topic = paper_name_entry.get()
    content_text,url = scrape_wikipedia_content(topic)
    topic_text.insert(tk.END,f"url: {url}\n")
    generate_word_cloud(content_text,wordcloud_frame)
    texts = content_text.split("\n")  # Assuming each document is separated by a newline character
    texts = [doc.strip() for doc in content_text.split('\n') if doc.strip()]
    
    # Example usage
    lda_model, feature_names, coherence_score, perplexity = perform_topic_modeling(texts)

    topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Topic {topic_idx+1}"] = top_words

    for topic, words in topics.items():
        topic_text.insert(tk.END, f"{topic}: {', '.join(words)}\n")

    common_ngrams = content_mining([content_text], n=2, top_n=10)
    ngrams_text.delete('1.0', tk.END)  # Clear previous content
    for ngram in common_ngrams:
        ngrams_text.insert(tk.END, f"{ngram}\n")

    paper_summaries = [summarize_content(content_text)]
    for summary in paper_summaries:
        summary_text.insert(tk.END, f"{summary}\n")

    # Search for keywords and get their count and positions
    keywords = keyword_entry.get().strip().split(',')  # Strip extra spaces before splitting
    keyword_results = search_keywords(content_text, keywords)
    keyword_result_text.delete('1.0', tk.END)
    for keyword, data in keyword_results.items():
        keyword_count = data["count"]
        keyword_positions = data["positions"]
        positions_info = ", ".join([str(pos) for pos in keyword_positions])
        keyword_result_text.insert(tk.END, f"{keyword}: \nCount: {keyword_count}, \nPositions: {positions_info}\n\n")
    senti_score=perform_sentiment_analysis(content_text)


    # Insert "Positive" and "Negative" categories with real precision, recall, and F-score values
    sentiment_table.insert("", "end", values=("Score", coherence_score,perplexity))




# Create GUI
root = tk.Tk()
root.title("Wikipedia Content Analysis")

# Main Frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Widgets
paper_label = tk.Label(main_frame, text="Enter Topic:")
paper_label.grid(row=0, column=0)

paper_name_entry = tk.Entry(main_frame, width=50)
paper_name_entry.grid(row=0, column=1)

keyword_label = tk.Label(main_frame, text="Enter Keywords (comma separated):")
keyword_label.grid(row=1, column=0)

keyword_entry = tk.Entry(main_frame, width=50)
keyword_entry.grid(row=1, column=1)

results_button = tk.Button(main_frame, text="Get Results", command=display_results)
results_button.grid(row=2, columnspan=2)


# Result Frames
result_frame_1 = tk.Frame(main_frame)
result_frame_1.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

# Word Cloud
wordcloud_frame = tk.Frame(result_frame_1, width=200, height=400)
wordcloud_frame.grid(row=1, column=2, padx=10, pady=5, sticky="w")

topic_label = tk.Label(result_frame_1, text="Topics:")
topic_label.grid(row=0, column=0, sticky="w")
topic_text = scrolledtext.ScrolledText(result_frame_1, width=80, height=12,wrap="word")
topic_text.grid(row=1, column=0, padx=10, pady=5)

keyword_result_label = tk.Label(result_frame_1, text="Keyword Search Results:")
keyword_result_label.grid(row=2, column=0, sticky="w")
keyword_result_text = scrolledtext.ScrolledText(result_frame_1, width=80, height=12,wrap="word")
keyword_result_text.grid(row=3, column=0, padx=10, pady=5)

# Add scrollbar to the keyword_result_text widget
scrollbar = tk.Scrollbar(result_frame_1, orient="vertical", command=keyword_result_text.yview)
scrollbar.grid(row=3, column=1, sticky="ns")
keyword_result_text.configure(yscrollcommand=scrollbar.set)


result_frame_2 = tk.Frame(main_frame)
result_frame_2.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

ngrams_label = tk.Label(result_frame_2, text="Common Ngrams:")
ngrams_label.grid(row=0, column=1, sticky="w")
ngrams_text = scrolledtext.ScrolledText(result_frame_2, width=80, height=12,wrap="word")
ngrams_text.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

summary_label = tk.Label(result_frame_2, text="Article Summaries:")
summary_label.grid(row=0, column=0, sticky="w")
summary_text = scrolledtext.ScrolledText(result_frame_2, width=80, height=12,wrap="word")
summary_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

result_frame_3 = tk.Frame(main_frame)
result_frame_3.grid(row=5, column=0,rowspan=1, columnspan=2, padx=10, pady=5, sticky="nsew")

# Create a custom style for the Treeview
style = ttk.Style()
style.configure("Custom.Treeview", font=('Helvetica', 8))  # Set font size

# Sentiment Metrics Table
sentiment_label = tk.Label(result_frame_3, text="Sentiment Metrics")
sentiment_label.pack(side="bottom")

sentiment_table = ttk.Treeview(result_frame_3, columns=("Category", "Coherence", "Perplexity"), style="Custom.Treeview")
sentiment_table.heading("#0", text="", anchor="w")
sentiment_table.heading("Category", text="Category")
sentiment_table.heading("Coherence", text="Coherence")
sentiment_table.heading("Perplexity", text="perplexity")

# Adjust the overall width and height of the Treeview
sentiment_table.pack(side="bottom", fill="both", expand=True)
sentiment_table.config(height=3)  # Set the height


# Configure grid weights for resizing
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.columnconfigure(2, weight=1)
main_frame.rowconfigure(3, weight=1)

# Add scrollbar to the overall output window
scrollbar = tk.Scrollbar(main_frame, orient="vertical")
scrollbar.grid(row=0, column=3, rowspan=4, sticky="ns")

root.mainloop()
