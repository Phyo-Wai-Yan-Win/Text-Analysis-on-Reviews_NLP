# Importing Python Libraries and Modules
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# load data
df = pd.read_csv('MSU_reviews.csv')

# Sentiment labeling
def label_sentiment(rating):
    if rating > 3:
        return 'positive'
    else:
        return 'negative'

# apply function to create new column for sentiment
df['sentiment'] = df['rating'].apply(label_sentiment)

custom_stop_words = {"n't", "'s", "’", "“", "”", "``", "''", "--", "university", "midwestern", "MSU", "dont"}
stop_words = set(stopwords.words('english')).union(custom_stop_words)
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text, is_topic_modeling=False):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens if token not in stop_words]
    if is_topic_modeling:
        tokens = [token for token, pos in nltk.pos_tag(tokens) if pos.startswith('N') or pos.startswith('J')]
    return " ".join(tokens)

df['clean_text'] = df['body'].apply(clean_text)
df['topic_clean_text'] = df['body'].apply(lambda x: clean_text(x, is_topic_modeling=True))

# Vectorize data for sentiment analysis
tfidf_sentiment = TfidfVectorizer()
X_sentiment = tfidf_sentiment.fit_transform(df['clean_text'])
y = df['sentiment']

# Split data for sentiment analysis
X_train, X_test, y_train, y_test = train_test_split(X_sentiment, y, test_size=0.2, random_state=42)

# Train model
svc = LinearSVC()
svc.fit(X_train, y_train)

# Evaluate model
y_pred = svc.predict(X_test)

print("Sentiment Analysis Results:")
print(classification_report(y_test, y_pred))

# Sentiment analysis by categories
df['categories'] = df['categories'].apply(ast.literal_eval)
df_exploded = df.explode('categories')
sia = SentimentIntensityAnalyzer()
df_exploded['sentiment_scores'] = df_exploded['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
category_sentiment = df_exploded.groupby('categories')['sentiment_scores'].mean()
print("\nSentiment Scores by Categories:")
print(category_sentiment,'\n')

# Visualize sentiment scores by categories
plt.figure(figsize=(12, 6))
sns.barplot(x=category_sentiment.index, y=category_sentiment.values, palette='viridis')
plt.title('Sentiment Scores by Categories', fontsize=18)
plt.xlabel('Categories', fontsize=14)
plt.ylabel('Sentiment Scores', fontsize=14)
plt.xticks(rotation=60, fontsize=12, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Get top words for most positive and most negative reviews using TF-IDF
feature_names = np.array(tfidf_sentiment.get_feature_names_out())

# Get top N words for rating 4 & 5
N = 10
rating_4_5_indices = (df['rating'] == 4) | (df['rating'] == 5)
rating_4_5_tf_idf = X_sentiment[rating_4_5_indices]
rating_4_5_mean_tf_idf = rating_4_5_tf_idf.mean(axis=0).A1
rating_4_5_top_N_indices = rating_4_5_mean_tf_idf.argsort()[-N:]
rating_4_5_top_N_words = feature_names[rating_4_5_top_N_indices]

# Get top N words for rating 1 & 2
N = 10
rating_1_2_indices = (df['rating'] == 1) | (df['rating'] == 2)
rating_1_2_tf_idf = X_sentiment[rating_1_2_indices]
rating_1_2_mean_tf_idf = rating_1_2_tf_idf.mean(axis=0).A1
rating_1_2_top_N_indices = rating_1_2_mean_tf_idf.argsort()[-N:]
rating_1_2_top_N_words = feature_names[rating_1_2_top_N_indices]

print("Top words in rating 4 & 5 reviews:", ', '.join(rating_4_5_top_N_words))
print("Top words in rating 1 & 2 reviews:", ', '.join(rating_1_2_top_N_words), '\n')

# Word cloud for top words in rating 4 & 5 reviews
rating_4_5_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(rating_4_5_top_N_words))
plt.figure(figsize=(12, 6))
plt.imshow(rating_4_5_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Top words in rating 4 & 5 reviews", fontsize=18)
plt.show()

# Word cloud for top words in rating 1 & 2 reviews
rating_1_2_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(rating_1_2_top_N_words))
plt.figure(figsize=(12, 6))
plt.imshow(rating_1_2_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Top words in rating 1 & 2 reviews", fontsize=18)
plt.show()

# topic classification
n_topics = 3

# Vectorize data for topic modeling
tfidf_topic = TfidfVectorizer()
X_topic = tfidf_topic.fit_transform(df['topic_clean_text'])

# Fit the LDA model
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X_topic)

# Print the top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

n_top_words = 10
tfidf_topic_feature_names = tfidf_topic.get_feature_names_out()
print_top_words(lda, tfidf_topic_feature_names, n_top_words)

# Extract the coefficients of the LinearSVC model
coef = svc.coef_[0]

# Get the indices of the top 10 positive and negative coefficients
top_pos_indices = np.argsort(coef)[-10:]
top_neg_indices = np.argsort(coef)[:10]

# Get the corresponding feature names
top_pos_words = feature_names[top_pos_indices]
top_neg_words = feature_names[top_neg_indices]

# Get the corresponding coefficients
top_pos_coef = coef[top_pos_indices]
top_neg_coef = coef[top_neg_indices]

# Combine the positive and negative words and their coefficients
words_coef = list(zip(top_pos_words, top_pos_coef)) + list(zip(top_neg_words, top_neg_coef))

# Sort the words and their coefficients
words_coef.sort(key=lambda x: x[1], reverse=True)

# Separate the words and coefficients for the bar chart
words, coef = zip(*words_coef)

# Convert the words and coefficients to a DataFrame for easier plotting with seaborn
df_coef = pd.DataFrame({'Words': words, 'Coefficient': coef})

# Create a color map based on the sentiment of the word
df_coef['Color'] = df_coef['Coefficient'].apply(lambda coef: 'red' if coef > 0 else 'blue')

# Sort the DataFrame by Coefficient for proper visualization
df_coef = df_coef.sort_values('Coefficient')

# Create the plot
plt.figure(figsize=(10, 10))
sns.barplot(x=df_coef['Coefficient'], y=df_coef['Words'], palette=df_coef['Color'])
plt.xlabel("\nCoefficient Magnitude\n\nNote: Red = Positive Sentiment | Blue = Negative Sentiment", fontsize=14)
plt.ylabel("")
plt.title('Top Features in Positive and Negative Reviews', fontsize=18)

# Add the coefficient values on top of the bars
for i in range(df_coef.shape[0]):
    plt.text(df_coef['Coefficient'].iloc[i], i, round(df_coef['Coefficient'].iloc[i], 2), va = 'center')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()