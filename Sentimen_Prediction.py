import pandas as pd
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv('UT_reviews.csv')

# function to label sentiment based on rating
def label_sentiment(rating):
    if rating > 3:
        return 'positive'
    else:
        return 'negative'

# apply function to create new column for sentiment
df['sentiment'] = df['rating'].apply(label_sentiment)

custom_stop_words = {"n't", "'s", "’", "“", "”", "``", "''", "--"}
stop_words = set(stopwords.words('english')).union(custom_stop_words)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['body'].apply(clean_text)

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

def predict_sentiment(review_text):
    clean_review = clean_text(review_text)
    vectorized_review = tfidf_sentiment.transform([clean_review])
    sentiment_prediction = svc.predict(vectorized_review)[0]
    return sentiment_prediction

reviews = [
"I absolutely loved my time at this university! The professors were knowledgeable and passionate about their subjects, and the campus was beautiful and well-maintained. I also made lifelong friends through the clubs and organizations on campus.",
"The academics at this university are top-notch. The coursework is challenging but rewarding, and the professors are always willing to help students succeed. The career services department is also fantastic and helped me secure a great job after graduation.",
"I was impressed with the diversity and inclusivity on this campus. The university has a strong commitment to creating a welcoming environment for all students, regardless of their background. There are also numerous opportunities to learn about different cultures and perspectives.",
"The facilities on this campus are exceptional. The library is well-stocked and has plenty of quiet study spaces, and the gym has top-of-the-line equipment. The dorms are also comfortable and spacious, and the dining options are varied and delicious.",
"I appreciated the emphasis on hands-on learning at this university. Many of my classes involved real-world projects and experiences, which helped me apply what I was learning in a practical way. This made me feel well-prepared for my future career.",
"This university was a huge disappointment. The professors were disorganized and unengaged, and many of the courses felt like a waste of time. The campus was also run-down and not very appealing.,",
"The administration at this university was frustrating to deal with. It was difficult to get clear answers to my questions or resolve issues I was having. I also found the financial aid process to be unnecessarily complicated.",
"I was disappointed by the lack of diversity and inclusion on this campus. There were few resources or initiatives dedicated to supporting underrepresented students, and I often felt like an outsider.",
"The facilities at this university were subpar. The dorms were cramped and outdated, and the dining options were limited and unappetizing. The library was often overcrowded and noisy, making it difficult to study.",
"I found the academic rigor at this university to be overwhelming. The coursework was often too difficult or unclear, and the workload was excessive. I also found the professors to be unapproachable and unsympathetic to student concerns.",
]
for review in reviews:
    predicted_sentiment = predict_sentiment(review)
    print(f"{review}\n{predicted_sentiment}\n")

# Plot distribution of ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=df, palette='viridis')
plt.title('Distribution of Ratings', fontsize=18)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()