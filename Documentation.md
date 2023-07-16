# Project Code Overview
## Data Collection (Scraping.py)
This stage of the project involves collecting reviews of universities from the website Niche.com. We have chosen two universities for this project - Midwestern State University (MSU) and The University of Texas (UT).
To scrape the reviews, we use the Selenium WebDriver. This tool provides a programming interface to create and execute test cases in a variety of web applications. In this case, we use it to automate the browser to navigate to the review pages and extract the reviews.
The main function for scraping is scrape_reviews(entity_id, ratings=None, pages=5), which takes as input the entity_id (the unique identifier for the university on niche.com), ratings (optional, a list of ratings to filter the reviews), and pages (the number of pages of reviews to scrape).
Inside the scrape_reviews function, a Selenium WebDriver is initialized. The function then calls get_reviews(driver, api_url, entity_id, rating, pages), which navigates to the review page and extracts the reviews.
The get_reviews function constructs a URL to the review page, taking into account the entity_id, rating (if provided), and the page number. It navigates to the URL and waits until the page loads.
Once the page has loaded, the function finds the element containing the review data, which is in JSON format. It parses the JSON data and extracts the following information for each review:
- guid: The unique identifier for the review
- body: The text of the review
- rating: The rating given by the reviewer
- author: The author of the review
- created: The date the review was created
- categories: The categories the review falls under
The function collects the reviews for the specified number of pages, sleeping for 7 seconds between each page to avoid overloading the server.
Finally, the scrape_reviews function combines all the reviews into a DataFrame and saves them as a CSV file.
In the usage section, we scrape specific ratings (1 to 5) for MSU and UT, and the top reviews (which form a biased sample) for UT. These data are saved as 'MSU_reviews.csv', 'UT_reviews.csv', and 'Biased_samples.csv' respectively.

<div align="center">
  <img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/8bdc0571-0e0d-4798-8a5f-096129e408f2" width="70%" alt="Image">
</div>

## Model building & Analysis (Analyze_MSU.py & Analyze_UT.py)
The provided code performs text analysis on two sets of review data from two universities: Midwestern State University (MSU) and University of Texas at Austin (UT). The analysis includes sentiment analysis, category-wise sentiment scores, topic modeling, and keyword extraction. Here's a step-by-step explanation of each part of the code.

**Import Libraries**

The code begins by importing the necessary libraries. These include:
- pandas for data manipulation and analysis
- nltk for text processing and analysis
- sklearn for machine learning and data analysis
- matplotlib and seaborn for data visualization
- numpy for numerical operations
- wordcloud to create word clouds

**Load Data**

The data is loaded from two CSV files using the pandas.read_csv() function. These files contain reviews for Midwestern State University (MSU) and University of Texas at Austin (UT).

**Label Sentiment**

A new column named 'sentiment' is created in the dataframe. The sentiment is labeled as 'positive' if the rating is greater than 3, and 'negative' otherwise.

**Text Preprocessing**

Next, the review text is preprocessed. This involves the following steps:

- Removing punctuation
- Tokenizing the text into individual words
- Converting all text to lowercase
- Removing stopwords, i.e., common words like 'is', 'the', 'and', etc., that do not contribute much information for text analysis
- Lemmatizing the words, i.e., converting words to their base form (for example, 'running' becomes 'run')

This preprocessed text is saved in two new columns in the dataframe: 'clean_text' and 'topic_clean_text'. The 'topic_clean_text' column is further processed to only include nouns and adjectives, as these parts of speech are generally more informative for topic modeling.

**Sentiment Analysis**

The preprocessed text is vectorized using the TF-IDF method, which converts the text into numerical data that can be used for machine learning. The data is then split into a training set and a test set.

A Linear Support Vector Classifier (LinearSVC) model is trained on the training set and used to predict the sentiment of the reviews in the test set. The performance of the model is evaluated using a classification report, which provides metrics such as precision, recall, and F1-score.

**Sentiment Analysis by Categories**

The reviews are grouped by categories, and the average sentiment score for each category is calculated. This is done by first converting the 'categories' column to a list (as it is currently a string representation of a list), and then using the explode() function to create a new row for each category in a review. The Sentiment Intensity Analyzer from NLTK is used to calculate the sentiment scores of the reviews. The mean sentiment score for each category is then calculated.
The sentiment scores by category are also visualized using a bar plot.

<div align="center">
  <img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/f633ce4d-a3c8-4937-9f67-7161edcd6804" width="70%" alt="Sentiment_UT">
</div>

<div align="center">
  <img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/f9f96296-0da9-4aab-9355-e6ca4c7c78d0" width="70%" alt="Sentiment_MSU">
</div>


**Keyword Extraction**

The top 10 words for the most positive (ratings 4 and 5) and the most negative (ratings 1 and 2) reviews are extracted using TF-IDF. These words are the ones with the highest average TF-IDF scores in their respective groups of reviews.
Word clouds are also created for the top words in the most positive and most negative reviews.

<table align="center">
  <tr>
    <td><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/29cd7e50-b721-4f28-b0c2-30a203cd7af4" alt="WordCloud_MSU_P" width="400"></td>
    <td><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/2c53b685-57f1-4ab5-bf02-7e001ec6235c" alt="WordCloud_MSU_N" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/446579cc-83b8-46c1-b4e6-2965b53342d5" alt="WordCloud_UT_P" width="400"></td>
    <td><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/7df7f960-f35a-44fd-bf04-941c3110fcf9" alt="WordCloud_UT_N" width="400"></td>
  </tr>
</table>


**Topic Modeling**

Topic modeling is performed using the Latent Dirichlet Allocation (LDA) method. The preprocessed text is vectorized using the TF-IDF method, and the LDA model is fit to this data. The top 10 words for each topic are printed.

**Features in Positive and Negative Reviews**

The top 10 positive and negative features (words) in the reviews are extracted from the coefficients of the LinearSVC model. The features with the highest positive coefficients are the ones that are most indicative of a positive sentiment, and the features with the highest negative coefficients are the ones that are most indicative of a negative sentiment.

These features and their coefficients are visualized using a horizontal bar chart. The color of the bars indicates the sentiment of the feature (red for positive, blue for negative), and the magnitude of the coefficient is displayed on top of each bar.
This entire process is repeated for both datasets (MSU and UT).

<table align="center">
  <tr>
    <td><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/d02c84aa-74d4-48b3-aeb3-c30234112f45" alt="TopWords_UT" width="400"></td>
    <td><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/4f4746c3-9b18-40ab-ad79-08dca6e4576b" alt="TopWords_MSU" width="400"></td>
  </tr>
</table>

## Sentiment_Prediction.py

This script conducts sentiment analysis on a dataset of university reviews, following the same procedure as in "Analyze_MSU.py" and "Analyze_UT.py". The script processes the text data, transforms it into a matrix of TF-IDF features, and trains a Linear Support Vector Classifier (SVC) on these features. The performance of the model is evaluated using a classification report.

In addition, the script uses the trained model to predict sentiments for a new set of artificially generated reviews (collected from ChatGPT). This demonstrates the model's ability to generalize and predict sentiments for unseen data.

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Negative     | 0.74      | 0.94   | 0.83     | 52      |
| Positive     | 0.91      | 0.65   | 0.76     | 48      |
| **Accuracy** |           |        | 0.80     | 100     |
| Macro Avg    | 0.83      | 0.79   | 0.79     | 100     |
| Weighted Avg | 0.82      | 0.80   | 0.79     | 100     |

<img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/54370b77-1633-4fab-bd14-801aacca8fa3" alt="Balanced" width="400">

## Biased_Prediction.py

This script is identical to Sentiment_Prediction.py, but it uses a heavily biased dataset that contains predominantly positive reviews. The procedure for model training and evaluation is the same.

However, the model's performance is notably poor in this case. The classification report reveals high precision for the 'positive' class but extremely low recall for the 'negative' class. This indicates that the model struggles to correctly identify negative sentiments, likely due to the lack of negative examples during training.

When the model is used to predict sentiments for the same set of artificially generated reviews, it incorrectly classifies even clearly negative reviews as 'positive'. This highlights the potential issues of training a model on a biased dataset: the model may not accurately represent sentiments that are underrepresented in the training data. The results underscore the importance of using balanced datasets for training machine learning models.

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Negative     | 1.00      | 0.00   | 0.00     | 9       |
| Positive     | 0.91      | 1.00   | 0.95     | 91      |
| **Accuracy** |           |        | 0.91     | 100     |
| Macro Avg    | 0.96      | 0.50   | 0.48     | 100     |
| Weighted Avg | 0.92      | 0.91   | 0.87     | 100     |

<img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/8d159182-637b-410f-a5e6-893132fe7ad9" alt="Biased" width="400">

## Conclusion
This document has provided a comprehensive guide to the 'Text Analysis on University Reviews' project, covering everything from the data collection stage to sentiment prediction. We hope this documentation is helpful for understanding the project and its results.

