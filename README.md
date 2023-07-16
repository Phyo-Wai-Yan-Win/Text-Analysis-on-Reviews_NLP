# Text Analysis on University Reviews: From Data Collection to Insights Extraction
This repository contains a project that uses Natural Language Processing (NLP) techniques to extract insights from university reviews. The project covers various stages from data collection to model building, evaluation, and insights extraction using visualizations.
### Overview
The project uses reviews from two universities: The University of Texas and Midwestern State University, collected from niche.com. The main objectives are:
- **Data Collection**: Collecting reviews using Selenium WebDriver for web scraping.
- **Data Cleaning & Preprocessing**: Cleaning the text data and preparing it for analysis.
- **Sentiment Analysis**: Building a sentiment analysis model to understand the sentiment (positive/negative) in the reviews.
- **Topic Modeling**: Performing topic modeling to extract key themes or topics from the reviews.
- **Visualizing Insights**: Visualizing sentiment scores across different categories, creating word clouds, and identifying top features in positive and negative reviews.

Please note that this project is for educational purposes only and not intended for commercial use. As an MBA student at Midwestern State University, this project is part of my ongoing learning and skill development in the field of data science and machine learning.
### Data
The data used in this project are reviews collected from niche.com, encompassing reviews from The University of Texas and Midwestern State University. Each review includes the review text (body), rating, and categories.
### Methodology
The project involves several stages:
- **Data Collection**: Using Selenium WebDriver, a web scraping script is developed to collect reviews from niche.com. Both balanced and biased datasets are collected to illustrate the impact of data bias on model performance.
- **Data Cleaning & Preprocessing**: The reviews are cleaned and preprocessed, including text cleaning, tokenization, lemmatization, and sentiment labeling based on the review rating.
- **Feature Extraction**: TF-IDF vectorization is used to transform the preprocessed text data into feature vectors suitable for machine learning models.
- **Model Building**: A sentiment analysis model is built using a LinearSVC classifier. Also, Latent Dirichlet Allocation (LDA) is used to perform topic modeling on the reviews. The model is trained and evaluated on both the balanced and biased datasets.
- **Model Evaluation**: The sentiment analysis model is evaluated using unseen new data. The effects of data bias on model performance are demonstrated by comparing the model's performance on the balanced and biased datasets.
- **Insights Extraction**: Various visualizations are created to extract actionable business insights from the data. The management can leverage these insights and potentially increase student retention, strengthen its reputation and appeal to prospective students.
### Key Findings 
- ***The Power of Balanced Data***: The sentiment analysis model demonstrated an impressive accuracy of 80% when trained on a balanced dataset. This highlights the power of well-balanced, representative data in machine learning applications.
- ***Bias is a Model's Worst Enemy***: When the same model was trained on a biased dataset, it struggled to correctly identify negative sentiments. This stark contrast in performance serves as a stark reminder that bias in training data can severely cripple a model's ability to make accurate predictions.
- ***Machine Learning as a Mirror***: This project reaffirms a critical truth about machine learning - a model is essentially a mirror that reflects the characteristics of the data it is trained upon. The quality of the insights a model can provide is intrinsically tied to the quality of the data it learns from.
### Actionable Recommendations
- **Improving Campus Life**: Despite high satisfaction in 'Overall Experience' and 'Online Learning', MSU could further enhance student retention by improving 'Party Scene' and overall campus life through more events and better amenities.
- **Enhancing Academics**: Drawing from the University of Texas – Austin's high 'Academics' sentiment scores, MSU could improve its own academic offerings through innovative teaching methods and additional student resources.
- **Boosting Housing and Food Services**: Lower sentiment scores in 'Housing' and 'Food' categories suggest areas for improvement at MSU. Enhanced housing and food options could greatly improve overall student experience and contribute to higher retention rates.
### Technologies Used
- Python
- Selenium WebDriver
- Pandas
- NLTK
- Scikit-Learn
- Matplotlib
- Seaborn
### Project Structure
The project is divided into several Python scripts:
1. **Scraping.py**: Contains the code for collecting reviews using Selenium WebDriver.
2. **Analyze_MSU.py**: Contains the code for analyzing the reviews from Midwestern State University.
3. **Analyze_UT.py**: Contains the code for analyzing the reviews from The University of Texas.
4. **Sentiment_Prediction.py**: Contains the code for building and evaluating the sentiment analysis model.
5. **Biased_Prediction.py**: Contains the code for evaluating the model on a biased dataset.
### Running the Code
To run the code, open each Python script in PyCharm or any Python IDE and execute them. Ensure that all necessary libraries are installed in your environment.
### Output Screenshots
The README includes several screenshots from the project to provide a visual understanding of the insights that can be extracted:

<table>
  <tr>
    <td valign="top"><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/537e2192-b298-4370-a45b-38abd059486e" alt="Sentiment_MSU"></td>
    <td rowspan="2"><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/55f841d2-976f-4209-ad14-958d89b52ab7" alt="TopWords_MSU"></td>
  </tr>
  <tr>
     <td valign="top"><img src="https://github.com/Phyo-Wai-Yan-Win/Text-Analysis-on-Reviews_NLP/assets/132968939/6cdcc898-cc90-43de-b3a6-964d41bc3837" alt="WordCloud_MSU_P"></td>
  </tr>
</table>



These screenshots provide a glimpse of the visual insights that can be gained from the analysis.
### Future Work
This project can be extended in multiple ways:
- **Handling Data Bias**: In this project, the effects of data bias on model performance were illustrated. Future work could focus on strategies to handle data bias, such as oversampling, undersampling, or generating synthetic samples, such as (SMOTE) method.
- **Incorporating More Data**: More reviews can be collected from different sources or universities to create a more comprehensive analysis.
- **Using More Advanced Models**: More advanced models, such as deep learning models, can be used for the sentiment analysis or topic modeling.


