from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import json
import pandas as pd

def get_reviews(driver, api_url, entity_id, rating=None, pages=5):
    reviews = []
    page = 1
    limit = 20

    while page <= pages:
        url = f'{api_url}?e={entity_id}&page={page}&limit={limit}'
        if rating is not None:
            url += f'&rating={rating}'
        driver.get(url)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'pre')))

        pre_element = driver.find_element(By.TAG_NAME, 'pre')
        data = json.loads(pre_element.text)

        if not data.get("reviews"):
            break

        for review in data['reviews']:
            re = {
                'guid' : review['guid'],
                'body': review['body'],
                'rating': review['rating'],
                'author': review['author'],
                'created': review['created'],
                'categories': review['categories'],
            }
            reviews.append(re)

        page += 1
        sleep(7)

    return reviews

def scrape_reviews(entity_id, ratings=None, pages=5):
    driver = webdriver.Safari()
    api_url = 'https://www.niche.com/api/entity-reviews/'
    all_reviews = []

    if ratings:
        for rating in ratings:
            reviews = get_reviews(driver, api_url, entity_id, rating, pages)
            all_reviews.extend(reviews)
            sleep(7)
    else:
        reviews = get_reviews(driver, api_url, entity_id, pages=pages)
        all_reviews.extend(reviews)

    driver.quit()

    df = pd.DataFrame(all_reviews)
    return df

# Usage
# Scrape specific ratings for MSU
df_MSU = scrape_reviews("fae34fc1-40a9-4b23-9ae1-781dd361796b", ratings=[1,2,3,4,5], pages=2)

# Scrape specific ratings for UT
df_UT = scrape_reviews("bc90e2b6-e112-43ed-ac5c-3548829ea3dd", ratings=[1,2,3,4,5], pages=5)

# Scrape top reviews (biased sample) for UT
df_biased = scrape_reviews("bc90e2b6-e112-43ed-ac5c-3548829ea3dd", pages=25)

# Save as CSV files
df_MSU.to_csv('MSU_reviews.csv', index=False)
df_UT.to_csv('UT_reviews.csv', index=False)
df_biased.to_csv('Biased_samples.csv', index=False)