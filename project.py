import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import re

# Step 1: Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        urls = file.readlines()
    return urls

# Step 2: Preprocess URLs
def preprocess_urls(urls):
    processed_urls = []
    for url in urls:
        # Remove 'get' and 'post' prefixes
        url = re.sub(r'^(get|post)', '', url, flags=re.IGNORECASE)
        
        # Remove 'http://' prefix
        url = url.replace('http://', '')
        
        # Remove special characters and encoding issues
        url = url.encode('ascii', 'ignore').decode()
        
        # Normalize URLs (e.g., convert to lowercase)
        url = url.lower()
        
        # Remove query parameters
        url = url.split('?')[0]
        
        # Remove trailing slashes
        url = url.rstrip('/')
        
        # Remove leading and trailing whitespaces
        url = url.strip()
        
        processed_urls.append(url)
    
    return processed_urls

# Step 3: Main function
def main():
    # Load and preprocess normal URLs
    normal_urls = load_dataset('normalRequestTraining.txt')
    normal_urls = preprocess_urls(normal_urls)

    # To test on normalURLs testing dataset, uncomment.
    # X_train, X_test = train_test_split(normal_urls, test_size=0.2, random_state=42)

    # Load and preprocess anomalous URLs
    # Comment these for normalURLs testing
    anomalous_urls = load_dataset('anomalousRequestTest.txt')
    anomalous_urls = preprocess_urls(anomalous_urls)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Combine normal and anomalous URLs for fitting the vectorizer
    all_urls = normal_urls + anomalous_urls

    # Uncomment to test on normalURLs and comment above cmd
    # all_urls = normal_urls

    # Fit TF-IDF vectorizer on all URLs
    vectorizer.fit(all_urls)
    
    # Transform normal URLs
    # use X_train instead of normal_urls while testing normalURLS
    X_normal = vectorizer.transform(normal_urls)
    
    # Transform anomalous URLs
    # use X_test instead of anomalous_urls while testing normalURLS
    X_anomalous = vectorizer.transform(anomalous_urls)
    
    # Train Isolation Forest model
    model = IsolationForest(random_state=42, contamination=0.2) # Adjust contamination according to expected anomaly rate
    model.fit(X_normal)

    # Test model on anomalous URLs
    predictions = model.predict(X_anomalous)

    total_1s = (predictions == 1).sum()
    total_minus_1s = (predictions == -1).sum()
    
    # Calculate percentage of 1s among all predictions
    total_predictions = len(predictions)
    percentage_1s = (total_1s / total_predictions) * 100
    
    # 1s represent normal and -1s represent anomalies
    print("Total occurrences of 1s:", total_1s)
    print("Total occurrences of -1s:", total_minus_1s)
    print("Percentage of 1s among all predictions:", percentage_1s)

if __name__ == "__main__":
    main()
