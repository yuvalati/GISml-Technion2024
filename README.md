# Fake News Detection with Geographical Context

## Project Summary

This project aims to develop a machine learning system for detecting fake news, with an additional focus on on the geographical context of it. The project combines the fake news detection with a second model that evaluates the geographical distribution of the news. This allows for better analysis by visualizing the locations where fake or real news is most common, providing a spatial layer to fake news detection.

## Objectives

- **Detect Fake News**: Build a primary model that predicts whether a news article is real or fake using common textual features of fake news .
- **Evaluate Geographical Context**: Implement a secondary model that evaluates the location information of the news.
- **Visualize on Map**: Present the locations of the predicted fake and real news articles on a map for a geographical analysis of disinformation trends.
- **Assess Impact**: Examine how geographical patterns can indicate clusters of disinformation and provide insights into localized disinformation campaigns.

## Key Features

- **Text-Based Detection**: natural language processing (NLP) techniques.
- **Geographical Modeling**: After the fake news prediction, a secondary model analyzes the news' origin and referenced locations to detect geographical patterns.
- **Visualization**: The final output includes a map-based visualization, where the locations of predicted fake and real news are displayed, allowing users to see the geographical distribution and potential hotspots for fake/real news.
  
## Dataset

- **News Dataset**: Includes a collection of two datsets where one if real new and the other is fake news, with approximatly 20,000 articles in each one of them.
- **Geographical Data**: The dataset also includes the source location of the news articles as well as any geographical references made within the articles.

## Project Workflow
1. **Merging the two datasets into one**:
   - Producing one dataset which consists of all the rows from the datasets, adding a new column for label - 0 for fake news, 1 for real news. 

2. **Fake News Detection**:
   - The first model classifies news articles as real or fake based on textual data.
  
3. **Geographical Analysis**:
   - A secondary model evaluates the geographical context by analyzing where the news originates and what locations are mentioned within the articles.
  
4. **Map Visualization**:
   - The output of both models is visualized on a map, highlighting the locations of the predicted fake and real news, offering a spatial understanding of disinformation.

## Project Structure

```plaintext
├── data_links.md/          # Contains datasets
├── notebooks/              # Jupyter notebooks
├── src/                    # Source codes 
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── geo_model.py        # Evaluates geo-location
│   ├── map_visualization.py # Visualizes results on a map
├── models/                 # Saved models for prediction and evaluation
├── README.md               # Project overview and documentation
