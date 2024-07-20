import pandas as pd
from sklearn.utils import shuffle

# Read the CSV files
true_news_df = pd.read_csv('True.csv')
fake_news_df = pd.read_csv('Fake.csv')

# Add a label column
true_news_df['label'] = 1  # 1 for true news
fake_news_df['label'] = 0  # 0 for fake news

# Concatenate the DataFrames
combined_df = pd.concat([true_news_df, fake_news_df])

# Shuffle the combined DataFrame
combined_df = shuffle(combined_df)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('Combined.csv', index=False)

print("The data has been merged and saved to combined_news.csv")
