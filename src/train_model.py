import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

# Loding clean data
df = pd.read_csv("notebook/data/data_clean.csv")
print(df.head(1))

# Selecting input and output features
# Features and label
x = df[['Product Title', 'title_char_count', 'title_longest_word_len']]
y = df['category']

# Defining preprocessor
# Preprocessor TF-IDF for text, MinMaxScaler for numeric feature
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "Product Title"),
        ("char_count", MinMaxScaler(), ["title_char_count"]),
        ("longest_word_len", MinMaxScaler(), ["title_longest_word_len"])
    ]
)

# Creating pipeline with the algorithm with best results
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])

# Train the model on the entire dataset
pipeline.fit(x, y)

# Save the model to a file
joblib.dump(pipeline, "model/final_model.pkl")

print("Model trained and saved as 'model/final_model.pkl'")