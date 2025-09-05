import joblib
import pandas as pd
import re

# Load the saved model
model = joblib.load("model/final_model.pkl")

print("Model loaded successfuly!")
print("Type 'exit' at any point to stop.\n")

while True:
    product_title = input("Enter product title:  ")
    if product_title.lower() == "exit":
        print("Exiting...")
        break
    
    # Calculate number of characters in title
    title_char_count = len(product_title)
    # Calculate longest word in title length
    s = product_title.strip()
    words = re.findall(r"\w+", s)
    title_longest_word_len = max(map(len, words), default=0)
    
    # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "Product Title": product_title,
        "title_char_count": title_char_count,
        "title_longest_word_len": title_longest_word_len
    }])
    
    # Predict category
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction} \n" + "-" * 40)