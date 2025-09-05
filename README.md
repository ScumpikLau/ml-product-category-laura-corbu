# ml-product-category-laura-corbu
# Product Category Analysis ML Project (Complete Pipeline)

This repository contains a complete machine learning pipeline for **category analysis** of products using Python and Scikit-learn

The project was developed as part of a learning module and represents the Final Task for finishing the course

## Project Structure
|—— data/
| ∟ IMLP4_TASK_03-products.csv
|—— notebook/
   | ∟ data/
      ∟ data_clean.csv
   | ∟ model_comparison_analysis.ipynb
   | ∟ product_category_analysis.ipynb
|—— src/  
   | ∟ train_model.py  # Script for training and saving the model 
   | ∟ predict_category.py # Script for testing saved model
∟ README.md

## What I Did for this Project

# 1.Project Setup
 - Created a new GitHub repository
 - Defined project folder structure
 - Uploaded **raw** and **cleaned** dataset

# 2. Data Exploration
 - Loaded and analyzed a large dataset with **product titles** and **features**
 - Used Matplotlib and Seaborn for visualizations 
 - Investigated distribution of product Category Labels

# 3. Data Cleaning and Preprocessing
 - Removed missing values 
 - Standardized Product Category labels
 - Converted 'Listing Date' column to DateTime
 - Explored the 'Product Title' column calculating the number of characters, number of words, presence of numbers and special characters, longest word length and if title contains brands name

# 4.Feature Engineering
 - After visualizing the plots ilustrating correlations, I selected meaninful input features: 'Product Title', 'title_char_count' and 'title_longest_word_len'
 - Removed irrelevant columns

# 5.Model Training and Evaluation
- Compared multiple ML models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM)
- Used Column Transformer and Pipeline for unified preprocessing
- Evaluated using precision, recall, F1-score and confusion matrix

# 6.Final Model Training
- Trained final model on full dataset
- Saved the pipeline using joblib to 'final_model.pkl'

# 7.Inference and Usage
- Loaded saved model
- Built an interactive interface for predicting category of new products by title
- Enabled real time testing via console input

## How to use 

  # Train the model:
  ```bash
   cd src
   python train_model.py
  ``` 

   - This will create a file called 'final_model.pkl' in the root directory

  # Run Inference
   - Use the interactive script ('predict_category.py') to classify new products using the trained model


 ## Author
 This repository was developed as part of an educational program on practical machine learning using Python

 ## Licence
 This project is open-source and freely available for educational use     

