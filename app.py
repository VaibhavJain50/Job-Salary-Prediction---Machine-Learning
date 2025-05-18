from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import nltk

app = Flask(__name__)

# Load the pre-trained components
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
encoders = pickle.load(open('encoder.pkl', 'rb'))

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

def combine_text_cols(row, cols):
    return ' '.join(row[col] for col in cols)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create a DataFrame from the input
        input_data = pd.DataFrame([data])
        
        print('Hello')
        # Preprocess categorical columns
        categorical_cols = ['Qualifications', 'location', 'Country', 'Work Type',
                           'Job Title', 'Role', 'Company']
        
        # for col in categorical_cols:
        #     input_data[col] = encoder.transform([input_data[col].values[0]])[0]

        for col2 in categorical_cols:
            le = encoders[col2]
    # Apply transformation with a fallback for unseen categories
            input_data[col2] = input_data[col2].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1).astype(np.int16) 
        
        # Process experience
        if 'Experience' in input_data.columns:
            input_data[['MinExp','MaxExp']] = input_data['Experience'].str.split(' to ', expand=True)
            input_data['MaxExp'] = input_data['MaxExp'].str.split(' ', expand=True)[0].astype('int64')
            input_data['MinExp'] = input_data['MinExp'].str.split(' ', expand=True)[0].astype('int64')
            input_data.drop(['Experience'], axis=1, inplace=True)
        
        # Combine text columns
        text_cols = ['Job Description', 'Responsibilities', 'skills']
        input_data['combined_text'] = input_data.apply(lambda row: combine_text_cols(row, text_cols), axis=1)
        input_data['Benefits_processed'] = input_data['Benefits'].apply(preprocess_text)
        input_data['job_desc_processed'] = input_data['combined_text'].apply(preprocess_text)
        
        # TF-IDF transformation
        tfidf_matrix = tfidf_vectorizer.transform(input_data['job_desc_processed'] + " " + input_data['Benefits_processed'])
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=tfidf_vectorizer.get_feature_names_out(), 
                               index=input_data.index)
        
        # Ensure all TF-IDF columns are present
        for col in tfidf_vectorizer.get_feature_names_out():
            if col not in df_tfidf.columns:
                df_tfidf[col] = 0
        
        df_tfidf = df_tfidf[tfidf_vectorizer.get_feature_names_out()]

        input_data=input_data[['Qualifications', 'location', 'Country', 'Work Type', 'Company Size',
       'Job Title', 'Role', 'Company', 'MinExp', 'MaxExp']]

        # Combine all features
        input_data = pd.concat([input_data, df_tfidf], axis=1)
        
        # Drop unnecessary columns
        cols_to_drop = ['Job Description', 'skills', 'Responsibilities', 'Benefits',
                       'combined_text', 'job_desc_processed', 'Benefits_processed']
        input_data.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
        
        # Make prediction
        prediction = model.predict(input_data)
        min_sal = round(prediction[0][0], 2)
        max_sal = round(prediction[0][1], 2)
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Salary Range: ${min_sal}K - ${max_sal}K')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error in prediction: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)