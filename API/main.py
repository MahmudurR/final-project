from flask import Flask, request, render_template
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load pre-trained models
gaming_ar_model = joblib.load('models/gaming_ar_model.pkl')
gaming_vr_model = joblib.load('models/gaming_vr_model.pkl')
entertainment_ar_model = joblib.load('models/entertainment_ar_model.pkl')
entertainment_vr_model = joblib.load('models/entertainment_vr_model.pkl')

# Define preprocess function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in string.punctuation]
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sector = request.form['sector']
    technology = request.form['technology']
    
    cleaned_text = preprocess_text(text)
    
    if sector == 'gaming' and technology == 'ar':
        prediction = gaming_ar_model.predict([cleaned_text])
    elif sector == 'gaming' and technology == 'vr':
        prediction = gaming_vr_model.predict([cleaned_text])
    elif sector == 'entertainment' and technology == 'ar':
        prediction = entertainment_ar_model.predict([cleaned_text])
    elif sector == 'entertainment' and technology == 'vr':
        prediction = entertainment_vr_model.predict([cleaned_text])
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
