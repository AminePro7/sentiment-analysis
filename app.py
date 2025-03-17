from flask import Flask, render_template, request, jsonify
import pickle
import keras
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Global variables
stopword = set(stopwords.words('english'))
max_len = 200

# Load models
def load_all_models():
    try:
        # Load the vectorizer
        with open('results/vectoriser.pickle', 'rb') as file:
            vectorizer = pickle.load(file)
        
        models_dict = {'vectorizer': vectorizer}
        
        # Load ML models with error handling
        model_files = {
            'logistic_regression': 'results/logisticRegression.pickle',
            'svm': 'results/SVM.pickle',
            'naive_bayes': 'results/NaivesBayes.pickle'
        }
        
        for model_name, file_path in model_files.items():
            try:
                with open(file_path, 'rb') as file:
                    models_dict[model_name] = pickle.load(file)
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")
                
        # Load RNN model
        try:
            models_dict['rnn'] = keras.models.load_model('results/rnn_model.hdf5')
        except Exception as e:
            print(f"Error loading RNN model: {str(e)}")
        
        return models_dict
    except Exception as e:
        print(f"Error in load_all_models: {str(e)}")
        return None

# Text preprocessing
def process_tweets(tweet):
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    
    # Lower Casing
    tweet = tweet.lower()
    # Removing all URls 
    tweet = re.sub(urlPattern,'',tweet)
    # Removing all @username
    tweet = re.sub(userPattern,'', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
    # Tokenizing words (simple split for robustness)
    tokens = tweet.split()
    # Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]
    # Reducing words to their base form
    wordLemm = WordNetLemmatizer()
    finalwords = []
    for w in final_tokens:
        if len(w) > 1:
            word = wordLemm.lemmatize(w)
            finalwords.append(word)
    return ' '.join(finalwords)

# Load all models at startup
models = load_all_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'success': False, 'error': 'Models not loaded properly'})
        
    try:
        text = request.json['text']
        processed_text = process_tweets(text)
        results = {}
        
        # ML Models predictions
        if 'vectorizer' in models:
            ml_input = models['vectorizer'].transform([processed_text])
            
            for model_name in ['logistic_regression', 'svm', 'naive_bayes']:
                if model_name in models:
                    try:
                        pred = models[model_name].predict(ml_input)[0]
                        results[model_name] = 'Positive' if pred == 1 else 'Negative'
                    except Exception as e:
                        results[model_name] = f'Error: {str(e)}'
        
        # RNN prediction
        if 'rnn' in models:
            try:
                sequence = pad_sequences([[1]], maxlen=max_len)
                rnn_pred = models['rnn'].predict(sequence)
                results['rnn'] = 'Positive' if rnn_pred[0][0] > 0.5 else 'Negative'
            except Exception as e:
                results['rnn'] = f'Error: {str(e)}'
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
