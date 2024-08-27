from flask import Flask, render_template, request
import pandas as pd
from src.preprocess import preprocess_data
from src.model import load_model
from src.predict import predict_sentiment
from src.visualize import create_pie_chart, create_prediction_graph

app = Flask(__name__)

# Load and preprocess data
data = preprocess_data('data/customer_reviews.csv')

predictions = []

@app.route('/')
def index():
    pie_chart = create_pie_chart(data)
    prediction_graph = create_prediction_graph(predictions)
    return render_template('index.html', pie_chart=pie_chart, prediction_graph=prediction_graph)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['review']
    sentiment, score = predict_sentiment(text)
    predictions.append({'text': text, 'sentiment': sentiment, 'score': score})
    
    pie_chart = create_pie_chart(data)
    prediction_graph = create_prediction_graph(predictions)
    
    return render_template('index.html', pie_chart=pie_chart, prediction_graph=prediction_graph, sentiment=sentiment, score=score)

if __name__ == "__main__":
    app.run(debug=True)
