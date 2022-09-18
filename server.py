import numpy as np 
import pandas as pd
from flask import Flask,request,jsonify,render_template
import joblib
server=Flask(__name__)
model=joblib.load("Recommendation_Model.pkl")
@server.route('/')
def home():
    return render_template('index.html')

@server.route('/predict',methods=['POST'])
def recommend(movie):
    index = new_movies[new_movies['title'] == movies].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    
    
    for i in movies_list:
        print(new_movies.iloc[i[0]].title)

@server.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    server.run(debug=True)