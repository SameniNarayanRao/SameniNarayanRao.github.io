from flask import Flask,
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("randomforest.pkl", "rb"))

@app.route('/')
def main():
    return ('index.html')

@app.route('/predict', methods = ['POST'])
def index():
    Age = request.form['a']
    Sex = request.form['b']
    Height = request.form['c']
    Weight = request.form['d']
    array = np.array([[Age, Sex, Height, Weight]])
    pred = model.predict(array)
    return ("result.html", data = pred)



if __name__ == "__main__":
    app.run(debug = True)








