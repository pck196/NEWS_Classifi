import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
tv = pickle.load(open('tv.pickle', 'rb'))
svc = pickle.load(open('svc.pickle', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    new_data = [str(x) for x in request.form.values()]
    #new_data
    new_vector =tv.transform(new_data)
    pred = svc.predict(new_vector)

    return render_template('index.html', prediction_text='Category of the news should be {}'.format(pred[0].upper()))



if __name__ == "__main__":
    app.run(debug=True)