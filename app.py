import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    output_ = {0:"A",1:"B",2:"C"}

    return render_template('index.html', prediction_text='Wine belongs to class: {}'.format(output_[output]))

if __name__ == "__main__":
    app.run(debug=False)