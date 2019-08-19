from flask import Flask, request, render_template
from predict import Predictor

app = Flask(__name__)
predictor = Predictor()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        text = request.form['email']
        pred_class, pred_prob = predictor.predict_input_class(text)

        return render_template('prediction.html', probability = pred_prob, class_pred= pred_class)
