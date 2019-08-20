from flask import Flask, request, render_template, jsonify
from predict import Predictor

app = Flask(__name__)
predictor = Predictor()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        text = request.form['text']
        pred_class, pred_prob = predictor.predict_input_class(text)
        return render_template('prediction.html', probability = pred_prob, class_pred= pred_class)

@app.route('/v1/api/<text>')
def api_predict(text):
    pred_class, pred_prob = predictor.predict_input_class(text)
    return jsonify({pred_class: pred_prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0')