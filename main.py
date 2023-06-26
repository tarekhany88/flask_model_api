from flask import Flask, request, jsonify
import offensive_detection

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    data = request.json
    text = data.get('text')

    if text:
        result = offensive_detection.predict_sentiment(text)
        return jsonify({'prediction': result})
    else:
        return jsonify({'error': 'Missing "text" argument'})


@app.route('/feedback', methods=['POST'])
def process_feedback():
    data = request.json
    text = data.get('text')
    correct_pred = data.get('correct_pred')

    if text and correct_pred is not None:
        offensive_detection.feedback_learing(text, correct_pred)
        return jsonify({'status': 'Feedback has been sent successfully!'})
    else:
        return jsonify({'error': 'Missing "text" or "correct_pred" argument'})


if __name__ == '__main__':
    app.run()
