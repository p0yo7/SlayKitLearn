from flask import Flask, jsonify, request
import warnings
warnings.filterwarnings("ignore")
from funciones import all_predictions

    
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id parameter'}), 400
    
    try:
        result = all_predictions(user_id)
        if result is None:
            return jsonify({'error': 'User not found or has no transactions'}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)