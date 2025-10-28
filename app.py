from flask import Flask, render_template, request, jsonify
from verifier import verify_signatures
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify-signatures', methods=['POST'])
def verify():
    cheque = request.files.get('file1')
    reference = request.files.get('file2')

    if not cheque or not reference:
        return jsonify({"error": "Missing file(s)"}), 400

    result, confidence = verify_signatures(cheque, reference)
    return jsonify({
        "result": result,
        "confidence": confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
