from flask import Flask, request, jsonify
import os
import base64

import limit_knn as script_4

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Get the selected option
    option = request.form.get('option', '')

    # Call the script with the file path
    script_4.main(file_path)

    # Return a success response
    return jsonify({'message': 'File successfully uploaded and processed'})

  

if __name__ == '__main__':
    app.run(debug=True, port=5000)
