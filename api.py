import cv2
import pytesseract
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

def image_to_text(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)

    # Text Recognition
    text = pytesseract.image_to_string(gray, lang='eng')

    return text

@app.route('/recognize_text', methods=['POST'])
def recognize_text():
    # Check if a file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    file_path = app.config['UPLOAD_FOLDER'] + '/' + filename
    file.save(file_path)

    # Recognize the text in the uploaded image
    recognized_text = image_to_text(file_path)

    # Return the recognized text as a JSON response
    response_data = {'recognized_text': recognized_text}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
