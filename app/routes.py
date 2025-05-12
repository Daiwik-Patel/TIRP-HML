from flask import request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
import pandas as pd
from Models.app.models.predict import predict
import os

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Read uploaded CSV and make prediction
            df = pd.read_csv(filepath)
            pred = predict(df.values)  # assumes numeric input only
            return f'Predictions: {pred.tolist()}'

    return render_template('upload.html')
