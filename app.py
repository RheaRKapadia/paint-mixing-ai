from flask import Flask, render_template, request, redirect, url_for
from paint_mixing import get_paint_mixes, extract_dominant_colors  # or your main function
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(filepath)

   
            img = Image.open(filepath)
            dominant_rgb = extract_dominant_colors(filepath) 
            suggested_mix, final_color = get_paint_mixes(dominant_rgb)

            return render_template('index.html',
                                   suggested_mix=suggested_mix,
                                   final_color=final_color,
                                   uploaded=True)

    return render_template('index.html', uploaded=False)

if __name__ == '__main__':
    app.run(debug=True)

