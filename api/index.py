from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__, template_folder='../templates')


def img_to_base64(img_array):
    """Mengubah array gambar OpenCV menjadi string Base64 untuk ditampilkan di HTML"""
    _, buffer = cv2.imencode('.png', img_array)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400

    file = request.files['image']
    manual_val = int(request.form.get('threshold', 127))

    # Baca gambar langsung dari memori (tanpa simpan ke hardisk)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Format gambar tidak valid'}), 400

    # 1. Proses Pengolahan Citra (Sesuai algoritma asli Anda)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    _, thresh_manual = cv2.threshold(
        gray_eq, manual_val, 255, cv2.THRESH_BINARY)
    otsu_val, thresh_otsu = cv2.threshold(
        gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    comparison = cv2.hconcat([thresh_manual, thresh_otsu])

    # 2. Fungsi Hitung Histogram (Output berupa list Python untuk Chart.js)
    def get_hist(image, channel=0):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        return [int(x[0]) for x in hist]

    hist_r = get_hist(img_rgb, 0)
    hist_g = get_hist(img_rgb, 1)
    hist_b = get_hist(img_rgb, 2)
    hist_gray = get_hist(gray, 0)
    hist_eq = get_hist(gray_eq, 0)
    hist_manual = get_hist(thresh_manual, 0)
    hist_otsu = get_hist(thresh_otsu, 0)

    # 3. Kembalikan gambar (Base64) dan data histogram (JSON) ke Frontend
    return jsonify({
        'otsu_value': round(float(otsu_val), 1),
        'manual_value': manual_val,
        'images': {
            'original': img_to_base64(img),
            'gray': img_to_base64(gray),
            'equalized': img_to_base64(gray_eq),
            'manual': img_to_base64(thresh_manual),
            'otsu': img_to_base64(thresh_otsu),
            'comparison': img_to_base64(comparison)
        },
        'histograms': {
            'rgb': {'r': hist_r, 'g': hist_g, 'b': hist_b},
            'gray': hist_gray,
            'equalized': hist_eq,
            'manual': hist_manual,
            'otsu': hist_otsu
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
