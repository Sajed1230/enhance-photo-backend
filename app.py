import os
import cv2
import torch
import io
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from gfpgan import GFPGANer

app = Flask(__name__)
CORS(app)

# إعداد الموديل (سيقوم بتحميل الأوزان تلقائياً في أول مرة)
# يمكنك وضع ملف .pth في مجلد اسمه weights
model_path = 'weights/GFPGANv1.3.pth' 

# إذا لم يكن الملف موجوداً، سيحاول الموديل تحميله من الإنترنت
restorer = GFPGANer(
    model_path=model_path,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None # هنا يمكنك إضافة RealESRGAN لاحقاً
)

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # المعالجة
        _, _, restored_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # تحويل النتيجة
        is_success, buffer = cv2.imencode(".jpg", restored_img)
        if not is_success:
            return jsonify({"error": "Failed to encode image"}), 500

        # التعديل الهام هنا: استخدام BytesIO بشكل صحيح
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='enhanced.jpg'
        )

    except Exception as e:
        print(f"ERROR: {str(e)}") # هذا سيطبع الخطأ الحقيقي في التيرمينال
        return jsonify({"error": str(e)}), 500

        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg'
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    # التأكد من وجود مجلد الأوزان
    if not os.path.exists('weights'):
        os.makedirs('weights')
        
    # فحص توافر CUDA قبل بدء السيرفر
    print("="*30)
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU mode (Slow)")
    print("="*30)

    app.run(debug=True, port=5000)