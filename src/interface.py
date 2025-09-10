import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
MODEL_PATH = "results/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (change if needed)
class_labels = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Preprocess function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_class, confidence

# GUI setup
root = tk.Tk()
root.title("Alzheimer's MRI Prediction")
root.geometry("600x500")

label = tk.Label(root, text="Upload an MRI Image", font=("Arial", 16))
label.pack(pady=20)

canvas = tk.Label(root)
canvas.pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        canvas.config(image=img_tk)
        canvas.image = img_tk

        predicted_class, confidence = predict_image(file_path)
        result_label.config(
            text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
        )

upload_btn = tk.Button(root, text="Upload MRI", command=upload_image, font=("Arial", 12))
upload_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
exit_btn.pack(pady=10)

root.mainloop()
