from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Create the main window
Window = Tk()
Window.geometry('330x350')
Window.title('Image Classifier GUI')

# Load the trained model
model = tf.keras.models.load_model('models/imageclassifier.h5')

# Function to load and display the image
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        img_path_entry.delete(0, END)
        img_path_entry.insert(0, file_path)
        
        img = Image.open(file_path)
        img_width, img_height = img.size
        label_width = 256
        label_height = 256

        scale_w = label_width / img_width
        scale_h = label_height / img_height
        scale = max(scale_w, scale_h)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

        left = (new_width - label_width) / 2
        top = (new_height - label_height) / 2
        right = (new_width + label_width) / 2
        bottom = (new_height + label_height) / 2

        img = img.crop((left, top, right, bottom))
        
        img_display = ImageTk.PhotoImage(img)
        img_label.config(image=img_display)
        img_label.image = img_display

# Function to analyze the image using the model
def analyze_image():
    file_path = img_path_entry.get()
    if not file_path:
        messagebox.showerror("Error", "Please load an image first.")
        return

    img = Image.open(file_path)
    img = img.resize((256, 256), Image.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    analyze_result_entry.delete(0, END)
    match predicted_class:
        case 0: analyze_result_entry.insert(0, f'Predicted class: bicycle')
        case 1: analyze_result_entry.insert(0, f'Predicted class: bus')
        case 2: analyze_result_entry.insert(0, f'Predicted class: car')
        case 3: analyze_result_entry.insert(0, f'Predicted class: human')
        case 4: analyze_result_entry.insert(0, f'Predicted class: traffic light')
        case 5: analyze_result_entry.insert(0, f'Predicted class: traffic sign')
    

# Labels
img_path_label = Label(Window, text="Image path")
analyze_label = Label(Window, text="Analyze")

# Textboxes
img_path_entry = Entry(Window, width=30)
analyze_result_entry = Entry(Window, width=30)

# Buttons
load_img_button = Button(Window, text="Load img", command=load_image)
analyze_button = Button(Window, text="Analyze", command=analyze_image)

# Image display placeholder
img_label = Label(Window, width=256, height=256, relief="solid")
img_label.place(x=38, y=10, width=256, height=256)

# Placement of other widgets
img_path_label.place(x=10, y=275)
img_path_entry.place(x=80, y=275)
load_img_button.place(x=250, y=273)

analyze_label.place(x=10, y=300)
analyze_result_entry.place(x=80, y=300)
analyze_button.place(x=250, y=300)

# Run the main loop
Window.mainloop()
