
# Face Recognition System with Liveness detection

A simple, efficient, and user-friendly face recognition system built with Python, Flask, and InsightFace. This project enables you to train a face recognition model and recognize faces in real-time or static images.

---

## Features

- **Train Face Recognition Models**: Train the model with a clear image of a person's face.
- **Real-time Face Recognition**: Use a webcam to detect faces dynamically.
- **Static Image Recognition**: Upload an image for face recognition.
- **Lightweight and Scalable**: Uses ONNX runtime for performance.
- **Easy to Use Interface**: Intuitive web interface for training and recognition.

---

## Prerequisites

- Python 3.8 or later
- A working webcam (if using real-time recognition)
- Internet browser

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/jagdishlamba/face-recognition-system.git
   cd face-recognition-system
   
   ```

2. **Install Dependencies**  
   Run the following command to install the required libraries:
   ```bash
   pip install Flask insightface opencv-python-headless numpy onnxruntime dlib 

   or 

   conda env create -f environment.yml
   ```

---

## Usage

### 1. Start the Application

Run the application by executing:
```bash
python main.py
```

Open your browser and navigate to:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 2. Train the Model

1. Go to the **Train Model** section on the webpage.
2. Enter the person's name in the "Name" field.
3. Upload a clear image of the person's face using the "Upload Image" button.
4. Click "Train Model."
5. Once the face is detected, the system will save the data and display a success message.

---

### 3. Recognize Faces

1. Go to the **Recognize Faces** section on the webpage.
2. Select the input source:
   - **Webcam**: Detect faces in real-time using your webcam.
   - **Image File**: Upload a static image for recognition.
3. If using an image file, upload the image.
4. Click **Recognize Faces**.
5. The system will process the input and display the recognized faces.

---

## Notes

- **Uploaded Images**: All uploaded images are stored in the `uploads/` folder.
- **Trained Model**: The trained model is saved as `models/insightface_model.pkl`.
- **Webcam Recognition**: Press `q` to quit the webcam recognition mode.

---

## Troubleshooting

- **No Face Detected**: Ensure the image is clear and the face is prominently visible.
- **Webcam Issues**: Check permissions and ensure it is not being used by another application.
- **Missing Dependencies**: Run `pip install -r requirements.txt` if dependencies are missing.

---

## File Structure

```
face-recognition-system/
│
├── app.py                      # Main application file
├── templates/                  # HTML templates for the web interface
├── static/                     # Static assets (CSS, JS, images)
├── uploads/                    # Folder to store uploaded images
├── env.yml                     # List of dependencies
└── README.md                   # Project documentation
```

---

## Future Enhancements

- Support for multi-face recognition in a single image.
- Integration with cloud storage for large-scale deployments.
- Real-time analytics dashboard for monitoring recognition stats.

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Your Name**  
GitHub: [@jagdishlamba](https://github.com/jagdishlamba)  

For inquiries or suggestions, please contact: your_email@example.com  

---

Happy coding! 😊
