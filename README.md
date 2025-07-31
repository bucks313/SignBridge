✋🤟 SignBridge
Real-time Pakistan Sign Language (PSL) Recognition System

A real-time sign language translation system using deep learning (C3D, I3D, and ResNet-based models) to convert PSL gestures into text and speech.

🚀 Features
🔴 Real-Time Recognition: Live sign detection through webcam

🧠 Multiple Deep Learning Architectures:

C3D (3D Convolutional Neural Networks)

I3D (Inception 3D Networks)

ResNet-based models

LSTM for sequence modeling

🔊 Text-to-Speech Output: Converts recognized signs into audio using gTTS

🎥 Video File Support: Upload and analyze pre-recorded sign language videos

🖥️ GUI Interface: Built using Tkinter for user-friendly interaction

🎯 Comprehensive Dataset: Trained on ASL dataset with 300+ sign classes

🛠️ Technical Stack
Deep Learning: PyTorch, TensorFlow

Computer Vision: OpenCV, Albumentations

Models: C3D, I3D, ResNet50

Audio Processing: gTTS (Google Text-to-Speech)

GUI: Tkinter

Data Handling: NumPy, Pandas

📁 Project Structure
graphql
Copy
Edit
├── train_c3d.py              # C3D model training script
├── predict.py                # GUI-based video prediction
├── real_time_translation.py # Webcam-based real-time translation
├── c3d_model.py              # Custom C3D architecture
├── dataset/                  # ASL video dataset (300+ classes)
└── pretrained_models/        # Pretrained weights
🌍 Use Cases
♿ Accessibility: Bridge communication for deaf and hard-of-hearing individuals

📚 Education: Aid PSL learning in classrooms and homes

🏥 Healthcare: Medical sign interpretation tools

🔬 Research: CV and NLP-based sign language understanding

📊 Performance
✅ Trained on 2000+ PSL word classes

✅ High accuracy across various gestures

✅ Works well under different lighting and backgrounds

🧪 Current Work & Updates
⚠️ Real-time translation is still under development.
We are currently working on building a custom PSL dataset as existing ones are limited.
Stay tuned for updates — dataset contributions are welcome!

📩 Contact
Feel free to reach out if you have any questions or suggestions:
📧 bilalazhar2019@gmail.com

