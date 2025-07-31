# SignBridge
🤟 Sign Language Recognition System
A real-time Pakistan Sign Language (PSL) recognition system using deep learning models (C3D and I3D) to translate sign language gestures into text and speech. This project implements multiple neural network architectures for video-based sign language classification.
�� Features
Real-time Sign Language Recognition: Live video processing with webcam input
Multiple Model Architectures:
C3D (3D Convolutional Neural Networks)
I3D (Inception 3D Networks)
LSTM
ResNet-based models
Text-to-Speech Output: Converts recognized signs to spoken words
Video File Processing: Upload and analyze pre-recorded sign language videos
GUI Interface: User-friendly desktop application for video upload and prediction
Comprehensive Dataset: Trained on extensive ASL video dataset with 300+ sign classes
��️ Technical Stack
Deep Learning: PyTorch, TensorFlow
Computer Vision: OpenCV, Albumentations
Models: C3D, I3D, ResNet50
Audio: gTTS (Google Text-to-Speech)
GUI: Tkinter
Data Processing: NumPy, Pandas
📁 Project Structure
train_c3d.py - C3D model training pipeline
predict.py - Video prediction with GUI
real_time_translation.py - Live sign language recognition
c3d_model.py - Custom C3D architecture implementation
dataset/ - ASL video dataset (300+ classes)
Pre-trained models for immediate use
🎯 Use Cases
Accessibility: Help deaf and hard-of-hearing individuals communicate
Education: PSL learning and teaching tools
Healthcare: Medical sign language interpretation
Research: Computer vision and NLP research
�� Performance
Trained on 2000 PSL word classes
High accuracy on diverse sign language gestures
Robust to lighting and background variations
�� Note
I am still working on the realtime translation there are some issues in realtime work because there is not enough dataset to train our model.I am are trying to make the dataset myself will keep you guys updated.
if you have any questions you can hit me up on my email.ok Bye 

