import cv2
import torch
from pytorch_i3d import InceptionI3d
import numpy as np
from gtts import gTTS
import playsound

# Load Trained Model
model = InceptionI3d(num_classes=400, in_channels=3)
model.replace_logits(num_classes=5)  # Replace with your actual number of gestures
model.load_state_dict(torch.load('trained_i3d_model.pth'))
model = model.cuda()
model.eval()

# Gesture Mapping
gesture_to_text = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No", 4: "Stop"}

# Preprocess Frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
    frame = frame / 255.0  # Normalize
    return torch.tensor(frame).unsqueeze(0).float().cuda()

# Real-Time Video Inference
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    input_tensor = preprocess_frame(frame)
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Predicted Gesture: {gesture_to_text[predicted_class]}")

    # Text-to-Speech
    tts = gTTS(gesture_to_text[predicted_class], lang='ur')
    tts.save("output.mp3")
    playsound.playsound("output.mp3")
cap.release()
cv2.destroyAllWindows()
