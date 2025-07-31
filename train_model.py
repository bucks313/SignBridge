import sys
print("\n".join(sys.path))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_i3d import InceptionI3d
from videotransforms import CenterCrop, RandomCrop, RandomHorizontalFlip
from torchvision.transforms import Compose
import os

# Custom Dataset Class
class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, frames_dir, class_names, transform=None):
        self.frames_dir = frames_dir
        self.class_names = class_names
        self.transform = transform
        self.samples = []
        for class_name in os.listdir(frames_dir):
            class_path = os.path.join(frames_dir, class_name)
            if os.path.isdir(class_path):
                for sample in os.listdir(class_path):
                    self.samples.append((os.path.join(class_path, sample), class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, class_name = self.samples[idx]
        frames = self.load_frames(frame_path)
        label = self.class_names.index(class_name)
        if self.transform:
            frames = self.transform(frames)
        return torch.FloatTensor(frames), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def load_frames(frame_path):
        # Assume frames are stored as individual images in a directory
        frames = []
        for img_file in sorted(os.listdir(frame_path)):
            frame = torch.load(os.path.join(frame_path, img_file))
            frames.append(frame.unsqueeze(0))  # Add temporal dimension
        return torch.cat(frames, dim=0)

# Paths
frames_dir = r"C:\Users\Bilal\Downloads\fyp_app\frames_output"
pretrained_model_path = r"C:\Users\Bilal\Downloads\fyp_app\models\rgb_imagenet.pt"
trained_model_save_path = r"C:\Users\Bilal\Downloads\fyp_app\models\trained_i3d_sign_language.pth"

# Define Gesture Classes
class_names = os.listdir(frames_dir)  # Adjust if your classes are defined elsewhere

# Data Transformations
train_transforms = Compose([RandomCrop(224), RandomHorizontalFlip()])
val_transforms = Compose([CenterCrop(224)])

# DataLoader
train_dataset = GestureDataset(frames_dir=frames_dir, class_names=class_names, transform=train_transforms)
val_dataset = GestureDataset(frames_dir=frames_dir, class_names=class_names, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Initialize I3D Model
num_classes = len(class_names)  # Number of gesture classes
model = InceptionI3d(400, in_channels=3)  # I3D with 400 pre-trained classes
model.replace_logits(num_classes)  # Replace with the number of custom classes

# Load Pretrained Weights
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if pretrained_model_path:
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print("Loaded pretrained weights.")

model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5  # Adjust based on your dataset size and resources
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 30)
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Training Loss: {running_loss / len(train_loader):.4f}")

    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save Trained Model
torch.save(model.state_dict(), trained_model_save_path)
print(f"Model training complete. Saved to {trained_model_save_path}.")
