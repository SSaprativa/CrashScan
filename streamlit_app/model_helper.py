import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect CPU/GPU


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers except layer4 and fc layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Move image tensor to CPU/GPU

    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet().to(device)  # Move model to CPU/GPU
        trained_model.load_state_dict(torch.load(
            "/Users/saprativasarkar/Desktop/ML/2_Project/3_Damage_Detection/model/saved_model.pth",
            map_location=device  # Ensures compatibility with CPU
        ))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
