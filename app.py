import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

model_path = 'final_model_finetuned.pth'

# Загрузка модели
def load_model(model_path, num_classes=15):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Предобработка загруженного изображения
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Классификация изображения
def classify_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

class_names = [
    "Blazer", "Coat", "Dress", "Hoodie", "Jacket",
    "Jaket_Denim", "Jacket_Sports", "Jeans", "Pants", "Polo",
    "Shirt", "Shorts", "Skirt", "T-shirt", "Sweater"
]

st.title("Классификация предметов одежды")
st.write(f'''Загрузите изображение, и модель определит, что это за предмет одежды.
Обратите внимание, что модель может распознать пока что только следующие предметы одежды:
- пиджак
- пальто
- платье
- худи
- куртка
- джинсовая куртка
- спортивная куртка
- джинсы
- штаны
- поло
- рубашка
- шорты
- юбка
- футболка
- свитер''')

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Отображение загруженного изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    image_tensor = preprocess_image(image)

    model = load_model(model_path)

    # Классификация
    predicted_class_idx = classify_image(model, image_tensor)
    predicted_class = class_names[predicted_class_idx]

    st.write(f"Модель считает, что это:")
    st.title(f"**{predicted_class}**")