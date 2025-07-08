import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext

# === Etiquetas de las clases de imagen ===
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# === Configuración ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modelo CNN ===
class CNNFashion(nn.Module):
    def __init__(self):
        super(CNNFashion, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

    def extract_features(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers[0](x)  # Flatten
        x = self.fc_layers[1](x)  # Linear
        x = self.fc_layers[2](x)  # ReLU
        x = self.fc_layers[3](x)  # Dropout
        return x  # Vector de 128 dimensiones

# === Modelo Transformer para texto ===
class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x).permute(1, 0, 2)
        out = self.transformer_encoder(emb).mean(dim=0)
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out)).view(-1)

# === Hiperparámetros texto ===
EMBED_DIM = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
HIDDEN_DIM = 256
DROPOUT = 0.2
MAX_LEN = 200

# === Tokenizador y vocabulario ===
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(torchtext.datasets.IMDB(split='train')),
    specials=["<pad>", "<unk>"]
)
vocab.set_default_index(vocab["<unk>"])

# === Cargar modelos entrenados ===
model = CNNFashion().to(device)
model.load_state_dict(torch.load("../modelos/cnn_fashion/mejor_modelo.pth", map_location=device))
model.eval()

model_texto = SentimentTransformer(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_ENCODER_LAYERS,
    dropout=DROPOUT
).to(device)
model_texto.load_state_dict(torch.load("../modelos/transformer_imdb/mejor_modelo_imdb.pth", map_location=device))
model_texto.eval()

# === Transformación de imagen ===
def binarize(img, threshold=0.5):
    return img.point(lambda p: 255 if p > threshold * 255 else 0)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.Lambda(lambda img: binarize(img, threshold=0.5)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Preprocesamiento de texto ===
def preprocess_text(texto):
    tokens = tokenizer(texto)
    token_ids = vocab(tokens)[:MAX_LEN]
    padded = token_ids + [vocab["<pad>"]] * (MAX_LEN - len(token_ids))
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)

# === Función principal de la demo ===
def fusion_demo(imagen, texto):
    if imagen is None or texto.strip() == "":
        return "Falta imagen o texto."

    # === Imagen ===
    imagen_tensor = transform(imagen).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(imagen_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        vector_img = model.extract_features(imagen_tensor).cpu().numpy().flatten()

    # === Texto ===
    input_tensor = preprocess_text(texto)
    with torch.no_grad():
        sentimiento = model_texto(input_tensor).item()
    etiqueta_sentimiento = "Positiva" if sentimiento <= 0.5 else "Negativa"

    # === Resultado ===
    nombre_clase = classes[predicted_class]
    resultado = (
        f"Prenda detectada: {nombre_clase}\n"
        f"Confianza: {confidence:.2%}\n"
        f"Sentimiento del texto: {etiqueta_sentimiento} ({sentimiento:.2%})\n"
        f"Vector imagen (128 dim): {vector_img[:5]}...\n"
        f"Texto: {texto[:30]}..."
    )
    return resultado

# === Interfaz Gradio ===
demo = gr.Interface(
    fn=fusion_demo,
    inputs=[
        gr.Image(type="pil", label="Imagen de producto/prenda"),
        gr.Textbox(lines=3, label="Reseña del usuario")
    ],
    outputs=gr.Text(label="Resultado"),
    title="Demo Multimodal (Imagen + Texto)",
    description="Este prototipo detecta la prenda en la imagen y analiza el sentimiento del texto ingresado."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
