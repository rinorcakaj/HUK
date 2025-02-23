import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import r2_score

"""
Computer Vision Coding Challenge
"""

class CarPartsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        hood_score = row["perspective_score_hood"]
        backdoor_score = row["perspective_score_backdoor_left"]
        target = torch.tensor([hood_score, backdoor_score], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(dataloader, desc="Training"):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    """
      - MSE
      - MAE
      - R²
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)

            # MSE
            loss = criterion(preds, targets)
            running_loss += loss.item() * images.size(0)

            # MAE pro Batch
            mae_batch = torch.abs(preds - targets).mean(dim=0).mean().item()
            running_mae += mae_batch * images.size(0)

            # Sammeln für R²
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Durchschnittlicher MSE
    mse = running_loss / len(dataloader.dataset)
    # Durchschnittlicher MAE
    mae = running_mae / len(dataloader.dataset)

    # R² berechnen
    all_targets = np.concatenate(all_targets, axis=0)  # shape (N, 2)
    all_preds   = np.concatenate(all_preds,   axis=0)  # shape (N, 2)

    # Option A) ein R² für beide Outputs, gemittelt:
    r2_average = r2_score(all_targets, all_preds, multioutput='uniform_average')


    return mse, mae, r2_average

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    csv_file = "data/car_imgs_4000.csv"
    img_dir  = "data/imgs"

    # Transforms
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])

    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    dataset_full = CarPartsDataset(csv_file, img_dir, transform=train_transform)

    # 70/15/15 Split
    data_len = len(dataset_full)
    train_size = int(0.7 * data_len)
    val_size   = int(0.15 * data_len)
    test_size  = data_len - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset_full, [train_size, val_size, test_size])

    # Val/Test
    val_ds.dataset.transform = eval_transform
    test_ds.dataset.transform = eval_transform

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Modell - vortrainertes Modell für Transfer Learning
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    print("Finished loading model")

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Training")

    # Training
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train MSE: {train_loss:.4f} "
              f"| Val MSE: {val_mse:.4f} "
              f"| Val MAE: {val_mae:.4f} "
              f"| Val R2: {val_r2:.4f}")

    print("Training abgeschlossen.")

    # Test
    test_mse, test_mae, test_r2 = evaluate(model, test_loader, criterion, device)
    print(f"\nEndgültige Test-Performance:")
    print(f"MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

    torch.save(model.state_dict(), 'resnet18_carparts.pth')

if __name__ == "__main__":
    main()


"""
Weitere Optimierungen

1) Mögliche Optimierungen für das Modell:
   - Hyperparameter:
       * Learning Rate
       * Verschiedene Optimizer (Adam, SGD mit Momentum, usw.)
       * Unterschiedliche Batch-Größen (im Zusammenspiel mit Learning Rate anzupassen)
       * Early Stopping / regelmäßiges Speichern bester Checkpoints
   - Fine-Tuning:
       * Zuerst nur letzte Schicht anpassen (Feature Extraction),
         danach schrittweises Entfrieren früherer Schichten
   - Data Augmentation:
       * z.B. Random Rotation, ColorJitter, perspektivische Transformationen
   - Größere oder andere Architekturen:
       * ResNet50, EfficientNet, MobileNet (je nach Ressourcen) - wir brauchen kein ViT draufhauen
       * Oder ein kleines Modell, falls Inferenzeffizienz wichtig ist
   - Mehr Daten (Datenqualität erhöhen)
   - Cross-Validation:
       * Statt festem Split -> k-fache CV, um Varianz in den Schätzungen
         und Overfittingrisiko zu reduzieren

2) Anwendungsfälle im Versicherungsbereich:
   - Schadenserkennung & -Segmentierung über Object Detection und Semantische Segmentierung
   - Fahrzeugtyp / Bautteil Erkennung
   - Vorhersage von Reparaturkosten
"""