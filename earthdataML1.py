import os
import torch
import torch.nn as nn
import torch.optim as optim
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. DATA PREPARATION (POINT-CENTERED PATCHES)
# ==========================================
WINDOW_SIZE = 33
HALF_W = WINDOW_SIZE // 2

def get_training_patches(tif_path, gpkg_path, vector_class_column='wind'):
    print("Loading Raster data...")
    with rasterio.open(tif_path) as src:
        tif_image = src.read(1)
        transform = src.transform
        crs = src.crs
        
        nodata = src.nodata
        if nodata is not None:
            tif_image[tif_image == nodata] = 0.0
            
        # Normalize raster
        tif_image = (tif_image - np.min(tif_image)) / (np.max(tif_image) - np.min(tif_image) + 1e-8)
        
        # Pad the image so we can extract windows even at the borders
        padded_img = np.pad(tif_image, pad_width=HALF_W, mode='constant', constant_values=0)

    print("Loading Vector data...")
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
        
    gdf[vector_class_column] = gdf[vector_class_column].astype(int)

    # Extract windows around each point
    X_train = []
    y_train = []
    
    print("Extracting 33x33 neighborhoods around training points...")
    for idx, row in gdf.iterrows():
        geom = row.geometry
        # If it's a polygon/buffer, get centroid. If point, just get coords.
        centroid = geom.centroid if geom.geom_type != 'Point' else geom
        lon, lat = centroid.x, centroid.y
        
        # Convert coordinates to row, col
        r, c = rasterio.transform.rowcol(transform, lon, lat)
        
        # Extract window from padded image (shift indices by HALF_W)
        pr, pc = r + HALF_W, c + HALF_W
        window = padded_img[pr-HALF_W : pr+HALF_W+1, pc-HALF_W : pc+HALF_W+1]
        
        if window.shape == (WINDOW_SIZE, WINDOW_SIZE):
            X_train.append(window)
            y_train.append(row[vector_class_column])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Extracted Patches per class: {dict(zip(unique, counts))}")
    
    return X_train, y_train, padded_img, transform, tif_image.shape

# ==========================================
# 2. DATASETS
# ==========================================
class TrainingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # Add channel dim
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class InferenceDataset(Dataset):
    """ Dynamically pulls windows for the whole map during inference """
    def __init__(self, padded_img, valid_rows, valid_cols):
        self.padded_img = padded_img
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        
    def __len__(self): return len(self.valid_rows)
    def __getitem__(self, idx):
        # Shift by HALF_W because image is padded
        r = self.valid_rows[idx] + HALF_W
        c = self.valid_cols[idx] + HALF_W
        window = self.padded_img[r-HALF_W : r+HALF_W+1, c-HALF_W : c+HALF_W+1]
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0)

# ==========================================
# 3. DEEPER CLASSIFICATION CNN
# ==========================================
class PatchClassifierCNN(nn.Module):
    def __init__(self):
        super(PatchClassifierCNN, self).__init__()
        # Slightly deeper network for 33x33 patches
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # Size: 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # Size: 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Flattens cleanly regardless of input size
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4) # 4 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 4. WORKFLOW
# ==========================================
def run_patch_pipeline(tif_path, gpkg_path, output_csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Get Data
    X_train, y_train, padded_img, transform, full_shape = get_training_patches(tif_path, gpkg_path, 'wind')
    
    # 2. Train Model
    train_loader = DataLoader(TrainingDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = PatchClassifierCNN().to(device)
    
    # Calculate class weights in case 0s dominate the 600 points
    class_counts = np.bincount(y_train, minlength=4)
    weights = len(y_train) / (4.0 * (class_counts + 1e-5))
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30 # Can increase since dataset is small
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

    # ==========================================
    #cross validation code here
    # ==========================================

    # 3. Full Map Inference (Sliding Window)
    print("\nRunning sliding-window inference across the map...")
    print("This may take several minutes depending on map size...")
    
    # Only run inference on pixels where building density actually exists
    # (Removes empty space to save compute time and CSV size)
    original_img = padded_img[HALF_W:-HALF_W, HALF_W:-HALF_W]
    valid_mask = original_img > 0
    valid_rows, valid_cols = np.where(valid_mask)
    
    print(f"Total valid pixels to classify: {len(valid_rows)}")
    
    infer_dataset = InferenceDataset(padded_img, valid_rows, valid_cols)
    # Large batch size for faster GPU processing
    infer_loader = DataLoader(infer_dataset, batch_size=1024, shuffle=False)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i, imgs in enumerate(infer_loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(infer_loader)}...")
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    all_preds = np.array(all_preds)
    
    out_unique, out_counts = np.unique(all_preds, return_counts=True)
    print(f"\nFinal Map Prediction Distribution: {dict(zip(out_unique, out_counts))}")

    # 4. Generate CSV
    print("Generating High-Resolution CSV...")
    lons, lats = rasterio.transform.xy(transform, valid_rows, valid_cols)
    
    df = pd.DataFrame({
        'system:index': np.arange(len(all_preds)),
        'Wind_Block_Class': all_preds,
        'latitude': lats,
        'longitude': lons
    })
    
    df.to_csv(output_csv_path, index=False)
    print(f"Done! Format matched. CSV saved to: {output_csv_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    TIF_PATH = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\Building Density.tif"
    GPKG_PATH = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\Wind1.gpkg"
    OUTPUT_CSV = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\CNN_Wind_Output.csv"
    
    run_patch_pipeline(TIF_PATH, GPKG_PATH, OUTPUT_CSV)