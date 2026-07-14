#First we need to train our CNN model using Faster-CNN to identify wind obstructing Geometries
#These include patches which disallow central wind not to dislocate
#The ouput we will get is a csv file which will include index-score-lat-long
import os
import torch
import torch.nn as nn
import torch.optim as optim
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. DATA PREPARATION (FIXED BACKGROUND = -1)
# ==========================================
def prepare_data(tif_path, gpkg_path, vector_class_column='wind'):
    print("Loading Raster data...")
    with rasterio.open(tif_path) as src:
        tif_image = src.read(1)
        transform = src.transform
        crs = src.crs
        shape = src.shape
        
        # Replace NoData with 0 for the input features
        nodata = src.nodata
        if nodata is not None:
            tif_image[tif_image == nodata] = 0.0
            
        # Normalize raster data (0 to 1) for the CNN
        tif_image = (tif_image - np.min(tif_image)) / (np.max(tif_image) - np.min(tif_image) + 1e-8)

    print("Loading Vector data...")
    gdf = gpd.read_file(gpkg_path)
    
    # Ensure CRS matches
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
        
    # BUFFER POINTS: Turn your single points into 30-unit circles so the CNN can see them
    # (If your CRS is in degrees instead of meters, change 30 to something like 0.0003)
    gdf['geometry'] = gdf.geometry.buffer(30)
    
    # Ensure class column is integer
    gdf[vector_class_column] = gdf[vector_class_column].astype(int)

    print("Rasterizing vector data...")
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[vector_class_column]))
    
    # CRITICAL FIX: Fill empty space with -1, NOT 0. 
    # 0 is a valid class in your data!
    label_image = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=-1, 
        dtype='int64'
    )
    
    # Print out how many usable training pixels we actually have
    unique, counts = np.unique(label_image, return_counts=True)
    print(f"Training Pixels per class: {dict(zip(unique, counts))}")
    
    if tif_image.shape == label_image.shape:
        print("matched dimensions")
    else:
        raise ValueError("Dimensions do not match!")
        
    return tif_image, label_image, transform, shape

# ==========================================
# 2. DATASET CREATION (FILTERING OUT EMPTY PATCHES)
# ==========================================
class SpatialPatchDataset(Dataset):
    def __init__(self, image, labels, patch_size=32):
        self.image = image
        self.labels = labels
        self.patch_size = patch_size
        self.patches = self._generate_patches()

    def _generate_patches(self):
        patches = []
        h, w = self.image.shape
        for i in range(0, h - self.patch_size, self.patch_size):
            for j in range(0, w - self.patch_size, self.patch_size):
                patch_labels = self.labels[i:i+self.patch_size, j:j+self.patch_size]
                # ONLY train on patches that contain at least one of your actual points (not just -1)
                if np.any(patch_labels != -1):
                    patches.append((i, j))
        print(f"Created {len(patches)} valid training patches.")
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]
        img_patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
        label_patch = self.labels[y:y+self.patch_size, x:x+self.patch_size]
        
        img_tensor = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_patch, dtype=torch.long)
        return img_tensor, label_tensor, y, x

# ==========================================
# 3. CNN ARCHITECTURE
# ==========================================
class SimpleSegmentationCNN(nn.Module):
    def __init__(self):
        super(SimpleSegmentationCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1) # 4 output classes: 0, 1, 2, 3
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ==========================================
# 4. TRAINING & INFERENCE WORKFLOW
# ==========================================
def train_and_predict(tif_path, gpkg_path, output_csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    image, labels, transform, full_shape = prepare_data(tif_path, gpkg_path, vector_class_column='wind')
    
    # 2. Create Dataset
    patch_size = 32
    dataset = SpatialPatchDataset(image, labels, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 3. Initialize Model
    model = SimpleSegmentationCNN().to(device)
    
    # CRITICAL FIX: Tell the loss function to completely IGNORE the -1 background
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 15 # Increased slightly since we are training on much cleaner, focused data now
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, lbls, _, _ in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

    # ==========================================
    #cross validation code here
    # ==========================================

    # 5. Full Image Inference (Applying the trained model to the whole map)
    print("Running inference to generate high-resolution map...")
    model.eval()
    predicted_full_map = np.zeros(full_shape, dtype=np.uint8)
    
    # We must create a new dataset to predict over the ENTIRE map, not just the training patches
    class InferenceDataset(Dataset):
        def __init__(self, image, patch_size=32):
            self.image = image
            self.patch_size = patch_size
            self.patches = [(i, j) for i in range(0, image.shape[0], patch_size) 
                                   for j in range(0, image.shape[1], patch_size)]
        def __len__(self): return len(self.patches)
        def __getitem__(self, idx):
            y, x = self.patches[idx]
            img_patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
            # Pad if patch is at the edge and smaller than patch_size
            if img_patch.shape != (self.patch_size, self.patch_size):
                pad_y = self.patch_size - img_patch.shape[0]
                pad_x = self.patch_size - img_patch.shape[1]
                img_patch = np.pad(img_patch, ((0, pad_y), (0, pad_x)), mode='constant')
            return torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0), y, x

    infer_dataset = InferenceDataset(image, patch_size=patch_size)
    infer_loader = DataLoader(infer_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for imgs, y_batch, x_batch in infer_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            preds_cpu = preds.cpu().numpy()
            for i in range(len(y_batch)):
                y, x = y_batch[i].item(), x_batch[i].item()
                # Ensure we don't write out of bounds for edge patches
                h_end = min(y + patch_size, full_shape[0])
                w_end = min(x + patch_size, full_shape[1])
                predicted_full_map[y:h_end, x:w_end] = preds_cpu[i, :h_end-y, :w_end-x]

    # Verify predictions are varied
    out_unique, out_counts = np.unique(predicted_full_map, return_counts=True)
    print(f"Final Map Prediction Distribution: {dict(zip(out_unique, out_counts))}")

    # 6. Convert to CSV (Ready for Step 2)
    print("Generating High-Resolution CSV...")
    
    # Filter out empty areas of the original map to save space
    valid_mask = image > 0 
    rows, cols = np.indices(full_shape)
    
    rows_valid = rows[valid_mask]
    cols_valid = cols[valid_mask]
    classes_valid = predicted_full_map[valid_mask]
    
    lons, lats = rasterio.transform.xy(transform, rows_valid, cols_valid)
    
    df = pd.DataFrame({
        'system:index': np.arange(len(classes_valid)),
        'Wind_Block_Class': classes_valid,
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
    
    train_and_predict(TIF_PATH, GPKG_PATH, OUTPUT_CSV)