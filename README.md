# Machine Learning Attribution of Urban Heat Islands in Kolkata Metropolitan Region

## Overview

This project presents a machine learning framework for identifying, predicting, and explaining the drivers of the **Urban Heat Island (UHI)** phenomenon in the Kolkata Metropolitan Region.

Unlike conventional studies that rely solely on **Land Surface Temperature (LST)**, this work combines satellite-derived environmental indices, building density, atmospheric pollution, wind obstruction, and real-time weather observations to determine **why a location becomes hot**, rather than simply identifying where heat exists.

The framework integrates **Google Earth Engine**, **QGIS**, **PyTorch**, **Random Forest**, and **XGBoost** to generate interpretable heat maps and identify the dominant environmental factor responsible for localized urban heating.

---

# Objectives

* Predict Urban Heat Island intensity across Kolkata.
* Identify the dominant environmental driver responsible for heating at every location.
* Incorporate directional wind effects through deep learning.
* Produce GIS-ready outputs for urban planning and climate mitigation.
* Support real-time prediction using live weather observations.

---

# Features

* Satellite data extraction using Google Earth Engine
* Automatic generation of NDVI, NDBI, AOD and LST datasets
* Building footprint extraction using Google Open Buildings
* CNN-based wind obstruction classification
* Random Forest Regression boosted using XGBoost
* Spatial interpolation of live weather stations
* Feature importance analysis
* Generation of city-wide Urban Heat maps
* Attribution of primary heat-causing factor
* GIS compatible outputs (GeoTIFF, CSV, SHP)

---

# Project Workflow

```
Google Earth Engine
        │
        ▼
Satellite Data
(NDVI, NDBI, AOD, LST)
        │
        ▼
Building Density Extraction
        │
        ▼
Wind Geometry Labels
        │
        ▼
CNN Training
        │
        ▼
Wind Obstruction Classification
        │
        ▼
Random Forest + XGBoost
        │
        ▼
LST Prediction
        │
        ▼
Urban Heat Attribution
        │
        ▼
Heat Maps + Cause Maps
```

---

# Environmental Variables

The model uses the following predictors:

| Variable               | Description                       |
| ---------------------- | --------------------------------- |
| NDVI                   | Vegetation Index                  |
| NDBI                   | Built-up Density Index            |
| AOD                    | Aerosol Optical Depth             |
| Building Density       | Urban morphology                  |
| Effective Cooling Wind | Wind after obstruction correction |
| Temperature            | Live weather station data         |
| Humidity               | Live weather station data         |

---

# Machine Learning Pipeline

## 1. Data Acquisition

Satellite datasets are obtained using **Google Earth Engine**.

Extracted datasets include

* NDVI
* NDBI
* Aerosol Optical Depth
* Land Surface Temperature
* Building Footprints

Outputs are exported as:

* GeoTIFF
* CSV
* Shapefiles

---

## 2. Wind Obstruction Modelling

Building density alone cannot explain urban heat.

A lightweight Convolutional Neural Network is trained to classify regions into four wind obstruction categories:

* No obstruction
* North wind blocked
* South wind blocked
* Fully blocked

The CNN produces a spatial wind obstruction layer used during temperature prediction.

---

## 3. Heat Prediction

A Random Forest Regressor enhanced using **XGBoost** is trained against observed Land Surface Temperature.

Training features include:

* Temperature
* Humidity
* Effective Cooling Wind
* NDVI
* NDBI
* Aerosol Optical Depth

The trained model predicts city-wide heat distribution.

---

## 4. Heat Attribution

For every predicted hot location, the framework determines the dominant reason for heating.

Possible causes include

* High concrete density (NDBI)
* Lack of vegetation (Low NDVI)
* Pollution (High AOD)
* Wind obstruction

This provides an explainable rather than purely predictive model.

---

# Technologies Used

### GIS

* Google Earth Engine
* QGIS

### Machine Learning

* PyTorch
* XGBoost
* Random Forest

### Python Libraries

* NumPy
* Pandas
* Rasterio
* GeoPandas
* SciPy
* Joblib

---

# Repository Structure

```
Project
│
├── GEE/
│   ├── NDVI Extraction
│   ├── NDBI Extraction
│   ├── AOD Extraction
│   ├── LST Extraction
│   └── Building Extraction
│
├── CNN/
│   └── Wind Geometry Classification
│
├── ML/
│   ├── Random Forest
│   ├── XGBoost
│   └── Prediction Pipeline
│
├── Outputs/
│   ├── Heat Maps
│   ├── GeoTIFF
│   ├── CSV
│   └── Shapefiles
│
└── README.md
```

---

# Output Products

The framework generates:

* Predicted Urban Heat Map
* Major Heat Cause Map
* Pollution Dominated Regions
* Concrete Dominated Regions
* Wind Obstruction Regions
* Cooling Zones
* GIS-ready CSV files
* GeoTIFF raster outputs

---

# Potential Applications

* Urban Planning
* Climate Change Studies
* Smart City Planning
* Heat Action Plans
* Green Infrastructure Planning
* Pollution Mitigation
* Traffic and Ventilation Corridor Design
* Environmental Impact Assessment

---

# Future Improvements

* Incorporate Sentinel-2 imagery
* Integrate Digital Elevation Models
* Add temporal forecasting using LSTM models
* Couple with Computational Fluid Dynamics for urban airflow simulation
* Deploy as a web-based decision support system
* Extend to other metropolitan regions

---

# Results

The framework demonstrates that Urban Heat Islands in Kolkata are not driven by a single factor.

Different regions exhibit different dominant mechanisms including:

* Dense concrete infrastructure
* Reduced vegetation
* Atmospheric pollution
* Wind obstruction caused by urban geometry

This allows targeted mitigation strategies instead of applying uniform cooling measures across the city.

