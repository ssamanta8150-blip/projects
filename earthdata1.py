import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from xgboost import XGBRFRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DEFINE FILE PATHS 
# ==========================================
PATH_NDVI = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\All ML input dataset\Kolkata_NDVI_30m_CSV.csv"
PATH_NDBI = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\All ML input dataset\Kolkata_NDBI_30m_CSV.csv"
PATH_AOD  = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\All ML input dataset\Kolkata_AOD_1km_CSV.csv"
PATH_CNN  = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\All ML input dataset\CNN_Wind_Output.csv"
PATH_LST  = r"D:\Skills\QGIS-Google Earth\QGIS\Project 1(UHI)\All ML input dataset\Kolkata_LST_30m_CSV.csv"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_spatial_file(filepath, value_name):
    df = pd.read_csv(filepath)
    col_mapping = {
        'latitude': 'Lat', 'latitude_y': 'Lat', 'Lat': 'Lat',
        'longitude': 'long', 'longitude_x': 'long', 'long': 'long',
        'Wind_Block_Class': 'geometry_class'
    }
    df = df.rename(columns=col_mapping)
    
    if value_name not in df.columns:
        df.rename(columns={df.columns[1]: value_name}, inplace=True)
        
    return df[['Lat', 'long', value_name]]

def calculate_effective_wind(row):
    wind_speed = row['Wind_Speed']
    wind_dir = row['Wind_Direction']
    geom = row['geometry_class']
    
    if geom == 3:
        return 0.0  
    elif geom == 0:
        return wind_speed  
    elif geom == 1: 
        if (wind_dir >= 315) or (wind_dir <= 45): 
            return 0.0
        return wind_speed
    elif geom == 2: 
        if (135 <= wind_dir <= 225): 
            return 0.0
        return wind_speed
    return wind_speed

# ==========================================
# 3. CORE PIPELINE CLASS
# ==========================================
class UHIMLPipeline:
    def __init__(self):
        self.model = XGBRFRegressor(n_estimators=150, max_depth=10, learning_rate=0.1, random_state=42)
        self.trees = {}
        self.data_cache = {}
        
    def initialize_spatial_snappers(self):
        print("Loading static spatial datasets and building KD-Trees...")
        self.data_cache['ndvi'] = load_spatial_file(PATH_NDVI, 'NDVI')
        self.data_cache['ndbi'] = load_spatial_file(PATH_NDBI, 'NDBI')
        self.data_cache['aod']  = load_spatial_file(PATH_AOD, 'AOD')
        self.data_cache['cnn']  = load_spatial_file(PATH_CNN, 'geometry_class')
        
        self.trees['ndvi'] = cKDTree(self.data_cache['ndvi'][['Lat', 'long']].values)
        self.trees['ndbi'] = cKDTree(self.data_cache['ndbi'][['Lat', 'long']].values)
        self.trees['aod']  = cKDTree(self.data_cache['aod'][['Lat', 'long']].values)
        self.trees['cnn']  = cKDTree(self.data_cache['cnn'][['Lat', 'long']].values)
        print("KD-Trees built successfully!")

    def _snap_features(self, target_coords):
        _, idx_ndvi = self.trees['ndvi'].query(target_coords)
        _, idx_ndbi = self.trees['ndbi'].query(target_coords)
        _, idx_aod  = self.trees['aod'].query(target_coords)
        _, idx_cnn  = self.trees['cnn'].query(target_coords)
        
        snapped_df = pd.DataFrame({
            'NDVI': self.data_cache['ndvi'].iloc[idx_ndvi]['NDVI'].values,
            'NDBI': self.data_cache['ndbi'].iloc[idx_ndbi]['NDBI'].values,
            'AOD': self.data_cache['aod'].iloc[idx_aod]['AOD'].values,
            'geometry_class': self.data_cache['cnn'].iloc[idx_cnn]['geometry_class'].values
        })
        return snapped_df

    def train_model(self, hist_temp=32.0, hist_hum=60.0, hist_wind_spd=5.0, hist_wind_dir=180):
        print("\n--- Phase 1: Training Model against LST ---")
        df_lst = load_spatial_file(PATH_LST, 'LST')
        coords = df_lst[['Lat', 'long']].values
        
        print("Snapping features to LST points...")
        features_df = self._snap_features(coords)
        
        features_df['Temperature'] = hist_temp
        features_df['Humidity'] = hist_hum
        features_df['Wind_Speed'] = hist_wind_spd
        features_df['Wind_Direction'] = hist_wind_dir
        
        features_df['Effective_Cooling_Wind'] = features_df.apply(calculate_effective_wind, axis=1)
        
        X_train = features_df[['Temperature', 'Humidity', 'Effective_Cooling_Wind', 'NDVI', 'NDBI', 'AOD']]
        y_train = df_lst['LST']
        
        print("Training XGBoost Boosted Random Forest...")
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, 'UHI_Trained_Model.pkl')
        print("Model Trained & Saved as 'UHI_Trained_Model.pkl'")

    def process_full_heatmap(self, live_weather_df):
        print("\n--- Phase 2: Generating Full City Heatmap from Live Stations ---")
        
        city_grid = self.data_cache['ndvi'][['Lat', 'long']].copy()
        print(f"Total points in city grid to predict: {len(city_grid)}")
        
        print("Blending live weather from stations across the entire region...")
        station_coords = live_weather_df[['Lat', 'long']].values
        city_coords = city_grid[['Lat', 'long']].values
        
        station_tree = cKDTree(station_coords)
        k_neighbors = min(3, len(live_weather_df))
        distances, indices = station_tree.query(city_coords, k=k_neighbors)
        
        distances = np.maximum(distances, 1e-6) 
        weights = 1.0 / (distances ** 2)
        weight_sums = weights.sum(axis=1)
        
        for col in ['Temperature', 'Humidity', 'Wind_Speed', 'Wind_Direction']:
            station_values = live_weather_df[col].values[indices]
            blended_values = np.sum(station_values * weights, axis=1) / weight_sums
            city_grid[col] = blended_values
            
        print("Applying NDVI, NDBI, AOD, and CNN data...")
        snapped_features = self._snap_features(city_coords)
        full_df = pd.concat([city_grid.reset_index(drop=True), snapped_features.reset_index(drop=True)], axis=1)
        
        print("Calculating aerodynamic cooling (CNN Wind Blocks) for all points...")
        full_df['Effective_Cooling_Wind'] = full_df.apply(calculate_effective_wind, axis=1)
        
        print("Predicting UHI Temperature distribution for the entire region...")
        X_live = full_df[['Temperature', 'Humidity', 'Effective_Cooling_Wind', 'NDVI', 'NDBI', 'AOD']]
        full_df['Predicted_LST'] = self.model.predict(X_live)
        
        threshold_hot = full_df['Predicted_LST'].quantile(0.70)
        threshold_cold = full_df['Predicted_LST'].quantile(0.30)
        
        conditions = [
            (full_df['Predicted_LST'] >= threshold_hot),
            (full_df['Predicted_LST'] <= threshold_cold)
        ]
        choices = ['Hot_Zone (UHI)', 'Cold_Zone (Cooling)']
        full_df['Heat_Classification'] = np.select(conditions, choices, default='Moderate')
        
        # --- LOCAL PIXEL CALCULATION ---
        z_ndbi = (full_df['NDBI'] - full_df['NDBI'].mean()) / np.maximum(full_df['NDBI'].std(), 1e-6)
        z_aod  = (full_df['AOD'] - full_df['AOD'].mean()) / np.maximum(full_df['AOD'].std(), 1e-6)
        z_ndvi = (full_df['NDVI'].mean() - full_df['NDVI']) / np.maximum(full_df['NDVI'].std(), 1e-6) 
        z_wind = (full_df['Effective_Cooling_Wind'].mean() - full_df['Effective_Cooling_Wind']) / np.maximum(full_df['Effective_Cooling_Wind'].std(), 1e-6) 
        
        heat_factors = pd.DataFrame({
            'High_Concrete (NDBI)': z_ndbi,
            'Lack_of_Vegetation (NDVI)': z_ndvi,
            'Pollution (AOD)': z_aod,
            'Wind_Blocked (CNN)': z_wind
        })
        
        full_df['Major_Heat_Cause'] = heat_factors.idxmax(axis=1)
        full_df.loc[full_df['Heat_Classification'] != 'Hot_Zone (UHI)', 'Major_Heat_Cause'] = 'Not a Hot Zone'
        
        # --- OUTPUT TO CSV ---
        output_cols = [
            'Lat', 'long', 'Predicted_LST', 'Heat_Classification', 'Major_Heat_Cause', 
            'NDBI', 'NDVI', 'AOD', 'Effective_Cooling_Wind', 'Temperature', 'Humidity'
        ]
        final_heatmap = full_df[output_cols]
        final_heatmap.to_csv('Kolkata_Entire_Region_Heatmap.csv', index=False)
        print(f"✅ SUCCESS! 'Kolkata_Entire_Region_Heatmap.csv' generated with {len(final_heatmap)} rows!")

        # ==============================================================
        # NEW: CITY-LEVEL REPORT PRINTED DIRECTLY TO CONSOLE
        # ==============================================================
        print("\n=================================================")
        print(" 🏙️  CITY-LEVEL REPORT: TODAY'S MAJOR HEAT DRIVERS")
        print("=================================================")
        
        # Filter to only look at the areas classified as "Hot_Zone (UHI)"
        hot_zones = full_df[full_df['Heat_Classification'] == 'Hot_Zone (UHI)']
        
        if not hot_zones.empty:
            # Count the occurrences of each cause and turn it into a percentage
            cause_counts = hot_zones['Major_Heat_Cause'].value_counts(normalize=True) * 100
            
            for cause, percentage in cause_counts.items():
                print(f" > {cause}: {percentage:.1f}% of Hot Zones")
        else:
            print(" > No severe Hot Zones detected under current weather conditions.")
            
        print("=================================================\n")


# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    pipeline = UHIMLPipeline()
    pipeline.initialize_spatial_snappers()
    pipeline.train_model(hist_temp=33.5, hist_hum=65.0, hist_wind_spd=4.0, hist_wind_dir=180)
    
    # YOUR 6 LIVE LOCATIONS:
    six_station_data = pd.DataFrame({
        'Lat': [22.5726, 22.5800, 22.5600, 22.5900, 22.5500, 22.5400], 
        'long': [88.3639, 88.3700, 88.3500, 88.3800, 88.3400, 88.3900], 
        'Temperature': [35.0, 35.2, 34.9, 36.1, 34.5, 35.5],
        'Humidity': [60, 62, 59, 55, 65, 61],
        'Wind_Speed': [12.0, 10.5, 14.0, 8.0, 15.0, 9.5],
        'Wind_Direction': [340, 340, 340, 180, 180, 90]  
    })
    
    pipeline.process_full_heatmap(six_station_data)