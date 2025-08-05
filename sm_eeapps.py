import streamlit as st
import geemap.foliumap as geemap
import ee
import numpy as np
import joblib
import torch
import asyncio
import sys
import pandas as pd
from streamlit_folium import st_folium
import gc
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math

gc.collect()
torch.cuda.empty_cache()  # even on CPU, this helps clean up cache

# Fix asyncio event loop issue on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize Earth Engine
ee.Initialize()
st.set_page_config(layout="wide")
st.title("🛰️ Global 10-m Surface (Top 5cm) Soil Moisture Prediction")

# Shrink top padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Set default values
if "latitude" not in st.session_state:
    st.session_state["latitude"] = 38.53259
if "longitude" not in st.session_state:
    
    st.session_state["longitude"] = -121.799176

def create_map(lat, lon, zoom=13):
    m = geemap.Map(center=[lat, lon], zoom=zoom)
    m.add_basemap("SATELLITE")
    m.add_child(geemap.folium.LatLngPopup())
    return m

if "map" not in st.session_state:
    st.session_state.map = create_map(st.session_state.latitude, st.session_state.longitude)

def check_available_dates(start_date, end_date, roi):
    """Check available dates for Sentinel-1 that have both Sentinel-2 and Landsat within ±3 days"""
    # Convert string dates to datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Get all Sentinel-1 dates in the period
    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    
    s1_dates = [datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d') 
               for x in s1_collection.aggregate_array('system:time_start').getInfo()]
    
    # Use a set to automatically avoid duplicates
    valid_s1_dates = set()
    
    for s1_date in s1_dates:
        s1_dt = datetime.strptime(s1_date, "%Y-%m-%d")
        window_start = (s1_dt - timedelta(days=3)).strftime('%Y-%m-%d')
        window_end = (s1_dt + timedelta(days=3)).strftime('%Y-%m-%d')
        
        # Check Sentinel-2
        s2_collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filterDate(window_start, window_end) \
            .filterBounds(roi) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        s2_available = s2_collection.size().getInfo() > 0
        
        # Check Landsat
        landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA") \
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA")) \
            .filterDate(window_start, window_end) \
            .filterBounds(roi) \
            .filter(ee.Filter.lt('CLOUD_COVER', 5))
        landsat_available = landsat_collection.size().getInfo() > 0
        
        # Only keep if both Sentinel-2 and Landsat are available
        if s2_available and landsat_available:
            valid_s1_dates.add(s1_date)
    
    # Convert set back to sorted list
    return sorted(list(valid_s1_dates), reverse=True)

# Add this to your session state initialization
if "valid_dates" not in st.session_state:
    st.session_state.valid_dates = []

# --- Display the map ---
col1, col2 = st.columns([1, 3])

with col2:
    map_data = st_folium(st.session_state.map, height=600, width=800, key="main_map")
    
# --- USER INPUT FORM ---
with col1:
    st.header("User Inputs")

    start = datetime(2016, 1, 1)
    end = datetime.today().replace(day=1) - relativedelta(months=1)
    date_options = pd.date_range(start=start, end=end, freq='MS').strftime('%Y-%m-%d').tolist()

    start_date = st.selectbox("Select Start Date:", date_options[::-1], index=date_options[::-1].index('2023-07-01'))
    
    # Calculate end_date options based on selected start_date
    selected_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Generate full list of end_date options
    end_start = datetime(2016, 2, 1)
    end_end = datetime.today().replace(day=1)
    end_date_options = pd.date_range(start=end_start, end=end_end, freq='MS').strftime('%Y-%m-%d').tolist()

    # Automatically select the one that is 1 month after start_date
    next_month = (selected_start_dt + relativedelta(months=1)).strftime("%Y-%m-%d")
    if next_month in end_date_options:
        end_index = end_date_options.index(next_month)
    else:
        end_index = 0  # fallback if not in list

    end_date = st.selectbox("Select End Date:", end_date_options[::-1], index=len(end_date_options) - 1 - end_index)
    
    lat = st.number_input("Enter Latitude:", value=st.session_state.latitude, format="%.6f")
    lon = st.number_input("Enter Longitude:", value=st.session_state.longitude, format="%.6f")
    
    # Create ROI for date checking (smaller than prediction ROI)
    date_check_roi = ee.Geometry.Point([lon, lat]).buffer(1000)
    
    if st.button("Check Available Dates"):
        st.session_state.valid_dates = check_available_dates(start_date, end_date, date_check_roi)
        
        if not st.session_state.valid_dates:
            st.warning("No valid dates found with all required imagery")
        else:
            st.success(f"Found {len(st.session_state.valid_dates)} valid dates")

    # Only show the selectbox if we have valid dates
    if st.session_state.valid_dates:
        selected_valid_date = st.selectbox(
            "Select from available dates:", 
            st.session_state.valid_dates,
            index=0,
            help="These dates have Sentinel-1 with both Sentinel-2 and Landsat within 3 days"
        )
        
    if map_data and map_data.get("last_clicked"):
        # Store the clicked coordinates immediately
        st.session_state.temp_latitude = map_data["last_clicked"]["lat"]
        st.session_state.temp_longitude = map_data["last_clicked"]["lng"]
        st.caption(f"📍 Clicked: {st.session_state.temp_latitude:.6f}, {st.session_state.temp_longitude:.6f}")
        
        if st.button("Apply"):
            # Apply the coordinates immediately when button is clicked
            st.session_state.latitude = st.session_state.temp_latitude
            st.session_state.longitude = st.session_state.temp_longitude
            # Force a rerun to update the map
            st.rerun()

if not map_data.get("last_clicked"):
    st.session_state.latitude = lat
    st.session_state.longitude = lon

# === RUN PREDICTION ===       
if st.button("Run Prediction"):
    if st.session_state.get("valid_dates") and 'selected_valid_date' in locals():
        st.info(f"⚙️ Running prediction for lat: {lat:.6f}, lon: {lon:.6f}, date: {selected_valid_date}")
        st.session_state.run_prediction = True
    else:
        st.warning("Please check available dates and select one first")
        st.session_state.run_prediction = False

    roi = ee.Geometry.Rectangle([
        st.session_state.longitude - 0.008, st.session_state.latitude - 0.008,
        st.session_state.longitude + 0.008, st.session_state.latitude + 0.008
    ])
    
    st.session_state["roi"] = roi
    
    # Generate latitude and longitude bands
    lon_lat = ee.Image.pixelLonLat().select(['longitude', 'latitude'])
    
    def resample_to_10m(image):
        """Resample an image to 10m resolution."""
        return image.resample('bilinear').reproject(crs='EPSG:4326', scale=10)

    # Sentinel-1 Processing
    if st.session_state.get("valid_dates") and 'selected_valid_date' in locals():
        # Use single selected date (±1 day to ensure we get the image)
        s1_date = datetime.strptime(selected_valid_date, "%Y-%m-%d")
        s1_start = (s1_date - timedelta(days=1)).strftime('%Y-%m-%d')
        s1_end = (s1_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        sentinel_1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
            .filterDate(s1_start, s1_end)\
            .filterBounds(roi)\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
            .first()\
            .select(['VV', 'VH', 'angle'])
    else:
        st.error("No available data in this time range")
        st.stop()  # This will stop execution of the rest of the app
    
    if isinstance(sentinel_1, ee.ImageCollection):
        sentinel_1 = sentinel_1.first()
    
    # Sentinel-2 Processing
    sentinel_2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))\
        .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'])\
        .map(resample_to_10m)

    # Add NDVI and NDMI to Sentinel-2
    def add_s2_indices(image):
        band_names = image.bandNames().map(lambda name: ee.String('sentinel2_').cat(name))
        image = image.rename(band_names)
        ndvi = image.normalizedDifference(['sentinel2_B8', 'sentinel2_B4']).rename('sentinel2_NDVI')
        ndmi = image.normalizedDifference(['sentinel2_B8', 'sentinel2_B11']).rename('sentinel2_NDMI')
        return image.addBands(ndvi).addBands(ndmi)

    sentinel_2 = sentinel_2.map(add_s2_indices)         
                                                                                            
    # Landsat Processing
    landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")\
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA"))\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .filter(ee.Filter.lt('CLOUD_COVER', 5))\
        .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'])\
        .map(resample_to_10m)

    # Add prefix and NDVI/NDMI to Landsat-8
    def add_ls_indices(image):
        band_names = image.bandNames().map(lambda name: ee.String('landsat_').cat(name))
        image = image.rename(band_names)
        ndvi = image.normalizedDifference(['landsat_B5', 'landsat_B4']).rename('landsat_NDVI')
        ndmi = image.normalizedDifference(['landsat_B5', 'landsat_B6']).rename('landsat_NDMI')
        return image.addBands(ndvi).addBands(ndmi)

    landsat = landsat.map(add_ls_indices)
    
    # Calculate NDVI and NDMI for Landsat
    ndvi_landsat = landsat.map(lambda image: image.normalizedDifference(['landsat_B5', 'landsat_B4']).rename('landsat_NDVI'))
    ndmi_landsat = landsat.map(lambda image: image.normalizedDifference(['landsat_B5', 'landsat_B6']).rename('landsat_NDMI'))

    alos_dsm_col = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2")\
        .filterBounds(roi)\
        .select('DSM')\
        .map(resample_to_10m)

    alos_dsm = alos_dsm_col.median()
    alos_dsm_available = alos_dsm_col.size().gt(0)

    srtm_dem = ee.Image("USGS/SRTMGL1_003").select('elevation')\
        .rename('DSM')\
        .resample('bilinear')\
        .reproject(crs='EPSG:4326', scale=10)

    dsm_final = ee.Image(ee.Algorithms.If(alos_dsm_available, alos_dsm, srtm_dem))
    
    # Closest image selection function
    def get_closest_image(selected_valid_date, collection):
        def add_diff(image):
            return image.set('diff', image.date().difference(selected_valid_date, 'day').abs())
        return collection.map(add_diff).sort('diff').first()
    
    # Match closest Sentinel-2 and Landsat-8 images to the Sentinel-1 date
    closest_s2 = get_closest_image(s1_start, sentinel_2)
    closest_ls = get_closest_image(s1_start, landsat)

    # Set map view and add layers
    st.session_state.map.set_center(lon, lat, zoom=14)
    
    def apply_wcm_sentinel1(image, ndvi_img):
        ndvi_eff = ndvi_img.clamp(0.0, 0.8)

        vv = image.select('VV')
        vh = image.select('VH')
        angle = image.select('angle')

        theta_rad = angle.multiply(math.pi / 180)
        cos_theta = theta_rad.cos()
        sec_theta = cos_theta.pow(-1)

        # === VV Correction
        A_vv = ndvi_eff.multiply(0.12)
        B_vv = ndvi_eff.multiply(0.70)
        tau2_vv = B_vv.multiply(ndvi_eff).multiply(sec_theta).multiply(-2).exp()
        sigma_veg_vv = A_vv.multiply(ndvi_eff).multiply(cos_theta).multiply(tau2_vv.subtract(1).multiply(-1))
        vv_lin = vv.divide(10).exp()
        sigma_soil_vv = vv_lin.subtract(sigma_veg_vv).divide(tau2_vv)
        vv_soil_db = sigma_soil_vv.log10().multiply(10).rename('VV_soil')

        # === VH Correction
        A_vh = ndvi_eff.multiply(0.05)
        B_vh = ndvi_eff.multiply(1.45)
        tau2_vh = B_vh.multiply(ndvi_eff).multiply(sec_theta).multiply(-2).exp()
        sigma_veg_vh = A_vh.multiply(ndvi_eff).multiply(cos_theta).multiply(tau2_vh.subtract(1).multiply(-1))
        vh_lin = vh.divide(10).exp()
        sigma_soil_vh = vh_lin.subtract(sigma_veg_vh).divide(tau2_vh)
        vh_soil_db = sigma_soil_vh.log10().multiply(10).rename('VH_soil')

        return image.addBands([vv_soil_db, vh_soil_db])

    # Define specific normalization parameters
    min_max_dict = {
        'VV_soil': (-54.837955, 24.112611),
        'VH_soil': (-56.586738, 24.244538),
        'DSM': (-5.0, 3690.0)
    }
    bands_10000 = [
        'sentinel2_B1', 'sentinel2_B2', 'sentinel2_B3', 'sentinel2_B4', 'sentinel2_B5',
        'sentinel2_B6', 'sentinel2_B7', 'sentinel2_B8', 'sentinel2_B8A', 'sentinel2_B9',
        'sentinel2_B10', 'sentinel2_B11', 'sentinel2_B12'
    ]

    bands_1000 = [
        'latitude', 'longitude', 'landsat_B10', 'landsat_B11'
    ]

    # === Selective normalization function ===
    def selective_normalize(image):
        normalized_bands = []

        # Min-max normalize specific bands
        for band, (band_min, band_max) in min_max_dict.items():
            norm = image.select(band).subtract(band_min).divide(band_max - band_min).multiply(2).subtract(1)
            normalized_bands.append(norm.rename(band))

        # Scale Sentinel-2 reflectance bands
        for band in bands_10000:
            scaled = image.select(band).divide(10000)
            normalized_bands.append(scaled.rename(band))

        # Scale lat/lon and thermal bands
        for band in bands_1000:
            scaled = image.select(band).divide(1000)
            normalized_bands.append(scaled.rename(band))

        # Keep remaining bands unscaled
        all_bands = image.bandNames().getInfo()
        normalized_names = list(min_max_dict.keys()) + bands_10000 + bands_1000
        remaining_bands = list(set(all_bands) - set(normalized_names))

        for band in remaining_bands:
            normalized_bands.append(image.select(band))

        return ee.Image.cat(normalized_bands)


    # Use Sentinel-2 NDVI for WCM
    ndvi_s2 = closest_s2.select('sentinel2_NDVI')

    # Apply WCM to Sentinel-1
    sentinel_1_corrected = apply_wcm_sentinel1(sentinel_1, ndvi_s2)

    # Combine corrected Sentinel-1 with other sources
    features_input = sentinel_1_corrected.addBands([
        lon_lat,
        closest_s2,
        closest_ls,
        dsm_final
    ]).clip(roi)

    # Apply normalization
    features_final = selective_normalize(features_input)

    # === Ensure feature order matches training ===
    feature_order = [
        'latitude', 'longitude', 'VV_soil', 'VH_soil',
        'sentinel2_NDVI', 'sentinel2_NDMI',
        'sentinel2_B1', 'sentinel2_B10', 'sentinel2_B11', 'sentinel2_B12',
        'sentinel2_B2', 'sentinel2_B3', 'sentinel2_B4', 'sentinel2_B5',
        'sentinel2_B6', 'sentinel2_B7', 'sentinel2_B8', 'sentinel2_B8A', 'sentinel2_B9',
        'landsat_NDVI', 'landsat_NDMI',
        'landsat_B1', 'landsat_B2', 'landsat_B3', 'landsat_B4', 'landsat_B5',
        'landsat_B6', 'landsat_B7', 'landsat_B8', 'landsat_B9', 'landsat_B10', 'landsat_B11',
        'DSM'
    ]

    features_final = features_final.select(feature_order)

    # Extract feature values
    feature_samples = features_final.sample(
        region=roi,
        scale=10,
        dropNulls=False
    )

    # Convert to DataFrame
    feature_list = feature_samples.reduceColumns(
        reducer=ee.Reducer.toList(len(feature_order)),
        selectors=feature_order
    ).get('list').getInfo()

    tabular_data = pd.DataFrame(feature_list, columns=feature_order)
    tabular_data = tabular_data[feature_order]  # Ensure column order is correct
    print(tabular_data)
    
    # Make predictions using the Random Forest model
    #predictions = rf_model.predict(tabular_data)
    # Generate predictions
    input_array = tabular_data.values.astype('float32')
    
    # Load models
    xgb_model = joblib.load("./models/xgb_model.pkl")
    tabnet_model = joblib.load("./models/tabnet_model.pkl")
    target_scaler = joblib.load("./models/tabnet_target_scaler.pkl")
    rf_model = joblib.load("./models/rf_model.pkl")

    with torch.no_grad():
        tabnet_preds_scaled = tabnet_model.predict(input_array)

    tabnet_preds = target_scaler.inverse_transform(tabnet_preds_scaled)
    xgb_preds = xgb_model.predict(input_array)
    rf_preds = rf_model.predict(input_array)

    # Assign weights: RF is the dominant model
    weights = [0.3, 0.4, 0.3]  # Adjusted to prioritize RF

    # Weighted ensemble prediction
    predictions = (
        weights[0] * rf_preds + 
        weights[1] * tabnet_preds.ravel() +
        weights[2] * xgb_preds
    )

       
    tabular_data['prediction'] = predictions
    print(tabular_data)
    print('--------')
    print(predictions)
    
    # Step 1: Convert lat/lon to 10m resolution scale and round
    tabular_data['lat_10m'] = np.round(tabular_data['latitude'] * 1000, 5)
    tabular_data['lon_10m'] = np.round(tabular_data['longitude'] * 1000, 5)

    # Step 2: Group by unique 10m coordinates and average predictions
    grouped = tabular_data.groupby(['lat_10m', 'lon_10m'], as_index=False)['prediction'].mean()

    # Step 3: Extract arrays
    latitudes = grouped['lat_10m'].values
    longitudes = grouped['lon_10m'].values
    predictions = grouped['prediction'].values

    # Step 4: Create EE Features from averaged coordinates
    prediction_features = []
    for lat, lon, pred in zip(latitudes, longitudes, predictions):
        point = ee.Geometry.Point([lon, lat])
        feature = ee.Feature(point, {'prediction': pred})
        prediction_features.append(feature)

    # Step 5: Create FeatureCollection
    prediction_fc = ee.FeatureCollection(prediction_features)

    # Step 6: Rasterize and reproject to 10m resolution
    prediction_ee = prediction_fc.reduceToImage(
        properties=['prediction'],
        reducer=ee.Reducer.mean()
    ).reproject(crs='EPSG:4326', scale=10)
    
    # Define a Gaussian kernel
    gaussian_kernel = ee.Kernel.gaussian(radius=200, sigma=1, units='pixels')

    # Convolve the prediction image with the Gaussian kernel
    smoothed_prediction = prediction_ee.convolve(gaussian_kernel)
    st.session_state["smoothed_prediction"] = smoothed_prediction
    sentinel_2_rgb = closest_s2.select(['sentinel2_B4', 'sentinel2_B3', 'sentinel2_B2'])

    # Set map view and add layers
    st.session_state.map = create_map(st.session_state.latitude, st.session_state.longitude, zoom=14)
    st.session_state.map.centerObject(roi, zoom=14)
    st.session_state.map.addLayer(sentinel_2_rgb, {'min': 0, 'max': 3000, 'bands': ['sentinel2_B4', 'sentinel2_B3', 'sentinel2_B2']}, "Sentinel-2 True Color")
    st.session_state.map.addLayer(smoothed_prediction, {
        'min': 0,
        'max': 0.5,
        'palette': [
            '8f7131',  # 0.0–0.05
            'b08e42',  # 0.05–0.1
            'c2b94d',  # 0.1–0.15
            'afc35e',  # 0.15–0.2
            '96c973',  # 0.2–0.25
            '86c58e',  # 0.25–0.3
            '64b9b2',  # 0.3–0.35
            '478fcd',  # 0.35–0.4
            '3461c6',  # 0.4–0.45
            '264cb5',  # 0.45–0.5
            '000000'   # >0.5 (black or another color)
        ]
    }, "Predicted Soil Moisture")
    
    legend_dict = {
        'Predicted Soil Moisture': '',
        '0.0 - 0.05': '8f7131',
        '0.05 - 0.1': 'b08e42',
        '0.1 - 0.15': 'c2b94d',
        '0.15 - 0.2': 'afc35e',
        '0.2 - 0.25': '96c973',
        '0.25 - 0.3': '86c58e',
        '0.3 - 0.35': '64b9b2',
        '0.35 - 0.4': '478fcd',
        '0.4 - 0.45': '3461c6',
        '0.45 - 0.5': '264cb5',
        '> 0.5': '000000'  # Black for values >0.5
    }
    
    # Add the legend to the map
    st.session_state.map.addLayerControl()
    st.session_state.map.add_legend(legend_dict=legend_dict)
    
    with col2:
        st_folium(st.session_state.map, height=600, width=800, key="main_map_updated")

    # Reset flag
    st.session_state.run_prediction = False
        
# Show download button
if "roi" in st.session_state and "smoothed_prediction" in st.session_state:
    if st.button("📥 Download Prediction Map (GeoTIFF)"):
        export_region = st.session_state["roi"].bounds()

        task = ee.batch.Export.image.toDrive(
            image=st.session_state["smoothed_prediction"],
            description='soil_moisture_prediction',
            folder='soil_moisture_{selected_valid_date}_{lat:.4f}_{lon:.4f}',
            fileNamePrefix=f"soil_moisture_{selected_valid_date}_{lat:.4f}_{lon:.4f}",
            region=export_region.getInfo()['coordinates'],
            scale=10,
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )

        task.start()

        st.success("✅ Export started! Check your Google Earth Engine Tasks tab.")
        st.markdown("📂 [Go to GEE Tasks](https://code.earthengine.google.com/tasks)")
        st.markdown(f"📁 The file will be saved in: Google Drive > soil_moisture_prediction > soil_moisture_{selected_valid_date}_{lat:.4f}_{lon:.4f}.tif")
else:
    st.warning("Please run a prediction before downloading.")
