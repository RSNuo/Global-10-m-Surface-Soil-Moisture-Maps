## Overview

This project provides a web-based tool for surface soil moisture prediction using an ensemble of machine learning models: **TabNet**, **Random Forest**, and **XGBoost**.

## Setup


1. **Create environment & install deps**  

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare data**
   Unzip data.7z. Inside is normalized_data.pkl, which contains cleaned and normalized samples:
   - target: soil moisture from the International Soil Moisture Network (soil_moisture)
   - features: derived from multi-sensor remote sensing.
     
## Usage Instructions

1. **Train Models**  
   Use the `training.ipynb` notebook to train the TabNet, Random Forest, and XGBoost models.  
   Ensure all required dependencies are installed and the training dataset is properly prepared before running the notebook.

2. **Enable Google Earth Engine (GEE)**  
   The web app uses the GEE Python API. Install and authenticate before launching:

   ```bash
   pip install earthengine-api
   earthengine authenticate
   ```
   Follow the browser flow (sign in, paste the token). In code, the app will call ee.Initialize().
   Docs: Intro to the Python API — https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api
   
4. **Launch the Web Application**  

   ```bash
   streamlit run sm_eeapps.py
   ```

3. **Make Predictions**  
   In the Streamlit app:
   - Click any point on the map **or** manually enter longitude and latitude within the continent.
   - Choose a start and end date, then click **"Check Available Dates"**.
   - From the returned list, select a valid date and click **"Run Prediction"**.
   - The tool will perform soil moisture prediction for the selected location and date using the trained ensemble models.
