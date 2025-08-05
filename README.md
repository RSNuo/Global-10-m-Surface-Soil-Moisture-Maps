## Overview

This project provides a web-based tool for surface soil moisture prediction using an ensemble of machine learning models: **TabNet**, **Random Forest**, and **XGBoost**.

## Usage Instructions

1. **Train Models**  
   Use the `training.ipynb` notebook to train the TabNet, Random Forest, and XGBoost models.  
   Ensure all required dependencies are installed and the training dataset is properly prepared before running the notebook.

2. **Launch the Web Application**  
   After training the models, start the web tool by running the following command in your terminal:

   ```bash
   streamlit run sm_eeapps.py
   ```

3. **Make Predictions**  
   In the Streamlit app:
   - Click any location on the continental map.
   - Select a date.
   - The tool will perform soil moisture prediction for the selected location and date using the trained ensemble models.
