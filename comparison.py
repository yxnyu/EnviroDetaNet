import torch
from detanet_model import True_nn_vib_analysis, Lorenz_broadening
import csv
from scipy.stats import spearmanr
import numpy as np
import logging
import sys
import os
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler("vib_analysis.log"), logging.StreamHandler(sys.stdout)])

device = torch.device('cpu')
dtype = torch.float32

logging.info("Starting script execution")

# Create directory for storing CSV files
output_dir = 'envir_ir_csv'
os.makedirs(output_dir, exist_ok=True)

# Load datasets
logging.info("Loading datasets")
try:
    datasets = torch.load('/path/to/your/pt')
    datasets_prediction = torch.load('/path/to/your/prediction_truth.pt')
    logging.info(f"Successfully loaded {len(datasets)} datasets")
    logging.info(f"datasets_prediction contains {len(datasets_prediction)} SMILES")
    
    # Print sample dataset to verify structure
    if datasets:
        logging.info(f"datasets sample: {datasets[0]}")
    if datasets_prediction:
        first_smiles = next(iter(datasets_prediction))
        first_data = datasets_prediction[first_smiles]
        logging.info(f"datasets_prediction sample:")
        logging.info(f"  SMILES: {first_smiles}")
        logging.info(f"  Data: {first_data}")
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    sys.exit(1)

def process_data_point(pos, z, Hi, Hij, dedipole, depolar, pre_Hij, pre_Hi, pre_dedipole, pre_depolar, smile, index):
    try:
        logging.info(f"Processing data point {index}")
        logging.info(f"Shapes: pos={pos.shape}, z={z.shape}, Hi={Hi.shape}, Hij={Hij.shape}, dedipole={dedipole.shape}, depolar={depolar.shape}, "
                     f"pre_Hi={pre_Hi.shape}, pre_Hij={pre_Hij.shape}, pre_dedipole={pre_dedipole.shape}, pre_depolar={pre_depolar.shape}")
        
        # Model using true values
        vib_model_true = True_nn_vib_analysis(device=device, Hi=Hi, Hij=Hij, dedipole=dedipole, depolar=depolar, Linear=False, scale=0.965)
        freq_true, iir_true, araman_true = vib_model_true(z=z, pos=pos)
        
        # Model using predicted values
        vib_model_pred = True_nn_vib_analysis(device=device, Hi=pre_Hi, Hij=pre_Hij, dedipole=pre_dedipole, depolar=pre_depolar, Linear=False, scale=0.965)
        freq_pred, iir_pred, araman_pred = vib_model_pred(z=z, pos=pos)
        
        x_axis = torch.linspace(500, 4000, 3501)
        yir_true = Lorenz_broadening(freq_true, iir_true, c=x_axis, sigma=15).detach().numpy()
        yir_pred = Lorenz_broadening(freq_pred, iir_pred, c=x_axis, sigma=15).detach().numpy()
        y_pred = yir_pred / yir_pred.max()
        y_true = yir_true / yir_true.max()
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        # Save y_pred, y_true and SMILES as CSV
        csv_filename = os.path.join(output_dir, f'y_pred_y_true_{index}.csv')
        df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 'SMILES': smile})
        df.to_csv(csv_filename, index=False)
        logging.info(f"Saved y_pred, y_true and SMILES for data point {index} to {csv_filename}")
        
        return spearman_corr
    except Exception as e:
        logging.error(f"Error processing data point {index}: {e}")
        return None

# Process all data points
logging.info("Processing data points")
spearman_correlations = []
for i, dataset in enumerate(datasets):
    try:
        # Ensure dataset has correct attributes
        required_attrs = ['pos', 'z', 'Hii', 'Hij', 'depolar', 'dedipole', 'smile']
        if not all(hasattr(dataset, attr) for attr in required_attrs):
            logging.warning(f"Data point {i} missing necessary attributes, skipping")
            continue
        
        pos = dataset.pos
        z = dataset.z
        Hi = dataset.Hii
        Hij = dataset.Hij
        depolar = dataset.depolar
        dedipole = dataset.dedipole
        smile = dataset.smile[0] if isinstance(dataset.smile, list) else dataset.smile
        
        # Find matching data in datasets_prediction
        if smile in datasets_prediction:
            matching_data = datasets_prediction[smile]
            pre_depolar = matching_data.get('predicted_depolar')
            pre_dedipole = matching_data.get('predicted_dedipole')
            pre_Hi = matching_data.get('predicted_Hi')
            pre_Hij = matching_data.get('predicted_Hij')
            
            if any(v is None for v in [pre_depolar, pre_dedipole, pre_Hi, pre_Hij]):
                logging.warning(f"SMILE: {smile} missing necessary predicted data")
                continue
        else:
            logging.warning(f"SMILE: {smile} not found in datasets_prediction")
            continue
        
        spearman_corr = process_data_point(pos, z, Hi, Hij, dedipole, depolar, pre_Hij, pre_Hi, pre_dedipole, pre_depolar, smile, i)
        if spearman_corr is not None:
            spearman_correlations.append(spearman_corr)
        
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i + 1} data points")
    except Exception as e:
        logging.error(f"Error processing data point {i}: {e}")

# Calculate average Spearman correlation
logging.info("Calculating average Spearman correlation")
if spearman_correlations:
    average_spearman_corr = np.mean(spearman_correlations)
    std_spearman_corr = np.std(spearman_correlations)
    logging.info(f"Average Spearman correlation: {average_spearman_corr:.4f} ± {std_spearman_corr:.4f}")
else:
    logging.warning("No valid Spearman correlation data")

# Save results to file
logging.info("Saving results to file")
try:
    with open('spearman_correlations.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Data Point', 'Spearman Correlation'])
        for i, corr in enumerate(spearman_correlations):
            writer.writerow([i, corr])
    logging.info("Results saved to 'spearman_correlations.csv'")
except Exception as e:
    logging.error(f"Error saving results to file: {e}")

logging.info("Script execution completed")