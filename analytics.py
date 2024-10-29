import pandas as pd
import argparse
import numpy as np
import os
from rdkit import Chem # type: ignore
from rdkit.Chem import Descriptors, Lipinski # type: ignore

# Loading in the csv file created
def load_data(csv_filename):
    df = pd.read_csv(csv_filename)
    return df

# Calculate Lipinski descriptors
# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation
def lipinski(smiles, chembl_ids, verbose=False):
    moldata, invalid_data, valid_indices = [], [], []

    for idx, (elem, chembl_id) in enumerate(zip(smiles, chembl_ids)):
        # Try converting non-string data (like floats) to strings
        if isinstance(elem, float) and not pd.isna(elem):
            elem = str(int(elem))
        
        # Process only if elem is not NaN or empty
        if pd.notna(elem) and isinstance(elem, str):
            mol = Chem.MolFromSmiles(elem)
            if mol:
                moldata.append(mol)
                valid_indices.append(idx) # Tacking valid indices
            else:
                invalid_data.append((chembl_id, elem))
                if verbose:
                    print(f"Invalid SMILES string for ChEMBL ID {chembl_id}: {elem}")
        else:
            invalid_data.append((chembl_id, elem))
            if verbose:
                print(f"Invalid or missing SMILES for ChEMBL ID {chembl_id}: {elem}")

    baseData = np.arange(1,1)
    i = 0
    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors])

        if(i == 0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i += 1

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data = baseData, columns = columnNames)

    return descriptors, invalid_data, valid_indices

# Function to convert the IC50 values to logarithmic scale
def pIC50(df):
    pIC50_values = []

    for value in df['standard_value']:
        if pd.notna(value):
            # Cap at 100,000,000
            value = min(value, 100000000)
            # Converts nM to M
            molar = value * 1e-9
            pIC50_values.append(-np.log10(molar))
        else:
            pIC50_values.append(np.nan)

    df['pIC50'] = pIC50_values
    df.drop('standard_value', axis=1, inplace=True)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load bioactivity data for analysis.')
    parser.add_argument('csv_filename', type=str, help='Name of the preprocessed CSV file')
    parser.add_argument('--output_folder', type=str, default='analytics_results', help='Folder to save all output files')
    args = parser.parse_args()

    global output_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the csv file
    df = load_data(args.csv_filename)

    ### Run the Lipinski descriptors ###
    df_lipinski, invalid_smiles, valid_indices = lipinski(df['canonical_smiles'], df['molecule_chembl_id'],verbose=True)
    print(f"\n{df_lipinski.head()}")

    # Save Lipinski descriptors df to csv file
    lipinski_filename = os.path.join(output_folder,'lipinski_descriptors.csv')
    df_lipinski.to_csv(lipinski_filename, index=False)
    print(f"\nLipinski descriptors saved to {lipinski_filename}")

    # Filter the original df to keep only valid rows based on valid_indices
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    merged_lipinski_df = pd.concat([df_filtered, df_lipinski], axis=1)

    # Save the merged df as a csv file
    combined_filename = os.path.join(output_folder, 'combined_bioactivity_lipinski_data.csv')
    merged_lipinski_df.to_csv(combined_filename, index=False)
    print(f"\nCombined bioactivity and Lipinski data saved to {combined_filename}\n")

    # Generate descriptive statistics
    print(f"Summary statistics for {combined_filename}:")
    print(merged_lipinski_df.standard_value.describe())

    ### Convert IC50 to pIC50 ###
    df_pIC50 = pIC50(merged_lipinski_df)
    print(f"\n{df_pIC50.head()}\n")
    df_pIC50_filename = os.path.join(output_folder, 'bioactivity_with_pIC50.csv')
    df_pIC50.to_csv(df_pIC50_filename, index=False)
    print(f"pIC50 values saved to {df_pIC50_filename}\n")

    print(f"\nSummary statistics for {df_pIC50_filename}:")
    print(df_pIC50.pIC50.describe())

    ### Removing the 'intermediate' class ###
    df_intermediate_removed = df_pIC50[df_pIC50['bioactivity_class'] != 'intermediate']
    print(f"\n{df_intermediate_removed.head()}\n")

    # Saving to csv file
    filtered_filename = os.path.join(output_folder, 'filtered_bioactivity_data.csv')
    df_intermediate_removed.to_csv(filtered_filename, index=False)
    print(f"Filtered (intermediate) bioactivity data saved to {filtered_filename}\n")

    print("Analytics done, move on to exploratory data analytics with eda.py.")

    