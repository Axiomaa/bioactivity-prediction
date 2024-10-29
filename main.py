### Search ChEMBL for target protein and preprocess data ###

import os
import pandas as pd
from chembl_webresource_client.new_client import new_client
import argparse 

# Function to search for target using the ChEMBL API
def search_target(query):
    target = new_client.target
    target_query = target.search(query)
    return pd.DataFrame.from_dict(target_query)

# Function to search for bioactivity data (IC50) for a specific target
def search_bioactivity(chembl_id):
    activity = new_client.activity
    res = activity.filter(target_chembl_id=chembl_id, standard_type="IC50")
    df = pd.DataFrame.from_dict(res)

    # Removing entries with missing 'standard_value' (drug potency)
    filtered_df = df[df['standard_value'].notna()]

    return filtered_df

# Function to classify compounds by bioactivity [nM]
def classify_bioactivity(df):
    bioactivity_class = []
    for i in df['standard_value']:
        try:
            value = float(i)
            if value >= 10000:
                bioactivity_class.append("inactive")
            elif value <= 1000:
                bioactivity_class.append("active")
            else:
                bioactivity_class.append("intermediate")
        except ValueError:
            bioactivity_class.append("unknown")
    
    # Add classification to new column
    df['bioactivity_class'] = bioactivity_class
    return df

# Drop duplicate data and create data frame with ID, canonical_smiles, standard_value and bioactivity_class
def remove_duplicates(df):
    unique_df = df.drop_duplicates(subset=['molecule_chembl_id', 'canonical_smiles', 'standard_value'])
    final_df = unique_df[['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value']]
    return final_df

# Main block code
if __name__ == '__main__':
    # Set up argparse to get the search term from the user
    parser = argparse.ArgumentParser(description="Search for bioactivity data using the ChEBML API")
    parser.add_argument("target_name", help="Enter the name of the target you want to search for (e.g., coronavirus)")
    parser.add_argument("--output_dir", default="target_preprocessing_results", help="Directory to save output files to.")
    args = parser.parse_args()

    # Create output dir if it doesn't exist
    global output_dir
    output_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Step 1: Perform target search
    target_name = args.target_name
    targets = search_target(target_name)

    # Check if any targets were found
    if targets.empty:
        print(f"No targets found for '{target_name}'")
    else:
        # Print DataFrame of targets
        print(f"Targets found for '{target_name}':")
        print(targets)

        # Write DataFrame to a .csv file
        csv_filename = os.path.join(args.output_dir, f"{target_name}_targets_output.csv")
        targets.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

        # Allow user to select the target by index
        index = int(input(f"Select the index of the target (0-{len(targets)-1}): "))
        selected_target_id = targets.target_chembl_id[index]
        print(f"\nSelected target ID: {selected_target_id}")

        # Step 2: Fetch bioactivity data (IC50) for a specific target
        bioactivity_df = search_bioactivity(selected_target_id)
        if bioactivity_df.empty:
            print(f"No bioactivity data found for target {selected_target_id}.")
        else: 
            print(f"\nBioactivity IC50 data for target {selected_target_id}:")
            print(bioactivity_df.head())

            # Save bioactivity data to .csv file
            bioactivity_filename = os.path.join(args.output_dir, f"{selected_target_id}_IC50.csv")
            bioactivity_df.to_csv(bioactivity_filename, index=False)
            print(f"\nBioactivity data saved to {bioactivity_filename}")

            # Step 3: Classify compounds by bioactivity (active, inactive, intermediate)
            classified_bioactivity_df = classify_bioactivity(bioactivity_df)
            print(f"\nClassified bioactivity data for target {selected_target_id}:")
            print(classified_bioactivity_df.head())

            # Step 4: Remove duplicates and select unique molecule entries
            final_df = remove_duplicates(classified_bioactivity_df)
            print("f\nFinal DataFrame with unique molecules and bioactivity classification:")
            print(final_df.head())

            # Save to .csv file
            final_df_filename = os.path.join(args.output_dir, f"{selected_target_id}_final_bioactivity_data.csv")
            final_df.to_csv(final_df_filename, index=False)
            print(f"\nFinal bioactivity data save to {final_df_filename}")
            print(f"\nPreprocessing done. Move on to the analytics.py file.")