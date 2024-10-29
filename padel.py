from padelpy import padeldescriptor # type: ignore
import pandas as pd
import argparse
import os

def load_data(csv_filename):
    df = pd.read_csv(csv_filename)
    return df

def prep_dataframe(df, output_folder, verbose=False):
    if 'canonical_smiles' not in df.columns or 'molecule_chembl_id' not in df.columns:
        raise ValueError("DataFrame must contain 'canonical_smiles' and 'molecule_chembl_id' columns.")
    
    selection = ['canonical_smiles', 'molecule_chembl_id']
    df_selection = df[selection]

    prep_filename = os.path.join(output_folder, 'molecule.smi')
    df_selection.to_csv(prep_filename, sep='\t', index=False, header=False)
    print(f"Data prepared and saved to {prep_filename}")

    if verbose:
        with open(prep_filename, 'r') as file:
            for i, line in enumerate(file):
                print(line.strip())
                if i >= 4:
                    break

        with open(prep_filename, 'r') as file:
            line_count = sum(1 for line in file)
        print(f"\nTotal numebr of lines in molecule.smi: {line_count}")

    return df_selection, prep_filename

def run_padel(input_file, output_folder):
    output_file = os.path.join(output_folder, 'padel_descriptors_output.csv')
    padeldescriptor(
        mol_dir = input_file,
        d_file = output_file,
        removesalt = True,
        standardizenitro = True,
        fingerprints = True,
    )

    print(f"\nPaDEL descriptors saved to {output_file}")
    return output_file

def remove_name_column(df):
    if 'Name' in df.columns:
        df = df.drop(columns = ['Name'])
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare bioactivity data for PaDEL-Descriptor.')
    parser.add_argument('csv_filename', type=str, help='Name of the csv file to prepare')
    parser.add_argument('--output_folder', type=str, default='padel_results', help='Folder for saving output files')
    args = parser.parse_args()

    global output_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load csv file with bioactivity data
    df = load_data(args.csv_filename)

    # Prep the dataframe
    smi_file, smi_filename = prep_dataframe(df, output_folder, verbose=True)

    # Run PaDEL
    padel_output_file = run_padel(smi_filename, output_folder)

    # Remove 'Name' column
    padel_df = load_data(padel_output_file)
    padel_X = remove_name_column(padel_df)
    print(f"\n{padel_X.head()}")

    # Converting IC50 to pIC50
    padel_Y = df['pIC50']
    print(f"\n{padel_Y.head()}")

    # Combining the PaDEL df with the pIC50 values variables
    pubchem_dataset = pd.concat([padel_X, padel_Y], axis=1)
    pubchem_filename = os.path.join(output_folder,'bioactivity_data_3class_pIC50_pubchem_fp.csv')
    pubchem_dataset.to_csv(pubchem_filename, index=False)
    print("\nCombined pubchem and pIC50 variables: ")
    print(f"{pubchem_dataset.head()}\n")
    print(f"Dataset saved as {pubchem_filename}.")

    print("PaDEL calculations done. Move on to regmodel.py for model building.")



    