# Bioactivity Prediction using the ChEMBL API

This project aims to develop predictive models for bioactivity data using molecular descriptors. Built using Python, it leverages tools such as PaDEL-Descriptor for feature generation and machine learning algorithms for regression modeling. The project is designed for analyzing chemical compound data with a focus on pIC50 values, which are key in bioactivity prediction for drug discovery.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [License](#license)

## Features

- **Data Collection and Preprocessing**: Collects bioactivity data from ChEMBL, cleans and preprocesses it, and caps extreme values for consistency.
- **Feature Calculation**: Calculates molecular descriptors using PaDEL-Descriptor to represent chemical features for modeling.
- **Exploratory Data Analysis**: Uses Lipinski descriptors to explore the dataset's chemical space, offering insights into molecular properties.
- **Regression Model Building**: Builds and evaluates multiple regression models (including Random Forest) to predicts pIC50 values. Algorithm comparison and model performance plots are saved for analysis.

## Installation

### Prerequisites
- **Python 3.x** should be installed on your machine.
- **Anaconda** or **Miniconda** should be installed on your machine.
- Optionally, use a **virtual environment** for isolating project dependencies.

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/Axiomaa/
    cd your-repo-name
    ```

2. Create and activate a Conda environment (optional but recommended):
    ```bash
    conda env create -f environment.yml
    ```
    ```bash
    conda activate your-environment-name
    ```
3. Run Pipeline
Each script should be run sequentially to execute data preprocessing, feature calculation, model building, and analysis.

## Usage
You can run each script from the command line. Below are the instructions for each script:

### `main.py`
This is the main entry point of the application, which fetches bioactivity data from the ChEMBL API based on a specific target.
```bash
python main.py <target_name>
```
Example:
```bash
python main.py coronavirus
```
#### Arguments
- `<target_name>`: The name of the biological target to search for (e.g., coronavirus, aromatase, etc.).
#### Output
The script will print the following to the console:
- List of potential targets matching the query.
- Bioactivity data for the selected target.
- Final processed data with molecule IDs, canonical SMILES, and bioactivity classifications.
---
### `analytics.py`
This script performs data analysis on bioactivity data by calculating Lipinski descriptors, converting IC50 values to pIC50, and filtering results for downstream analysis.
```bash
python analytics.py <path_to_input_file>
```
Example:
```bash
python analytics.py target_preprocessing_results\CHEMBL4523582_final_bioactivity_data.csv
```
#### Arguments
- `<path_to_input_file>`: Path to the file created in main.py after preprocessing.
#### Output
- **Lipinski descriptors**: `lipinski_descriptors.csv` - Contains calculated molecular properties for valid SMILES strings.
- **Combined Data**: `combined_bioactivity_lipinski_data.csv` - Merges the original bioactivity data with Lipinski descriptors.
- **pIC50 Data**: `bioactivity_with_pIC50.csv` - Adds a pIC50 column to the dataset after converting IC50 values.
- **Filtered Data**: `filtered_bioactivity_data.csv` - Removes compounds with an "intermediate" bioactivity classification. This is the file to be used in eda.py.
---
### `eda.py`
This script performs Exploratory Data Analysis (EDA) on the bioactivity data, generating various plots and conducting statistical tests to explore relationships within the data.
```bash
python eda.py <input_file>
```
Example:
```bash
python eda.py analytics_results\filtered_bioactivity_data.csv
```
#### Arguments
- `<input_file>`: The CSV file containing the Lipinski descriptors and bioactivity data (`filtered_bioactivity_data.csv`).
#### Functionality
- **Frequency Plot**: Creates a count plot showing the distribution of bioactivity classes.
- **Scatter Plot**: Visualizes the relationship between molecular weight (MW) and logP, with points colored by bioactivity class and sized by pIC50.
- **Box Plots**: Generates box plots to compare pIC50, MW, LogP, number of hydrogen donors, and acceptors across different bioactivity classes.
- **Mann-Whitney U Test**: Performs statistical tests comparing active and inactive compounds for specific descriptors (pIC50, MW, LogP, NumHDonors, NumHAcceptors) and saves the results to CSV files.
#### Output
The generated plots and Mann-Whitney test results are saved to the /results folder.
---
### `padel.py`
The `padel.py` script processes a pre-processed bioactivity dataset to generate molecular fingerprints and descriptors using the PaDEL-Descriptor tool. 
```bash
python padel.py <path_to_input_file>
```
Example:
```bash
python padel.py analytics_results\filtered_bioactivity_data.csv
```
#### Arguments
- `<path_to_input_file>`: The file constructed in analytics.py with Lipinski descriptors and bioactivity data (`filtered_bioactivity_data.csv`)
- `--output_folder`: (Optional) Directory where the output files will be saved. Default is `padel_results`.
#### Output
- `molecule.smi`: SMILES file for PaDEL input
- `padel_descriptors_output.csv`: File containing molecular descriptors generated by PaDEL.
- `bioactivity_data_3class_pIC50_pubchem_fp.csv`: Final dataset, containing both PaDEL descriptors and pIC50 values used for model building in `regmodel.py`.
---
### `regmodel.py`
This script `regmodel.py` is designed to perform regression analysis using Random Forest and additional model comparison techniques on a dataset prepared by PaDEL-Descriptor. It splits the data, builds and evaluates a Random Forest model, compares multiple regression models, and generates a series of visualizations to analyze model performance.
```bash
python regmodel.py <path_to_csv_file> --output_folder <output_folder>
```
Example: 
```bash
python regmodel.py padel_results\bioactivity_data_3class_pIC50_pubchem_fp.csv
```
#### Arguments
- `<path_to_csv_file>`: The name of the CSV file with PaDEL-Descriptor data. The dataset must include a column named `pIC50` for the target variable.
- `--output_folder`: (Optional) Folder where all output files will be saved. Default is `regression_models_results`
#### Output
- `RandomForestModel.pdf`: Trains a Random Forest Model, evaluates the R^2 score and the result is a scatter plot of experimental vs. predicted pIC50 values.
- `models_train.csv` and `models_test.csv`: Model performance data for training and testing sets.
- `predicitons_train.csv` and `predictions_test.csv`: Predictions and performance metrics for each model.
- `Model_Comparison.pdf`: Uses LazyPredict to compare a variety of regression algorithms and generates the bar plot of R-squared values for each model in `predictions_train.csv`.
- `RMSE_Comparison_Test.pdf` and `RMSE_Comparison_Train.pdf`: Bar plots comparing the RMSE values if the models.
- `time_traken_train_comparison.pdf` and `time_taken_test_comparison.pdf`: Comparison of time taken by models on training and test data.
---
## How It Works
### Step 1.1: Target Search
- The script uses the **ChEMBL API** to search for a specific biological target using a query (e.g., "coronavirus"). The API returns a list of matching target, and the user can select one of them.

### Step 1.2: Fetch Bioactivity Data
- After selecting the target (e.g., `CHEMBL4523582` for a specific coronavirus protein), the script retrieves bioactivity data related to that target. The bioactivity data includes IC50 values, which represent the drug potency.

### Step 1.3: Classify Compounds
- Compounds are classified based on their IC50 values:
    - **Active**: IC50 ≤ 1000 nM
    - **Intermediate**: IC50 between 1000 nM and 10000 nM
    - **Inactive**: IC50 ≥ 10000 nM

### Step 1.4: Remove Duplicates and Clean Data
- The script removes duplicate entries based on `molecule_chembl_id`, `canonical_smiles`, and `standard_value`. Only unique molecules are retained in the final dataset.

### Step 2.1 Calculate Lipinski descriptors
- **Descriptor Calculation**: For each compound's SMILES string, Lipinski descriptors are calculated to assess molecular properties like:
    - Molecular weight (MW)
    - LogP (Octanol-water partition coefficient)
    - Number of Hydrogen Donors
    - Number of Hydrogen Acceptors

### Step 2.2 Filter Valid Compounds and Merge Data
- **Data Filtering**: The original dataset is filtered to include only valid entries with accurate SMILES strings.
- **Merging Lipinski Data**: The filtered bioactivity data is then merged with the newly calculated Lipinski descriptors, creating a combined dataset that provides molecular and bioactivity details for each valid compound.

### Step 2.3 Convert IC50 to pIC50
- **IC50 to pIC50 conversion**: The `IC50` values are transformed to a logarithmic scale (pIC50) to improve the interpretability of potency data. Outliers are capped at 100,000,000 nM before conversion.

### Step 2.4 Remove intermediate bioactivity class
- **Class Filtering**: Compounds classified as "intermediate" in terms of bioactivity are removed from the dataset, focusing on "active" and "inactive" compounds.

### Step 3.1 Frequency and Scatter plot
- **Frequency visualization**: A frequency plot is generated for the active and inactive compounds. 
- **Descriptor relationship**: A scatter plot comparing the Molecular Weight (MW) and LogP values is generated with the `bioactivity_class` as hue and `pIC50` as size. 

### Step 3.2 Explore Lipinski Descriptor Distributions
- Lipinski descriptors, including MW, LogP, Number of Hydrogen Donors and Number of Hydrogen Acceptors, are analyzed and visualized with boxplots. These compare the distribution of each descriptor between active and inactive compounds. Key insights include assessing whether active and inactive compounds show distinct characteristics based on Lipinski rules.

### Step 3.3 Mann-Whitney U Test
- **Statistical Analysis**: The `mannwhitney` function performs the Mann-Whitney U test on each descriptor to compare active and inactive distributions, including `pIC50`, `MW`, `LogP`, `NumHDonors`, and `NumHAcceptors`.
    - Results are printed and saved in csv files in the output folder with a summary interpretation of the p-values.

### Step 4.1 Prepare data for PaDEL-Descriptor
- **Data Preparation**: the `prep_dataframe` function checks that the DataFrame includes the `canonical_smiles` and `molecule_chembl_id` columns, selects them and saves them in the .smi format for the PaDEL descriptor.

### Step 4.2 PaDEL-calculations and Processing Descriptor Data
- **PaDEL Descriptor Calculations**: `run_padel` takes the .smi file and generates the molecular descriptors.
- **Data Cleaning**: Unnecessary columns (like `Name`) are removed and the DataFrame is combined with the pIC50 values to be used in the model building.

### Step 5.1 Prepare the data for model building
- **Defining Input and Output**: The data from the PaDEL descriptor file with the pIC50 values are split into input and output.
    - The input (x) consists of the molecular descriptors.
    - The output (y) is the pIC50 values used as the target variable for model training. 
- **Feature Selection**: A variance threshold is used to filter out features with low variance to improve model performance and reducing overfitting.
- **Data Splitting**: The data is split into training and testing sets with an 80/20 ration. This allows for model validation to assess generalizability.

### Step 5.2 Build and Evaluate a Random Forest Model
- Train a Random Forest Regressor model on the training data, then evaluate the calculated R^2 score on the test data. The predicted vs. actual pIC50 values are plotted as a scatter plot.

### Step 5.3 Compare Algorithms Using Lazy Predict
- `LazyRegressor` is used to compare various regression algorithms on the training and test sets. The model comparison results are saved to csv files and model comparison plots of R^2 values for different models are generated.

### Step 5.4 RMSE Comparison
- To visually compare the models based on their RMSE scores, RMSE bar plots are generated for both training and test sets.

### Time Taken Comparison
- Generates bar plots comparing the time taken for each model on the training and test datasets.
---

## License
This project is licensed under the MIT license - see the [LICENSE](LICENCE) file for details. 
---

