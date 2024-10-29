### Regression Models ###

import pandas as pd
import seaborn as sns # type: ignore
import argparse
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages # type: ignore
import lazypredict # type: ignore
from lazypredict.Supervised import LazyRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.feature_selection import VarianceThreshold # type: ignore

def load_data(csv_filename):
    df = pd.read_csv(csv_filename)
    return df

def build_random_forest_model(x_train, y_train, x_test, y_test, output_folder):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(x_train, y_train)

    # Evaluate model
    r2 = model.score(x_test, y_test)
    print(f"R^2 score: {r2}")

    # Predict on test data
    y_pred = model.predict(x_test)

    # Scatter plot of Experimental vs. Predicted pIC50 Values
    sns.set_theme(color_codes=True)
    sns.set_style("white")

    pdf_filename = os.path.join(output_folder, 'RandomForestModel.pdf')

    # Save plot to PDF
    with PdfPages(pdf_filename) as pdf:
        ax = sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.4})
        ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
        ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.figure.set_size_inches(5, 5)
        plt.text(0.05, 0.9, f"$R^2$: {r2:.3f}", transform=ax.transAxes,
                 fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
        pdf.savefig(ax.figure)
        plt.show()
        plt.close(ax.figure)
    
    print(f"Regression scatter plot saved as {pdf_filename}")

def compare_algorithms(x_train, x_test, y_train, y_test, output_folder):
    clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models_train, predictions_train = clf.fit(x_train, x_train, y_train, y_train)
    models_test, predictions_test = clf.fit(x_train, x_test, y_train, y_test)

    print("\nTraining set model comparisons:\n", models_train)
    print("\nTest set model comparisons:\n",  models_test)  

    # Save models to csv
    models_train.to_csv(os.path.join(output_folder,'models_train.csv'))
    models_test.to_csv(os.path.join(output_folder,'models_test.csv'))
    predictions_test.to_csv(os.path.join(output_folder,'predictions_test.csv'))
    predictions_train.to_csv(os.path.join(output_folder,'predictions_train.csv'))
    print("\nTraining and test set model comparisons saved to csv.")              

    # Bar plot for R-squared values from predicitons_train
    plt.figure(figsize=(5, 10))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(y=predictions_train.index, x="Adjusted R-Squared", data=predictions_train, palette="viridis")
    ax.set(xlim=(0,1))
    ax.set_title("Model R-Squared Comparison (Training Data)")
    ax.set_xlabel("R-Squared")
    ax.set_ylabel("Model")

    comparison_filename = os.path.join(output_folder, 'Model_Comparison.pdf')

    with PdfPages(comparison_filename) as pdf:
        pdf.savefig(ax.figure)
        plt.tight_layout()
        plt.show()
        plt.close(ax.figure)

    print(f"\nModel comparison plot saved as {comparison_filename}")

    return models_train, predictions_train, models_test, predictions_test

def plot_rmse_comparison(csv_filename, title="Model RMSE Comparison", output_filename="RMSE_Comparison.pdf"):
    model_data = pd.read_csv(csv_filename)

    if 'RMSE' not in model_data.columns or 'Model' not in model_data.columns:
        print("The CSV file does not contain required 'RMSE' or 'Model' columns.")
        return
    
    plt.figure(figsize=(5, 10))
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(y=model_data['Model'], x="RMSE", data=model_data, palette="coolwarm")
    ax.set(xlim=(0, 10))
    ax.set_title(title)
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Model")

    plt.savefig(os.path.join(output_folder, output_filename))
    plt.show()
    plt.close()

    print(f"\nRMSE comparison plot saved as {output_filename}")

def plot_time_taken(csv_filename, title='Model Time Taken Comparison', output_pdf='Time_Taken_Comparison.pdf', xlim_upper=10):
    predictions_df = pd.read_csv(csv_filename)

    plt.figure(figsize=(5, 10))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(y=predictions_df['Model'], x="Time Taken", data=predictions_df)
    ax.set_title(title)
    ax.set_xlabel("Time Taken (s)")
    ax.set_ylabel("Model")
    ax.set(xlim=(0, xlim_upper))

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, output_pdf))
    plt.show()
    plt.close()

    print(f"\nTime Taken comparison plot saved as {output_pdf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Regression models on PaDEL data.')
    parser.add_argument('csv_filename', type=str, help='Name of the csv file to run model on')
    parser.add_argument('--output_folder', type=str, default='regression_models_results', help='Folder to save output files')
    args = parser.parse_args()

    global output_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load csv file with PaDEL data
    df = load_data(args.csv_filename)

    # Input & output features
    input_x = df.drop('pIC50', axis=1)
    output_y = df.pIC50
    print("Shape of input (X):", input_x.shape)
    print("Shape of output (y):", output_y.shape)

    # Remove low variance features
    selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
    input_x = selection.fit_transform(input_x)
    print(f"Shape of input features after removing low variance features: {input_x.shape}")

    # Data split (80/20 ratio)
    x_train, x_test, y_train, y_test = train_test_split(input_x, output_y, test_size = 0.2, random_state = 42)
    print(f"Training input shape: {x_train.shape}, Training output shape: {y_train.shape}")
    print(f"Testing input shape: {x_test.shape}, Testing output shape: {y_test.shape}")

    # Build Random Forest model
    build_random_forest_model(x_train, y_train, x_test, y_test, output_folder)

    # Compare algorithms
    models_train, predictions_train, models_test, predictions_test = compare_algorithms(x_train, x_test, y_train, y_test, output_folder)

    # Compare RMSE values
    plot_rmse_comparison(os.path.join(output_folder, 'predictions_train.csv'), title="Model RMSE Comparison (Training Data)", output_filename="RMSE_Comparison_Train.pdf")
    plot_rmse_comparison(os.path.join(output_folder, 'predictions_test.csv'), title="Model RMSE Comparison (Test Data)", output_filename="RMSE_Comparison_Test.pdf")

    # Compare time taken
    plot_time_taken(os.path.join(output_folder, 'predictions_train.csv'), title='Time Taken comparison (Training data)', output_pdf='time_taken_train_comparison.pdf', xlim_upper=4)
    plot_time_taken(os.path.join(output_folder, 'predictions_test.csv'), title='Time Taken comparison (Test data)', output_pdf='time_taken_test_comparison.pdf', xlim_upper=4)

