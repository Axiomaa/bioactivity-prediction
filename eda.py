import pandas as pd
import os
import argparse
import seaborn as sns # type: ignore
sns.set_theme(style='ticks')
import matplotlib.pyplot as plt # type: ignore
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu # type: ignore

### Open up csv file ###
def load_data(csv_filename):
    """Load csv data into dataframe"""
    try: 
        print(f"Loading data from {csv_filename}...")
        df = pd.read_csv(csv_filename)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_filename} was not found.")
        exit(1)

### Frequency plot ###
def plot_frequency(df, column, output_folder):
    """Create frequency plot for specified column"""
    plt.figure(figsize=(5.5, 5.5))

    unique_values = df[column].unique()
    palette = sns.color_palette("husl", len(unique_values))

    # Count plot
    sns.countplot(
        x=column, 
        data=df, 
        edgecolor='black', 
        palette=palette
    )

    # Labels & styling
    plt.xlabel(column.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')

    # Save to pdf
    plt.savefig(os.path.join(output_folder, f"plot_{column}.pdf"))
    plt.tight_layout()
    plt.show()
    print(f"\nFrequency plot saved as plot_{column}.pdf")

# Scatter plot of MW versus LogP
def plot_scatter(df, x_column, y_column, output_folder):
    plt.figure(figsize=(5.5, 5.5))
    sns.scatterplot(
        x=x_column,
        y=y_column,
        data=df,
        hue='bioactivity_class',
        size='pIC50',
        sizes=(20, 200),
        edgecolor='black',
        alpha=0.7,
        palette='husl'
    )

    # Labels & styling
    plt.xlabel(x_column.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.ylabel(y_column.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.title(f'{y_column.replace("_", " ").title()} vs {x_column.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.grid(True)

    # Save to pdf
    plt.savefig(os.path.join(output_folder, f"plot_{x_column}_vs_{y_column}.pdf"))
    plt.tight_layout()
    plt.show()
    print(f"Scatter plot saved as plot_{x_column}_vs_{y_column}.pdf")

# Box plots
def plot_box(df, x_value, y_value, output_folder):
    plt.figure(figsize=(5.5, 5.5))

    sns.boxplot(
        x = x_value,
        y = y_value,
        data = df,
        palette='husl'
    )

    plt.xlabel(x_value.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.ylabel(y_value.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.title(f'{y_value.replace("_", " ").title()} vs {x_value.replace("_", " ").title()}', fontsize=16, fontweight='bold')

    # Save to pdf
    plt.savefig(os.path.join(output_folder, f"plot_{y_value}.pdf"))
    plt.tight_layout()
    plt.show()
    print(f"Box plot saved as plot_{y_value}.pdf")

### Mann-Whitney U Test ###
def mannwhitney(df, descriptor, output_folder, verbose=False):

    # Actives and inactives
    active = df[df['bioactivity_class'] == 'active'][descriptor]
    inactive = df[df['bioactivity_class'] == 'inactive'][descriptor]

    # Check if there are any values in active and inactive groups
    if len(active) == 0 or len(inactive) == 0:
        raise ValueError(f"No data found for descriptor: {descriptor}, Check your data.")

    # Perform the Mann-Whitney U test
    stat, p = mannwhitneyu(active, inactive)
    print('Statistics=%.3f, p=%.3f' % (stat,p))

    # Interpret
    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'

    results = pd.DataFrame({
        'Descriptor': [descriptor],
        'Statistics': [stat],
        'p': [p],
        'alpha': [alpha],
        'Interpretation': [interpretation]
    })

    # Save results to csv
    filename = f"mannwhitneyu_{descriptor}.csv"
    results.to_csv(os.path.join(output_folder, filename), index=False)
    print(f"Results of Mann-Whitney U test saved to {filename}")

    if verbose:
        print(results)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis on bioactivity data.')
    parser.add_argument('csv_filename', type=str, help='Name of the csv file to analyze')
    parser.add_argument('--output_folder', type=str, default='eda_results', help='Folder to save all output files.')
    args = parser.parse_args()

    global output_folder
    output_folder = 'eda_results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load csv file
    df = load_data(args.csv_filename)

    # Call frequency plotting function
    plot_frequency(df, 'bioactivity_class', output_folder)

    # Call scatter plotting function
    plot_scatter(df, 'MW', 'LogP', output_folder)

    # Call box plot function
    plot_box(df, 'bioactivity_class', 'pIC50', output_folder)
    plot_box(df, 'bioactivity_class', 'MW', output_folder)
    plot_box(df, 'bioactivity_class', 'LogP', output_folder)
    plot_box(df, 'bioactivity_class', 'NumHDonors', output_folder)
    plot_box(df, 'bioactivity_class', 'NumHAcceptors', output_folder)

    # Call the Mann-Whitney function for a specific descriptor
    mannwhitney(df, 'pIC50', output_folder, verbose=True)
    mannwhitney(df, 'MW', output_folder, verbose=True)
    mannwhitney(df, 'LogP', output_folder, verbose=True)
    mannwhitney(df, 'NumHDonors', output_folder, verbose=True)
    mannwhitney(df, 'NumHAcceptors', output_folder, verbose=True)

    print("Exploratory Data Analysis completed successful. Move on to analytics.py.")

