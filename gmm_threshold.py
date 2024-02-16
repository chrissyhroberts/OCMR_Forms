#pip3 install matplotlib numpy scikit-learn

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import argparse

def load_json_data(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data

def extract_intensities(data):
    return [entry["intensity"] for key in data for entry in data[key]]

def plot_density(intensity_array, output_file=None):
    plt.figure(figsize=(10, 6))
    plt.hist(intensity_array, bins=30, density=True, alpha=0.6, color='g')

    if output_file:  # Save plot if output file is provided
        plt.savefig(f"{output_file}_density.png")

def perform_em_analysis(intensity_array, output_file=None):
    gmm = GaussianMixture(n_components=2, random_state=0).fit(intensity_array)
    threshold = np.mean(gmm.means_)
    print("Means of the two distributions:", gmm.means_.flatten())
    print("Suggested threshold value:", threshold)
    
    x = np.linspace(intensity_array.min(), intensity_array.max(), 1000).reshape(-1, 1)
    log_prob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(log_prob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    
    plt.fill_between(x.flatten(), pdf_individual[:, 0], alpha=0.5)
    plt.fill_between(x.flatten(), pdf_individual[:, 1], alpha=0.5)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title('Density and GMM Analysis of Intensity Values')
    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.legend()

    if output_file:  # Save plot if output file is provided
        plt.savefig(f"{output_file}_gmm.png")
    else:  # Show plot if no output file is provided
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process a JSON file to analyze intensity values.')
    parser.add_argument('-i', '--input', help='Input JSON file with intensity data', required=True)
    parser.add_argument('-o', '--output', help='Output file path for saving the chart', required=False)
    
    args = parser.parse_args()
    
    data = load_json_data(args.input)
    intensity_values = extract_intensities(data)
    intensity_array = np.array(intensity_values).reshape(-1, 1)
    
    plot_density(intensity_array, args.output)
    perform_em_analysis(intensity_array, args.output)

if __name__ == "__main__":
    main()
