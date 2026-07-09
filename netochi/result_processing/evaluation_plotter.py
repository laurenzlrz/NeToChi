import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_input_id(input_id):
    # Matches patterns like: [base_graph]_[prob]p_[seed]seed
    match = re.match(r"(.*)_([0-9.]+)p_([0-9]+)seed", input_id)
    if match:
        base_graph = match.group(1)
        prob = match.group(2)
        seed = match.group(3)
        return pd.Series([base_graph, prob, seed])
    else:
        # Fallback if format doesn't match
        return pd.Series([input_id, "Unknown", "Unknown"])


if __name__ == "__main__":
    RUN_NAME = "clustering_comparison"
    METRIC_IN_DF = 'Core Dist. Diff. (Clustering)'
    TITLE = "Difference in Core Distances"
    X_LABEL = 'Core Distance Difference'


    # 1. Load the data
    # Replace 'data.csv' with the path to your actual CSV file
    df = pd.read_csv(f'results/{RUN_NAME}/evaluation_results.csv')

    # 2. Filter the dataframe for the clustering accuracy metric
    df_acc = df[df['metric'] == METRIC_IN_DF].copy()
    df_acc['value'] = pd.to_numeric(df_acc['value'])


    # 3. Parse the input_id to extract base graph, probability, and seed


    df_acc[['base_graph', 'prob', 'seed']] = df_acc['input_id'].apply(parse_input_id)

    sorted_mappers = sorted(df_acc['mapper'].unique())
    # 4. Create one plot per base graph
    unique_graphs = df_acc['base_graph'].unique()

    for graph in unique_graphs:
        # Subset data for the current graph
        subset = df_acc[df_acc['base_graph'] == graph]

        # Create the figure
        plt.figure(figsize=(10, 6))

        # Plot bars grouped by prob.
        # 'errorbar="sd"' tells seaborn to plot the standard deviation across the seeds
        sns.barplot(
            data=subset,
            x='mapper',
            y='value',
            hue='prob',
            errorbar='sd',
            capsize=0.1,  # Adds little caps to the error bars
            order=sorted_mappers,
            palette='viridis'
        )

        # Formatting the plot
        plt.title(f'{TITLE} for {graph}', fontsize=14, fontweight='bold')
        plt.xlabel('Mapper', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(X_LABEL, fontsize=12)
        #plt.ylim(0, 1.05)  # Accuracy is usually between 0 and 1

        # Format the legend
        plt.legend(title='Connection Probability', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to prevent clipping of the legend
        plt.tight_layout()

        # Save or show the plot
        filename = f"accuracy_plot_{graph}.png"
        plt.savefig(filename, dpi=300)
        plt.show()

        print(f"Saved plot for {graph} as {filename}")