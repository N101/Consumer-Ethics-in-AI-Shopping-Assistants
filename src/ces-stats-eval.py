import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt


def load_ai_data(model_data_paths):
    """
    Load and process AI response data from multiple models.

    Parameters:
    model_data_paths (dict): Dictionary containing paths for each model
        Format: {
            'model_name': {
                'raw': 'path/to/raw_data.csv',
                'avg': 'path/to/averages.csv'
            }
        }

    Returns:
    dict: Dictionary containing DataFrames for each model
    """
    model_data = {}

    for model_name, paths in model_data_paths.items():
        raw_df = pd.read_csv(paths["raw"])
        avg_df = pd.read_csv(paths["avg"])
        model_data[model_name] = {"raw": raw_df, "avg": avg_df}

    return model_data


def load_human_data(survey_path):
    """
    Load human survey responses for original CES.

    Parameters:
    survey_path (str): Path to survey responses CSV with format: #,students,non-students

    Returns:
    dict: Dictionary with keys 'Students' and 'Non-Students' containing their respective data
    """
    human_df = pd.read_csv(survey_path)
    
    # Extract student and non-student data into separate arrays
    human_data = {
        'Students': [],
        'Non-Students': []
    }
    
    # For each question row, add the values to the respective groups
    for _, row in human_df.iterrows():
        human_data['Students'].append(row['students'])
        human_data['Non-Students'].append(row['non-students'])
    
    return human_data


def get_category_questions():
    """
    Define which questions belong to which categories for the original CES.
    Adjust these mappings according to your original CES categories.

    Returns:
    dict: Mapping of categories to question numbers
    """
    return {
        "Active": [1, 2, 3, 4, 5],              # Actively benefiting from illegal activities
        "Passive": [6, 7, 8, 9, 11],            # Passively benefiting
        "Questionable": [12, 13, 14, 15, 16],   # Actively benefiting from questionable but legal actions
        "No_Harm": [17, 18, 19, 20, 21],        # No harm/no foul
        "Downloading": [22, 23],                # downloading copyrighted materials/buying counterfeit goods
        "Recycling": [24, 25, 26, 27],          # recycling/environmental awareness
        "Doing_Good": [28, 29, 30, 31]          # doing good/doing the right thing
    }


def process_data_for_analysis(human_data, ai_model_data, categories):
    """
    Process and organize data for statistical analysis.

    Parameters:
    human_data (dict): Dictionary with human data for 'Students' and 'Non-Students'
    ai_model_data (dict): Dictionary containing AI model data
    categories (dict): Category to question mapping

    Returns:
    dict: Processed data organized by category
    """
    processed_data = {}

    # Process each category
    for category, questions in categories.items():
        # Initialize category data with human groups
        category_data = {
            'Students': [],
            'Non-Students': []
        }
        
        # Add models to category_data
        for model_name in ai_model_data.keys():
            category_data[model_name] = []

        # Process human data
        for q in questions:
            q_index = q - 1  # Adjust for 0-indexing if needed
            if q_index < len(human_data['Students']):
                category_data['Students'].append(human_data['Students'][q_index])
                category_data['Non-Students'].append(human_data['Non-Students'][q_index])

        # Process AI data
        for model_name, model_dfs in ai_model_data.items():
            raw_df = model_dfs["raw"]
            for q in questions:
                mask = raw_df["#"] == q
                category_data[model_name].extend(raw_df.loc[mask, "Response"].tolist())

        processed_data[category] = category_data

    return processed_data


def analyze_all_categories(processed_data):
    """
    Run statistical analysis on all categories.

    Parameters:
    processed_data (dict): Processed data organized by category

    Returns:
    dict: Analysis results for all categories
    """
    results = {}

    for category, data in processed_data.items():
        # Prepare data for analysis
        all_data = []
        group_labels = []

        for group, scores in data.items():
            all_data.extend(scores)
            group_labels.extend([group] * len(scores))

        # Convert to numpy arrays for analysis
        all_data = np.array(all_data)
        
        # Filter out any NaN values
        valid_indices = ~np.isnan(all_data)
        all_data = all_data[valid_indices]
        group_labels = [label for i, label in enumerate(group_labels) if valid_indices[i]]

        # Remove any empty groups
        groups = []
        group_names = []
        for group, scores in data.items():
            scores_array = np.array(scores)
            valid_scores = scores_array[~np.isnan(scores_array)]
            if len(valid_scores) > 0:
                groups.append(valid_scores)
                group_names.append(group)

        # Perform ANOVA if we have at least 2 groups with data
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            
            # Calculate MSE (correct calculation)
            grand_mean = np.mean(all_data)
            mse = 0
            n_total = 0
            
            for group in groups:
                group_mean = np.mean(group)
                for value in group:
                    mse += (value - group_mean) ** 2
                n_total += len(group)
            
            mse = mse / (n_total - len(groups))  # Divide by degrees of freedom

            # Perform Tukey's HSD
            tukey = pairwise_tukeyhsd(all_data, group_labels)

            # Store results
            results[category] = {
                "f_statistic": f_stat,
                "p_value": p_val,
                "tukey_results": tukey,
                "means": {group: np.mean(np.array(scores)[~np.isnan(scores)]) if len(np.array(scores)[~np.isnan(scores)]) > 0 else np.nan 
                         for group, scores in data.items()},
                "std": {group: np.std(np.array(scores)[~np.isnan(scores)]) if len(np.array(scores)[~np.isnan(scores)]) > 0 else np.nan 
                       for group, scores in data.items()},
                "mse": mse
            }
        else:
            # Not enough groups for analysis
            results[category] = {
                "f_statistic": np.nan,
                "p_value": np.nan,
                "tukey_results": None,
                "means": {group: np.mean(np.array(scores)[~np.isnan(scores)]) if len(np.array(scores)[~np.isnan(scores)]) > 0 else np.nan 
                         for group, scores in data.items()},
                "std": {group: np.std(np.array(scores)[~np.isnan(scores)]) if len(np.array(scores)[~np.isnan(scores)]) > 0 else np.nan 
                       for group, scores in data.items()},
                "mse": np.nan
            }

    return results


def visualize_results(processed_data, category, output_dir=None):
    """
    Create visualization for a category's results.

    Parameters:
    processed_data (dict): Processed data for the category
    category (str): Category name
    output_dir (str, optional): Directory to save plots to
    """
    # Prepare data for plotting
    plot_data = []
    for group, scores in processed_data[category].items():
        scores_array = np.array(scores)
        valid_scores = scores_array[~np.isnan(scores_array)]
        for score in valid_scores:
            plot_data.append({"Group": group, "Score": score})

    df = pd.DataFrame(plot_data)
    
    if len(df) == 0:
        print(f"No valid data to plot for category: {category}")
        return

    # Create plot
    plt.figure(figsize=(12, 7))
    sns.boxplot(x="Group", y="Score", data=df, palette="Set3")
    
    # Add a title and labels
    plt.title(f"Distribution of Scores for {category}", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Group", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        plt.savefig(f"{output_dir}/{category}_boxplot.png", dpi=300, bbox_inches='tight')
    
    plt.show()


# Main execution
if __name__ == "__main__":
    # Define file paths for each model
    model_data_paths = {
        "GPT-3.5-turbo": {
            "raw": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/raw_data/GPT-3.5-turbo_raw_data.csv",
            "avg": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/averages/GPT-3.5-turbo_averages.csv",
        },
        "GPT4o": {
            "raw": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/raw_data/GPT-4o_raw_data.csv",
            "avg": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/averages/GPT-4o_averages.csv",
        },
        "GPT4o-mini": {
            "raw": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/raw_data/GPT-4o-mini_raw_data.csv",
            "avg": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/averages/GPT-4o-mini_averages.csv"
        },
        "Gemini-1.5-flash": {
            "raw": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/raw_data/Gemini_raw_data.csv",
            "avg": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/averages/Gemini_averages.csv"
        },
        "Grok": {
            "raw": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/raw_data/Grok_raw_data.csv",
            "avg": "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/averages/Grok_averages.csv"
        },
    }

    HUMAN_SURVEY_PATH = "/Users/noahkiefer/Documents/Uni/,BA_THESIS/LLM-consumer-ethics/resources/data/CES_modified_2005.csv"
    OUTPUT_DIR = None # Optional, set to None if you don't want to save plots

    # Load data
    print("Loading data...")
    ai_model_data = load_ai_data(model_data_paths)
    human_data = load_human_data(HUMAN_SURVEY_PATH)

    # Get category definitions
    categories = get_category_questions()

    # Process data
    print("Processing data...")
    processed_data = process_data_for_analysis(human_data, ai_model_data, categories)

    # Run analysis
    print("Running statistical analysis...")
    results = analyze_all_categories(processed_data)

    # Print results and create visualizations
    for category in categories.keys():
        print(f"\nResults for {category}:")
        print("-" * 60)
        print(f"ANOVA Results:")
        if np.isnan(results[category]['f_statistic']):
            print("Insufficient data for ANOVA analysis")
        else:
            print(f"F-statistic: {results[category]['f_statistic']:.4f}")
            print(f"p-value: {results[category]['p_value']:.4f}")
        
        print("\nGroup Means:")
        for group, mean in results[category]["means"].items():
            if not np.isnan(mean):
                std = results[category]["std"][group]
                print(f"{group}: {mean:.4f} (±{std:.4f})")
            else:
                print(f"{group}: No valid data")
        
        if not np.isnan(results[category]['mse']):
            print(f"\nMean Squared Error: {results[category]['mse']:.4f}")
        
        if results[category]["tukey_results"] is not None:
            print("\nTukey's HSD Results:")
            print(results[category]["tukey_results"])
        else:
            print("\nInsufficient data for Tukey's HSD test")

        # Create visualization
        visualize_results(processed_data, category, OUTPUT_DIR)
