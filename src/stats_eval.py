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
    Load human survey responses.

    Parameters:
    survey_path (str): Path to survey responses CSV

    Returns:
    DataFrame: Processed survey data
    """
    survey = pd.read_csv(survey_path)
    survey = survey.drop(columns=[survey.columns[i] for i in [0, 1, 2, 5, 7, 9]])
    survey = survey.drop(survey.columns[:12], axis=1)
    column_oder = [
        "Q4_company-license",
        "Q14_subscritpion",
        "Q6_screenshots",
        "Q8_streaming",
        "Q16_ad-blockers",
        "Q2_VPN_bypass",
        "Q11_VPN_prices",
        "Q12_free_trials",
        "Q7_bots",
        "Q5_status",
        "Q9_loyalty",
        "Q1_refunds",
        "Q15_fake_review",
        "Q10_return",
        "Q17_website_errors",
        "Q3_honest_reviews",
        "Q13_not_reporting",
    ]
    survey = survey[column_oder]
    survey = survey.drop([0, 1])
    survey = survey.astype(int)

    return survey


def get_category_questions():
    """
    Define which questions belong to which categories.
    Based on your provided documentation.

    Returns:
    dict: Mapping of categories to question numbers
    """
    return {
        "Resource_Misappropriation": [1, 2, 3, 4],  # RM category
        "System_Exploitation": [5, 6, 7, 8, 9],  # SE category
        "Trust_Violation": [10, 11, 12, 13, 14],  # TV category
        "Positive Digital_Citizenship": [15, 16],  # DC category
        "Negative Digital_Citizenship": [17],  # DC category
    }


def process_data_for_analysis(human_df, ai_model_data, categories):
    """
    Process and organize data for statistical analysis.

    Parameters:
    human_df (DataFrame): Human survey responses
    ai_model_data (dict): Dictionary containing AI model data
    categories (dict): Category to question mapping

    Returns:
    dict: Processed data organized by category
    """
    processed_data = {}

    # Process each category
    for category, questions in categories.items():
        # Initialize category data
        category_data = {"Human": []}
        # Add models to category_data
        for model_name in ai_model_data.keys():
            category_data[model_name] = []

        # Process human data
        for q in questions:
            column = human_df.columns[q - 1]
            if column in human_df.columns:
                category_data["Human"].extend(human_df[column].dropna().tolist())

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

        # Perform ANOVA
        groups = [scores for scores in data.values()]
        f_stat, p_val = stats.f_oneway(*groups)

        # Perform Tukey's HSD
        tukey = pairwise_tukeyhsd(all_data, group_labels)

        # Store results
        results[category] = {
            "f_statistic": f_stat,
            "p_value": p_val,
            "tukey_results": tukey,
            "means": {group: np.mean(scores) for group, scores in data.items()},
            "std": {group: np.std(scores) for group, scores in data.items()},
        }

    return results


def visualize_results(processed_data, category):
    """
    Create visualization for a category's results.

    Parameters:
    processed_data (dict): Processed data for the category
    category (str): Category name
    """
    # Prepare data for plotting
    plot_data = []
    for group, scores in processed_data[category].items():
        for score in scores:
            plot_data.append({"Group": group, "Score": score})

    df = pd.DataFrame(plot_data)

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Group", y="Score", data=df)
    plt.title(f"Distribution of Scores for {category}")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Define file paths for each model
    model_data_paths = {
        "GPT4o": {
            "raw": "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/raw_data/contemp_GPT-4o_raw_data.csv",
            "avg": "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/averages/contemp_GPT-4o_averages.csv",
        },
        "GPT4o-mini": {
            "raw": "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/raw_data/contemp_GPT-4o-mini_raw_data.csv",
            "avg": "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/averages/contemp_GPT-4o-mini_averages.csv",
        },
        "Grok": {
            "raw": "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/raw_data/contemp_Grok_raw_data.csv",
            "avg": "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/averages/contemp_Grok_averages.csv",
        },
    }

    HUMAN_SURVEY_PATH = "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/survey_data/CES-eCommerce-variation_12.01.csv"
    QUESTIONNAIRE_PATH = "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/contemporary_CES.md"

    # Load data
    print("Loading data...")
    ai_model_data = load_ai_data(model_data_paths)
    human_df = load_human_data(HUMAN_SURVEY_PATH)

    # Get category definitions
    categories = get_category_questions()

    # Process data
    print("Processing data...")
    processed_data = process_data_for_analysis(human_df, ai_model_data, categories)

    # Run analysis
    print("Running statistical analysis...")
    results = analyze_all_categories(processed_data)

    # Print results and create visualizations
    for category in categories.keys():
        print(f"\nResults for {category}:")
        print("-" * 50)
        print(f"ANOVA Results:")
        print(f"F-statistic: {results[category]['f_statistic']:.4f}")
        print(f"p-value: {results[category]['p_value']:.4f}")
        print("\nGroup Means:")
        for group, mean in results[category]["means"].items():
            std = results[category]["std"][group]
            print(f"{group}: {mean:.4f} (±{std:.4f})")
        print("\nTukey's HSD Results:")
        print(results[category]["tukey_results"])

        # Create visualization
        # visualize_results(processed_data, category)
