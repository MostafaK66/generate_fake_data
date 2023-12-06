import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_csv_to_dataframe(
    file_path=r"C:\Users\mostafa.karimzadeh\Desktop\RWS\EU Jira 2023-12-05T12 51 07+0100 1.csv",
):
    df = pd.read_csv(file_path)
    return df


def assign_project(df):
    if "Issue key" not in df.columns:
        return "The DataFrame does not have an 'Issue key' column."

    def determine_project(issue_key):
        if "GIR" in issue_key:
            return "GIR"
        elif "REV" in issue_key:
            return "REV"
        else:
            return "Unknown"

    df["Project"] = df["Issue key"].apply(determine_project)

    return df


def create_missing_data_heatmaps_per_issue_type(df):
    # Create 'gap_plots' directory if it doesn't exist
    if not os.path.exists("gap_plots"):
        os.makedirs("gap_plots")

    # Columns to exclude
    exclude_columns = [
        "Labels.1",
        "Labels.2",
        "Labels.3",
        "Sprint.1",
        "Sprint.2",
        "Sprint.3",
        "Sprint.4",
        "Sprint.5",
        "Issue key",
    ]

    # Columns to highlight in red
    highlight_columns = [
        "Fix Version/s",
        "Affects Version/s",
        "Sprint",
        "Custom field (Epic Link)",
        "Custom field (Story Points)",
    ]

    # Filter data for each project and issue type
    for project in ["GIR", "REV"]:
        project_df = df[df["Project"] == project]

        # Handle "Epic" and "Story" separately, and group other types
        for issue_type in ["Epic", "Story", "Others"]:
            if issue_type != "Others":
                type_df = project_df[project_df["Issue Type"] == issue_type]
            else:
                type_df = project_df[~project_df["Issue Type"].isin(["Epic", "Story"])]

            # Exclude specified columns
            type_df = type_df.drop(columns=exclude_columns, errors="ignore")

            # Create a heatmap for missing data
            plt.figure(figsize=(20, 20))
            sns_heatmap = sns.heatmap(type_df.isnull(), cbar=False, yticklabels=False)

            # Set title and labels with increased font size
            plt.title(
                f"Missing Data Heatmap for Project {project}, Issue Type: {issue_type}",
                fontsize=20,
            )
            plt.xlabel("Columns", fontsize=16)
            plt.ylabel("Issue Keys", fontsize=16)

            # Set x-tick labels with specific columns highlighted in red
            plt.xticks(fontsize=12, rotation=90)
            for label in sns_heatmap.get_xticklabels():
                if label.get_text() in highlight_columns:
                    label.set_color("red")

            # Increase y-tick label size
            plt.yticks(fontsize=12)

            # Save the heatmap
            heatmap_filename = f"missing_data_heatmap_{project}_{issue_type}.png"
            plt.savefig(os.path.join("gap_plots", heatmap_filename))
            plt.close()


def main():
    df = read_csv_to_dataframe()
    df = assign_project(df)
    create_missing_data_heatmaps_per_issue_type(df)
    return df


if __name__ == "__main__":
    df = main()
