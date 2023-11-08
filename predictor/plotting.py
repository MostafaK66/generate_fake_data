import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataPlotter:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "plotting_output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_projects(self, ada_projects, last_n_days=None):
        fig, axs = plt.subplots(len(ada_projects) * 2, 1, figsize=(10, 15), sharex=True)

        for idx, project in enumerate(ada_projects):
            ax_done = axs[idx * 2]
            ax_flow = axs[idx * 2 + 1]

            if last_n_days:
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.Timedelta(days=last_n_days)
                project = project[
                    (project.index >= start_date) & (project.index <= end_date)
                ]

            split_idx = int(len(project) * 0.8)

            ax_done.plot(
                project.index[:split_idx],
                project["DoneTicketsCount"][:split_idx],
                label="DoneTicketsCount (Train)",
                color="blue",
            )
            ax_done.plot(
                project.index[split_idx:],
                project["DoneTicketsCount"][split_idx:],
                label="DoneTicketsCount (Test)",
                color="green",
            )

            ax_flow.plot(
                project.index[:split_idx],
                project["FlowTicketsCount"][:split_idx],
                label="FlowTicketsCount (Train)",
                color="orange",
            )
            ax_flow.plot(
                project.index[split_idx:],
                project["FlowTicketsCount"][split_idx:],
                label="FlowTicketsCount (Test)",
                color="red",
            )

            ax_done.set_title(f"ada_project_{idx + 1} Done Tickets")
            ax_flow.set_title(f"ada_project_{idx + 1} Flow Tickets")

            ax_done.set_ylabel("Done Tickets Count")
            ax_flow.set_ylabel("Flow Tickets Count")

            ax_done.legend()
            ax_flow.legend()

            ax_done.tick_params(axis="x", rotation=45)
            ax_flow.tick_params(axis="x", rotation=45)

        axs[-1].set_xlabel("Date")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "projects_plot.png"))
        plt.close(fig)

    def plot_actual_vs_predicted(
        self, ada_projects, walk_forward_validation_results, split_ratio, n_in
    ):
        for key, value in walk_forward_validation_results.items():
            results_df = value["ResultsDF"]

            project_number = int(key.split("_")[-1].replace("df", "")) - 1
            project_df = ada_projects[project_number]

            split_idx = int(len(project_df) * split_ratio)

            test_timestamps = project_df.index[split_idx + n_in :]
            if len(test_timestamps) != len(results_df):
                min_length = min(len(test_timestamps), len(results_df))
                results_df = results_df.iloc[:min_length]
                test_timestamps = test_timestamps[:min_length]

            results_df.set_index(test_timestamps, inplace=True)

            plt.figure(figsize=(10, 5))

            plt.plot(
                results_df.index,
                results_df["Actual"],
                label="Actual (Test)",
                color="blue",
                marker="o",
            )
            plt.plot(
                results_df.index,
                results_df["Predicted"],
                label="Predicted (Test)",
                color="red",
                linestyle="--",
                marker="x",
            )

            plt.title(f"Actual vs Predicted - {key}")
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Ticket Count")
            plt.grid(True)
            plt.xticks(rotation=15)

            plot_path = os.path.join(self.output_dir, f"{key}_actual_vs_predicted.png")
            plt.savefig(plot_path)
            plt.close()
