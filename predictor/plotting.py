import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

class DataPlotter:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'plotting_output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_projects(self, ada_projects, last_n_days=None):

        fig, axs = plt.subplots(len(ada_projects), 1, figsize=(10, 15))

        if len(ada_projects) == 1:
            axs = [axs]

        if last_n_days:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=last_n_days)

        for idx, project in enumerate(ada_projects):

            if last_n_days:
                project = project[(project.index >= start_date) & (project.index <= end_date)]

            axs[idx].plot(project.index, project['DoneTicketsCount'], label='DoneTicketsCount')
            axs[idx].plot(project.index, project['FlowTicketsCount'], label='FlowTicketsCount')

            axs[idx].set_title(f'ada_project_{idx + 1}')

            axs[idx].set_xlabel('Date')
            axs[idx].set_ylabel('Tickets Count')

            axs[idx].legend()
            axs[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'projects_plot.png'))
        plt.close(fig)


