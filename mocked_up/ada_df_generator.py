import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd


class AdaBottleneckGenerator:
    def __init__(
        self,
        n_tickets=None,
        seed=None,
        progress_rates=None,
        days_choices=None,
        project_capacities=None,
        team_members_count=None,
    ):
        self.n_tickets = n_tickets
        self.progress_rates = progress_rates
        self.days_choices = days_choices
        self.team_members_count = team_members_count
        self.project_capacities = project_capacities
        self.status_values = ["Refined", "To Do", "In Progress", "In Review", "Done"]
        self.projects = ["ADA_Project_1", "ADA_Project_2", "ADA_Project_3"]
        self.teams = {
            "ADA_Project_1": ["ADA_Team_1", "ADA_Team_2", "ADA_Team_3"],
            "ADA_Project_2": ["ADA_Team_1", "ADA_Team_4"],
            "ADA_Project_3": ["ADA_Team_5", "ADA_Team_6"],
        }
        self.ticket_features = [f"ADA_Feature_{str(i)}" for i in range(1, 51)]
        np.random.shuffle(self.ticket_features)
        random.seed(seed)

    def assign_feature_name(self) -> str:
        """Randomly assign a feature name from the available choices."""
        return random.choice(self.ticket_features)

    def get_dataframe(self) -> pd.DataFrame:
        """Generate and return the dataframe with ticket details."""
        ticket_data = []
        start_date = datetime(2023, 1, 1).date()
        today = datetime.now().date()

        for i in range(1, self.n_tickets + 1):
            ticket_name = f"ADA_Ticket_{i}"
            ticket_project, ticket_team = self.assign_projects_and_teams()
            status_date = AdaBottleneckGenerator.random_date(start_date, today)
            ticket_created_date = self.ticket_created_date(status_date)
            feature_name = self.assign_feature_name()

            current_progress_rate = self.progress_rates[ticket_project]

            for status in self.status_values:
                if random.random() < current_progress_rate:
                    ticket_data.append(
                        [
                            ticket_name,
                            status,
                            ticket_project,
                            ticket_team,
                            status_date,
                            ticket_created_date,
                            feature_name,
                        ]
                    )
                    status_date = self.assign_status_date(
                        status_date, ticket_project, status
                    )
                else:
                    break

        df = pd.DataFrame(
            ticket_data,
            columns=[
                "TicketName",
                "TicketStatus",
                "TicketProject",
                "TicketTeam",
                "TicketStatusDate",
                "TicketCreatedDate",
                "TicketFeatureName",
            ],
        )
        df = self.assign_pi(df)
        return df

    def assign_projects_and_teams(self) -> Tuple[str, str]:
        """Randomly assign a ticket to a project and team, and return both."""
        project = random.choice(self.projects)
        team = random.choice(self.teams[project])
        return project, team

    def assign_status_date(self, current_date: date, project: str, status: str) -> date:
        """Compute the next status date based on the current date, project, and status."""

        days_list = self.days_choices.get(project, {}).get(
            status, self.days_choices[project]["Default"]
        )
        days = random.choice(days_list)

        return current_date + timedelta(days=days)

    def ticket_created_date(self, backlog_date: date) -> date:
        """Compute the ticket creation date based on the backlog date."""
        days_to_subtract = random.choice([1, 3, 5, 7, 9, 11])
        return backlog_date - timedelta(days=days_to_subtract)

    def assign_pi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign Program Increment (PI) values to each ticket based on TicketStatusDate."""

        pi_start_date = datetime(2023, 1, 1).date()

        def get_pi_value(date: date) -> str:
            """Helper function to get the corresponding PI value for a date."""
            time_diff = date - pi_start_date
            pi_number = time_diff.days // 14 + 1

            pi_major = pi_number // 10 + 1
            pi_minor = pi_number % 10

            if pi_minor == 0:
                pi_minor = 10
                pi_major -= 1

            return f"{pi_major}.{pi_minor}"

        df["PI"] = df["TicketStatusDate"].apply(get_pi_value)

        cols = df.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        df = df[cols]

        return df

    def assign_ticket_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a TicketType to each unique ticket - either Bug or Story."""
        unique_tickets = df["TicketName"].unique()

        n_bugs = int(0.4 * len(unique_tickets))
        n_stories = len(unique_tickets) - n_bugs

        ticket_types_list = ["Bug"] * n_bugs + ["Story"] * n_stories
        random.shuffle(ticket_types_list)

        ticket_type_mapping = dict(zip(unique_tickets, ticket_types_list))

        df["TicketType"] = df["TicketName"].map(ticket_type_mapping)

        return df

    def assign_ticket_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a random priority from a set list to each ticket ID."""
        priorities = ["Blocker", "Major", "Minor", "Not Blocking"]

        ticket_priority_map = {
            ticket_name: random.choice(priorities)
            for ticket_name in df["TicketName"].unique()
        }

        df["TicketPriority"] = df["TicketName"].map(ticket_priority_map)

        return df

    def generate_df_pis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a dataframe containing sorted unique PI values from the input dataframe."""
        unique_pis = df["PI"].unique()
        sorted_pis = sorted(
            unique_pis, key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1]))
        )

        df_pis = pd.DataFrame(sorted_pis, columns=["SortedPIs"])

        return df_pis.iloc[1:9]

    @staticmethod
    def random_date(start: date, end: date) -> date:
        """Generate a random date between start and end."""
        time_difference = end - start
        random_days = random.randint(0, time_difference.days)
        return start + timedelta(days=random_days)

    # TODO: maybe some improvments os scope column!!
    def add_ticket_scope(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a TicketScope column to the dataframe based on the TicketStatus column.
        """
        status_to_scope = {
            "Done": "Delivered",
            "In Progress": "Committed",
            "In Review": "Committed",
            "To Do": "Planned",
            "Refined": "Planned",
        }

        df["TicketScope"] = df["TicketStatus"].apply(
            lambda x: status_to_scope.get(x, "Unknown")
        )

        return df

    def add_team_members_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a TeamMembers column to the dataframe based on the team of the ticket."""

        team_members_count = self.team_members_count

        df["TeamMembers"] = df["TicketTeam"].map(team_members_count)

        return df

    def distribute_story_points(self, capacity: int, num_tickets: int) -> List[int]:
        if num_tickets == 0 or capacity == 0:
            return [1] * num_tickets

        fibonacci_range = [1, 2, 3, 5]
        points: List[int] = []
        remaining_capacity = capacity

        for _ in range(num_tickets):
            possible_points = [x for x in fibonacci_range if x <= remaining_capacity]
            if not possible_points:
                return points

            point = random.choice(possible_points)
            points.append(point)
            remaining_capacity -= point

        return points

    def add_story_points(self, df: pd.DataFrame) -> pd.DataFrame:
        ticket_story_points = defaultdict(int)

        project_capacities = self.project_capacities

        grouped = df.groupby(["PI", "TicketProject"])

        for (pi, project), group in grouped:
            num_tickets = group["TicketName"].nunique()
            story_points = self.distribute_story_points(
                project_capacities.get(project, 0), num_tickets
            )

            for ticket, point in zip(group["TicketName"].unique(), story_points):
                ticket_story_points[(pi, project, ticket)] = point

        df["TicketStoryPoint"] = df.apply(
            lambda row: ticket_story_points[
                (row["PI"], row["TicketProject"], row["TicketName"])
            ]
            or 1,
            axis=1,
        )
        return df
