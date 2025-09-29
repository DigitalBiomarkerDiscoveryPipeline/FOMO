import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class MissingPatternPlot:
    """wrapper class to draw a heatmap of missingness for a given matrix"""
    sort = ['person_id', 'missingness', 'cluster']
    direction = ['increasing', 'decreasing']

    def __init__(self, data, column_to_query, study_period, cluster):
        # check if data is a dataframe
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            plot_data = np.asarray(data)
            self.data = pd.DataFrame(plot_data)

        # check if column_to_query is valid
        if isinstance(column_to_query, str):
            self.column_to_query = column_to_query
        elif column_to_query is None:
            self.column_to_query = None
        else:
            raise ValueError("column_to_query must be a string or None")
        
        # study period
        self.study_period = study_period
        self.cluster = cluster

    @classmethod
    def initialize(cls, data, column_to_query, study_period, cluster):
      """function to initialize the class and parse the data"""
      plot_data = cls(data, column_to_query, study_period, cluster)
      plot_data.parse_data(plot_data.data, plot_data.column_to_query) # add study period later

      return plot_data

    def parse_data(self, data, column_to_query):
        """function to parse the data and get the column to query if needed"""

        if isinstance(data.columns, pd.MultiIndex) and column_to_query is None:
            raise ValueError("column_to_query needed for multi-index dataframe")

        # check pandas df type and get the column according to the column_to_query
        if isinstance(data.columns, pd.MultiIndex) and column_to_query in data.columns.levels[0]:
            self.data = data.xs(column_to_query, level=0, axis=1)
        elif isinstance(data.columns, pd.MultiIndex) and column_to_query in data.columns.levels[1]:
            self.data = data.xs(column_to_query, level=1, axis=1)
        else:
            if column_to_query is None:
                self.data = data
            elif column_to_query in data.columns:
                self.data = data[column_to_query]
            else:
                raise ValueError("column_to_query must be a valid column name")
            
        self.data.to_csv('curr_data.csv')
    
    def calc_missingness(self, direction):
        """function to calculate the missingness of the given person_id"""
        # make a copy of the dataframe
        plot_data = self.data.copy()

        # count the number of true inputs for each row
        true_count = self.data.astype(bool).sum(axis=1)
        total_inputs = len(self.data.columns)
        true_prop = true_count / total_inputs
        
        # add the true_prop to the dataframe
        self.data['true_prop'] = true_prop

        # sort the dataframe by true_prop
        plot_data = self.data.sort_values(by='true_prop', ascending=direction)

        # drop the true_prop column
        plot_data = plot_data.drop('true_prop', axis=1)

        return plot_data

    def sort_by_cluster(self, cluster, int_to_str_labels, cluster_n):
        # make a copy of the dataframe
        plot_data = self.data.copy()
        
        # add the cluster vector as a new column to the dataframe
        if isinstance(cluster, pd.Series):
            plot_data['cluster'] = cluster.to_numpy()
        else:
            plot_data['cluster'] = cluster

        # sort the df by the 'cluster' column
        df_sorted = plot_data.sort_values(by='cluster')


        # cluster column
        cluster_column = df_sorted['cluster']

        # get index of change in cluster_column values
        cluster_change = np.where(cluster_column.diff().fillna(0) != 0)[0]
        cluster_change = np.insert(cluster_change, 0, 0)
        cluster_change = [int(i) for i in cluster_change]
        cluster_change.append(len(cluster_column))
        
        # get midpoints between values in cluster_change
        cluster_midpoints = [(cluster_change[i] + cluster_change[i+1])//2 for i in range(len(cluster_change)-1)]

        # get cluster labels
        cluster_labels = df_sorted['cluster'].unique()
        if int_to_str_labels:
            cluster_labels = [letter for i, letter in zip(range(len(cluster_labels)), ['A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L','M','N','O','P','Q','R','S','T','U','V'])][::-1]

        cluster_labels = ['Cluster ' + str(i) for i in cluster_labels]
        
        if cluster_n:
            cluster_counts = df_sorted['cluster'].value_counts().sort_index()
            cluster_labels = [original_label + f'\n{count}' for original_label, count in zip(cluster_labels, cluster_counts)]

        # drop cluster column
        df_sorted = df_sorted.drop(columns='cluster')

        # make sure the column headers are datetime objects
        if not isinstance(df_sorted.columns, pd.DatetimeIndex):
            df_sorted.columns = pd.to_datetime(df_sorted.columns)

        return df_sorted, cluster_change, cluster_midpoints, cluster_labels

        df_sorted
    
    def plot(self, sort=None, direction=True, x_label=True, y_label=True, x_ticks='auto', y_ticks='auto', title=None, cmap='plasma', color_bar=False, fig_size=(15, 15), ax=None,
            int_to_str_labels=False, cluster_n=False):
        """function to plot the missingness heatmap of the given column"""
        """
        Parameters
        ----------
        df : dataframe (T/F inputs)
            dataframe with T/F values with person_id on the y axis and time on the x axis
        column_to_query: str
            name of column in df to check for missingness
        study_period: 
            time period of the study
        sort: str
            column to sort the data by
        direction: bool
            direction to sort the data by
        x_label: bool or str
            label for the x axis
        y_label: bool or str
            label for the y axis
        x_ticks: bool or list
            ticks for the x axis
            if auto: try to densely plot non-overlapping labels
            if true: use the column names, if false: no ticks
            if list: use the list
            if int: take every nth label from the column names
        y_ticks: bool or list
            ticks for the y axis
            if auto: try to densely plot non-overlapping labels
            if true: use the row index, if false: no ticks
            if list: use the list
            if int: take every nth label from the column names
        title: str
            title for the plot
        cmap: str
            color map for the plot
        color_bar: bool
            whether to show the color bar
        fig_size: tuple
            size of the plot
        """

        # PLOT HEURISTICS
        # data for the plot
        plot_data = self.data
        cluster_change = None
        cluster_labels = None
        cluster_midpoints = None
        # sort the data by sort and direction
        if sort is None:
            plot_data = self.data
        elif sort in self.sort and isinstance(direction, bool):
            if sort == 'person_id':
                plot_data = self.data.sort_index(axis=0, ascending=direction)
            elif sort == 'missingness':
                plot_data = self.calc_missingness(direction)
            elif sort == 'cluster':
                plot_data, cluster_change, cluster_midpoints, cluster_labels = self.sort_by_cluster(self.cluster, int_to_str_labels, cluster_n)
        else:
            raise ValueError("sort and direction must be valid inputs")

        # labels on y and x axis
        plot_x_label = None
        plot_y_label = None
        if isinstance(x_label, bool): 
            plot_x_label = self.data.columns.name if x_label else None
        elif isinstance(x_label, str):
            plot_x_label = x_label
        else:
            raise ValueError("x_label must be a bool or string")
        
        if isinstance(y_label, bool):
            plot_y_label = 'person_id' if y_label else None
        elif isinstance(y_label, str):
            plot_y_label = y_label
        else:
            raise ValueError("y_label must be a bool or string")
        
        # ticks on y and x axis
        plot_x_ticks = None
        plot_y_ticks = None
        if isinstance(x_ticks, bool):
            plot_x_ticks = plot_data.columns if x_ticks else False
        elif isinstance(x_ticks, list) or isinstance(x_ticks, int):
            plot_x_ticks = x_ticks
        elif isinstance(x_ticks, str) and x_ticks == 'auto':
            plot_x_ticks = x_ticks
        else:
            raise ValueError("x_ticks must be a bool, list, or integer")

        # plot the heatmap
        if ax is None:
            fig = plt.figure(figsize=fig_size)
            ax = plt.gca()
        else:
            fig = plt.gcf()
        cax = ax.pcolormesh(plot_data.columns.values, np.arange(len(plot_data.index)), plot_data.to_numpy(), cmap=cmap)
        if color_bar:
            fig.colorbar(cax, aspect=30, shrink=0.75)
#         ax = sns.heatmap(plot_data, cbar=color_bar, xticklabels=plot_x_ticks, yticklabels=plot_y_ticks, cmap=cmap)
    
        if isinstance(y_ticks, bool):
            if not y_ticks:
                ax.set_yticks([])
                ax.set_yticklabels([])
#             plot_y_ticks = plot_data.index if y_ticks else False
        elif isinstance(y_ticks, list) or isinstance(y_ticks, int):
            plot_y_ticks = y_ticks
        elif isinstance(y_ticks, str) and y_ticks == 'auto':
            plot_y_ticks = y_ticks
        else:
            raise ValueError("y_ticks must be a bool, list, or integer")
    
        plt.xlabel(plot_x_label)
        plt.ylabel(plot_y_label)
        if sort == 'cluster':
            sec = ax.secondary_yaxis(location=0)
            sec.set_yticks(cluster_change, labels=[])
            sec.tick_params('y', length=45, width=2)

            sec2 = ax.secondary_yaxis(location=0)
            sec2.set_yticks(cluster_midpoints, labels=cluster_labels)
            sec2.tick_params('y', length=60, width=0)

        if plot_x_ticks == 'auto' and isinstance(plot_data.columns, pd.DatetimeIndex):
            # get the earliest and latest date from the data
            earliest = plot_data.columns.min()
            latest = plot_data.columns.max()


            # get the number of days between the first and last date
            num_days = abs((latest - earliest).days)
            
            # set the formatting of x axis based on range
            formatter = None
            locator = None
            if num_days < 2:
                locator = mdates.HourLocator(interval=2)
                formatter = mdates.DateFormatter('%H:%M')
            elif num_days < 10:
                locator = mdates.HourLocator(interval=6)
                formatter = mdates.DateFormatter('%b-%d %H:%M')
            elif num_days < 30:
                locator = mdates.DayLocator()
                formatter = mdates.DateFormatter('%b-%d')
            elif num_days < 90:
                locator = mdates.WeekdayLocator()
                formatter = mdates.DateFormatter('%m-%d')
            elif num_days < 180:
                locator = mdates.WeekdayLocator(byweekday=mdates.SU, interval=1)
                formatter = mdates.DateFormatter('%a %b-%d')
            elif num_days < 365:
                locator = mdates.MonthLocator()
                formatter = mdates.DateFormatter('%b')
            else:
                locator = mdates.YearLocator()
                formatter = mdates.DateFormatter('%Y')
            
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        ax.tick_params(left=True, bottom=True)
        if isinstance(title, str):
            ax.set_title(f"{title}")
        elif self.column_to_query:
            ax.set_title(f"Missingness Plot for {self.column_to_query}")
        else:
            ax.set_title(f"Missingness Plot")
        
        return ax