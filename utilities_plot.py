import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
#from nltk.corpus import stopwords
#nltk.download('stopwords')
import string
import re
import numpy as np
import webcolors
from typing import List, Dict
import sys


def density_on_target(data: pd.DataFrame, target_name: str, column_name: str, ax: matplotlib.axes.Axes = None, fig_size: tuple = (10, 10)):
    """
    Plots the density distribution of a specified column for each unique value in the target column,
    along with a global mean line and includes the mean of each target group in the legend.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    target_name : str
        The name of the target column with categorical labels.
    column_name : str
        The name of the column for which the density plot will be generated.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object where the plot will be drawn. If None, a new figure and axes are created.
    fig_size : tuple, optional
        The size of the figure to be created if `ax` is None. Default is (10, 10).

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object containing the density plot.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    # Get unique target labels
    target_labels = data[target_name].unique()

    # Plot density for each target label and add the mean in the legend
    for label in target_labels:
        target_data = data[data[target_name] == label][column_name]
        mean_value = target_data.mean()
        sns.kdeplot(
            target_data,
            fill=True,
            ax=ax,
            label=f'{target_name} = {label} (Mean: {mean_value:.2f})'
        )

    # Plot global mean line for the column
    global_mean = data[column_name].mean()
    ax.axvline(global_mean, color='g', linestyle='--', label=f'Global Mean: {global_mean:.2f}')
    ax.text(
        global_mean * 1.05, ax.get_ylim()[1] * 0.8,
        f'Global {column_name} Mean\n{global_mean:.2f}',
        color='g', ha='center'
    )

    # Show the legend
    ax.legend()

    return ax


def histogram_on_target(data: pd.DataFrame, target_name: str, column_name: str, bins: float = 5, fig_size: tuple = (12, 6)):
    """
    Plots separate histograms for a specified column based on each unique value in the target column.
    Handles both numeric and categorical data.

    Args:
        data (pd.DataFrame): 
            The DataFrame containing the data to be plotted.
            
        target_name (str): 
            The name of the target column with categorical labels.
            
        column_name (str): 
            The name of the column for which histograms will be generated.
            
        bins (float): 
            The number of bins to use for numeric histograms. Ignored for categorical data.
            
        fig_size (tuple, optional): 
            The size of the figure to be created. Default is (12, 6).

    Returns:
        fig (matplotlib.figure.Figure): 
            The figure object containing the histograms.
            
        axes (array of matplotlib.axes.Axes): 
            The array of axes objects for each subplot.
    """

    # Get unique target labels
    target_labels = data[target_name].unique()

    # Determine if the column is numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(data[column_name])

    # Create subplots for each target label
    fig, axes = plt.subplots(1, len(target_labels), figsize=fig_size, sharey=True)
    fig.suptitle(f'Histogram of {column_name} for each {target_name} label')

    # Ensure axes is iterable if there's only one target label
    if len(target_labels) == 1:
        axes = [axes]

    # Generate histogram or countplot for each target label
    for i, label in enumerate(target_labels):
        target_data = data[data[target_name] == label][column_name]
        
        if is_numeric:
            # Plot histogram for numeric data
            axes[i].hist(target_data, bins=bins, color='skyblue', edgecolor='black', density=True, alpha=0.7)
            mean_value = target_data.mean()
            axes[i].set_title(f'{target_name} = {label}\nMean: {mean_value:.2f}')
        
        else:
            # Plot countplot for categorical data
            sns.countplot(x=target_data, ax=axes[i], color="skyblue")
            axes[i].set_title(f'{target_name} = {label}')

        # Set labels
        axes[i].set_xlabel(column_name)
        if i == 0:
            axes[i].set_ylabel('Density' if is_numeric else 'Count')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

    return axes

def boxplot_on_target(data: pd.DataFrame, target_name: str, column_name: str, fig_size: tuple = (12, 6)):
    """
    Plots separate box plots for a specified column based on each unique value in the target column, 
    with annotations for extreme values and mean.

    Args:
    -----
        data (pd.DataFrame): The DataFrame containing the data to be plotted.
        target_name (str): The name of the target column with categorical labels.
        column_name (str): The name of the column for which box plots will be generated.
        fig_size (tuple, optional): The size of the figure to be created. Default is (12, 6).

    Returns:
    --------
        fig (matplotlib.figure.Figure): The figure object containing the box plots.
        ax (matplotlib.axes.Axes): The axes object for the plot.
    """
    # Determine if the column is numeric
    if not pd.api.types.is_numeric_dtype(data[column_name]):
        raise ValueError(f"Column '{column_name}' must be numeric to create box plots.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    fig.suptitle(f'Box plot of {column_name} for each {target_name} label')

    # Plot box plot
    sns.boxplot(x=target_name, y=column_name, data=data, ax=ax, palette="pastel")

    # Annotate the mean, min, and max values for each box plot
    for label in data[target_name].unique():
        # Get data for the current target label
        target_data = data[data[target_name] == label][column_name]
        mean_value = target_data.mean()
        min_value = target_data.min()
        max_value = target_data.max()
        
        # Get x position for each box based on the label's position
        x_position = list(data[target_name].unique()).index(label)

        # Plot mean as a dashed line
        ax.plot([x_position - 0.3, x_position + 0.3], [mean_value, mean_value], 'g--', label=f'Mean' if label == data[target_name].unique()[0] else "")

        # Annotate mean, min, and max
        ax.text(x_position, mean_value, f'{mean_value:.2f}', color='green', ha='center', va='bottom')
        ax.text(x_position, min_value, f'{min_value:.2f}', color='blue', ha='center', va='top')
        ax.text(x_position, max_value, f'{max_value:.2f}', color='red', ha='center', va='bottom')

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='g', linestyle='--', label='Mean'))
    ax.legend(handles=handles)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return fig, ax


def display_classes_repartition(data, var1, var2):
    """The goal of this function is to display a density graph to see the repartition of one label's value
       depending on the value we want to predict.

    Args:
        data (pd.DataFrame): the dataframe we want to visualize
        var1 (str): the label we want to see the distribution
        var2 (str): the label we will predict in the future
    """
    
    # We take the unique labels of var2
    classes = data[var2].unique()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # We create the figure and the axes
    fig, axes = plt.subplots(nrows=1, ncols=len(classes), figsize=(14, 6), sharey=True)
    
    if len(classes) == 1:
        axes = [axes]
    
    for ax, clas, color in zip(axes, classes, colors):
        # We extract the data depending on the distinct label of var2
        subset = data[data[var2] == clas]
        
        # Calculate the distributions of var1
        counts_total = data[var1].value_counts().sort_index()
        counts_class = subset[var1].value_counts().sort_index()
        
        # Plot the total distribution
        ax.fill_between(counts_total.index, counts_total.values, color='grey', alpha=0.5, label='all people surveyed')
        
        # Plot the class-specific distribution
        ax.fill_between(counts_class.index, counts_class.values, color=color, alpha=0.5, label=f'highlighted group {var2}={clas}')
        
        # Mean value of the class
        mean_value = subset[var1].mean()
        
        # Add the mean to the graph
        ax.axvline(mean_value, color='g', linestyle='--', label=f'Mean: {mean_value:.2f}')
        
        ax.legend()
        
        ax.set_xlabel(var1)
        ax.set_ylabel('Count')
        ax.set_title(f'{var1} Distribution for {var2} {clas}')
    
    plt.tight_layout()
    plt.show()
    
    
def density_on_target(data: pd.DataFrame, target_name: str, column_name: str, ax: matplotlib.axes.Axes = None, fig_size: tuple = (10, 10)):
    """
    Plots the density distribution of a specified column for each unique value in the target column,
    along with a global mean line and includes the mean of each target group in the legend.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    target_name : str
        The name of the target column with categorical labels.
    column_name : str
        The name of the column for which the density plot will be generated.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object where the plot will be drawn. If None, a new figure and axes are created.
    fig_size : tuple, optional
        The size of the figure to be created if `ax` is None. Default is (10, 10).

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object containing the density plot.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    # Get unique target labels
    target_labels = data[target_name].unique()

    # Plot density for each target label and add the mean in the legend
    for label in target_labels:
        target_data = data[data[target_name] == label][column_name]
        mean_value = target_data.mean()
        sns.kdeplot(
            target_data,
            fill=True,
            ax=ax,
            label=f'{target_name} = {label} (Mean: {mean_value:.2f})'
        )

    # Plot global mean line for the column
    global_mean = data[column_name].mean()
    ax.axvline(global_mean, color='g', linestyle='--', label=f'Global Mean: {global_mean:.2f}')
    ax.text(
        global_mean * 1.05, ax.get_ylim()[1] * 0.8,
        f'Global {column_name} Mean\n{global_mean:.2f}',
        color='g', ha='center'
    )

    # Show the legend
    ax.legend()

    return ax
        
# Bar graph visualization  : 
def plot_barh_graph(values, labels, name_value, name_labels, color, ax):

    """ The goal of this function is to display a bar graph, with the option
    Args:
        values (list): list of value we want to display
        labels (list): list of the label's values
        name_value (str): label for the y axis
        name_labels (_type_): label for the x axis
        color (list): color description
        ax (plt): subplot where we want to display the fig

    Returns:
        fig : plt fig
        ax : plt axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 15))

    else:
        fig = ax.figure
        
    # Determine the colors for the bars
    if isinstance(color, str):
        try:
            # Try to convert web color name to hex
            color = webcolors.name_to_hex(color)
            colors = [color for _ in range(len(labels))]
            
        except ValueError:
            
            colormap = plt.get_cmap(color)
            colors = [colormap(i / len(values)) for i in range(len(values))]

    elif isinstance(color, (matplotlib.colors.LinearSegmentedColormap, matplotlib.colors.ListedColormap)):
        colors = [color(i / len(values)) for i in range(len(values))]

    else:
        colors = color

    data = pd.DataFrame({'values': values, 'labels': labels})
    
    
    # Sort the data
    data = data.sort_values('values', ascending=False)  # Ascending for horizontal bars
    
    # Plot with seaborn
    sns.barplot(x='values', y='labels', data=data, palette=colors, ax=ax, edgecolor='black', linewidth=1.5)

    # Add values at the end of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.2f'), 
                   (p.get_width(), p.get_y() + p.get_height() / 2.), 
                   ha='left', va='center', fontsize = 8,
                   xytext=(5, 0), 
                   textcoords='offset points')

    # Plot a dashed line for the maximum value
    max_value = max(values)
    ax.axvline(x=max_value, color='red', linestyle='--', label='Max Value')
    
    ax.legend(bbox_to_anchor=(0.95, 0.07))
    
    # Set xlabel and ylabel
    ax.set_xlabel(name_value)
    ax.set_ylabel(name_labels)
    
    # Set the title
    ax.set_title('Bar graph with max value')
    
    return ax

def plot_bar_graph(values, labels, name_value, name_labels, color, ax=None):
    """ The goal of this function is to display a vertical bar graph using sns.barplot.
    
    Args:
        values (list): list of values we want to display
        labels (list): list of the label's values
        name_value (str): label for the y axis
        name_labels (str): label for the x axis
        color (str or list): color description, either a seaborn palette name or a list of colors
        ax (plt.Axes, optional): subplot where we want to display the fig. Defaults to None.

    Returns:
        plt.Axes: The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(40, 20))

    else:
        fig = ax.figure
        
    # Determine the colors for the bars
    if isinstance(color, str):
        try:
            # Try to convert web color name to hex
            color = webcolors.name_to_hex(color)
            colors = [color for _ in range(len(labels))]
            
        except ValueError:
            
            colormap = plt.get_cmap(color)
            colors = [colormap(i / len(values)) for i in range(len(values))]

    elif isinstance(color, (matplotlib.colors.LinearSegmentedColormap, matplotlib.colors.ListedColormap)):
        colors = [color(i / len(values)) for i in range(len(values))]

    else:
        colors = color
        
    # Sort the values and labels in descending order
    sorted_indices = np.argsort(values)[::1]
    values = [values[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    # Create a DataFrame for Seaborn
    data = pd.DataFrame({'values': values, 'labels': labels})
    
    # Sort the data
    data = data.sort_values('values', ascending=False)  # Sort in descending order
    
    # Plot with seaborn
    sns.barplot(x='labels', y='values', data=data, palette=colors, ax=ax, edgecolor='black', linewidth=1.5)

    # Add values on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 9), fontsize = 8,
                   textcoords='offset points')

    # Plot a dashed line for the maximum value
    max_value = max(values)
    ax.axhline(y=max_value, color='red', linestyle='--', label='Max Value')
    
    ax.legend(bbox_to_anchor=(0.95, 0.07))
    
    # Set xlabel and ylabel
    ax.set_xlabel(name_labels)
    ax.set_ylabel(name_value)
    
    # Set the title
    ax.set_title('Bar graph with max value')
    
    return ax

def plot_repartition_proportionnelle(values, classes, colors, leg):
    
    # Configuration du graphique
    bar_width = 0.5
    index = np.arange(len(classes))

    # Créer les barres empilées
    plt.figure(figsize=(10, 6))

    bottoms = np.zeros(len(classes))

    for j in range(len(values)):
        plt.bar(index, values[j], bar_width, bottom=bottoms, label=leg[j], color=colors[j])
        bottoms += values[j]


    # Ajouter les étiquettes pour les valeurs
    for i in range(len(classes)):
        pos = 0
        for j in range(len(values)):
            pos += values[j][i]/2 - 20
            # for the last we put the value on the far top of the bar
            if j != len(values)-1:
                plt.text(index[i], pos, str(values[j][i]), ha='center', color='black', fontsize=12)
                pos += values[j][i]/2
            else:
                pos += values[j][i]/2 + 60
                plt.text(index[i], pos, str(values[j][i]), ha='center', color='black', fontsize=12)

    # Ajouter les détails du graphique
    plt.xlabel('Filtres')
    plt.ylabel('Amounts')
    plt.title('Plot proportion repartition')
    plt.xticks(index, classes)
    plt.legend(bbox_to_anchor=(0.7, 0.10))

    plt.xticks(rotation=65)
    # Afficher le graphique
    plt.show()

def plot_multiple_barv(datas, color, ax=None):
    """
    The goal of this function is to plot a double bar. Data 1, 2 and label need to have the same length.

    Args:
        datas (list): 
            dataframe with value we want to multiplot
            columns = name of catagories
            index = name of the filters

        colors (list of str): 
            list of color for the list of values

        ax (plt.Axes): 
            axes we want to plot this graph

    Returns:
        ax : the plt axes
    """
    column_names = datas.columns.tolist()
    index_name = datas.index.tolist()
    print(datas.index.tolist())

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 20))
    else:
        fig = ax.figure

            # Determine the colors for the bars
    if isinstance(color, str):
        try:
            # Try to convert web color name to hex
            color = webcolors.name_to_hex(color)
            colors = [color for _ in range(len(column_names))]
            
        except ValueError:
            
            colormap = plt.get_cmap(color)
            colors = [colormap(i / len(column_names)) for i in range(len(column_names))]

    elif isinstance(color, (matplotlib.colors.LinearSegmentedColormap, matplotlib.colors.ListedColormap)):
        colors = [color(i / len(column_names)) for i in range(len(column_names))]

    else:
        colors = color

    # Bar position
    indices = np.arange(len(index_name))
    bar_width = 0.8 / len(index_name)

    for j, col in enumerate(column_names):
        data  = datas[col].tolist()
        color = colors[j]
        bars  = ax.bar(indices + (j - len(index_name) / 2) * bar_width, data, bar_width, label=col, color= color)
 
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height>0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
    print(index_name)
    print(indices)
    ax.set_xticks(indices)
    ax.set_xticklabels(index_name, rotation= 20)
    ax.legend()

    plt.show()
    return fig, ax


def plot_multiple_barh(datas, columns, labels, color, ax):
    """
    The goal of this function is to plot a double barh. Data 1, 2 and label need to have the same length.

    Args:
        datas (list) : list of list values
        columns (str) : name of the 2 vars compared
        labels (list) : label list
        colors (str) : list of color for the first list of values
        ax (plt.fidure) : figure we want to plot this graph

    Returns:
        ax : the plt figure
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(20,20))
    else:
        fig = ax.figure

        # Determine the colors for the bars
    if isinstance(color, str):
        try:
            # Try to convert web color name to hex
            color = webcolors.name_to_hex(color)
            colors = [color for _ in range(len(labels))]
            
        except ValueError:
            
            colormap = plt.get_cmap(color)
            colors = [colormap(i / len(labels)) for i in range(len(labels))]

    elif isinstance(color, (matplotlib.colors.LinearSegmentedColormap, matplotlib.colors.ListedColormap)):
        colors = [color(i / len(labels)) for i in range(len(labels))]

    else:
        colors = color

    # bar position
    indices = np.arange(len(labels))
    bar_width = 0.5
    bar_width = 0.8 / len(datas)

    for i in range(len(datas)):
        data = datas[i]
        name = columns[i]
        color = colors[i]

        bars = ax.barh(indices + (i - len(datas)/2) * bar_width, data, bar_width, label=name, color=color)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')


    ax.set_yticks(indices)
    ax.set_yticklabels(labels)
    ax.legend()

    plt.show()
    return fig, ax


def get_isna_proportion_censored_label(df_, colors, ax,  highlight_label = {"Summary_Description_Description_PROJECT": "Project Description",
                                                                        "Objective_CASE": "Project Case",
                                                                        "StepDescription_StepDescription_STEP": "Project Step"}):
    """ the goal of this function is to display the missing value graph with color for step, case and project and
        also with the name of the columns hide except for the one highlighted

    Args:
        df_ (pd.DataFrame) : the dataframe we want to know the repartition of the missing value for each columns

    Returns:
        None
    
    """
    values = [round(i, 3) for i in (df_.isna().sum()/len(df_)).to_list()]
    labels = df_.isna().columns.to_list()
    #labels = [f"columns{i}" for i in range(len(values))]

    # display na distribution
    ax = plot_barh_graph(values, 
                        labels,
                        'Missing values', 'Columns', color='paleturquoise', ax=ax)

    plt.title(f'Missing value for each columns for project 2')

    highlight_label = {"Description_Description_PROJECT": "Project Description",
                        "Objective_CASE": "Project Case",
                        "StepDescription_StepDescription_STEP": "Project Step"}
        
    new_labels = []
    label_color = []

    for text in labels:
        if text in highlight_label.keys():
            new_labels.append(highlight_label[text])

            if '_STEP' in text:
                label_color.append(colors[0])
            elif '_CASE' in text:
                label_color.append(colors[1])
            elif '_PROJECT' in text:
                label_color.append(colors[2])
        else:
            new_labels.append("****")

            if '_STEP' in text:
                label_color.append(colors[0])
            elif '_CASE' in text:
                label_color.append(colors[1])
            elif '_PROJECT' in text:
                label_color.append(colors[2])

    ax.set_yticks(range(len(new_labels)))
    ax.set_yticklabels(new_labels)

    for tick_label, color in zip(ax.get_yticklabels(), label_color):
        tick_label.set_color(color)

    # Create custom legend for colors
    custom_legend = [
            plt.Line2D([0], [0], color=colors[0], lw=4, label='STEP'),
            plt.Line2D([0], [0], color=colors[1], lw=4, label='CASE'),
            plt.Line2D([0], [0], color=colors[2], lw=4, label='PROJECT')
        ]

    ax.legend(handles=custom_legend, bbox_to_anchor=(0.83, 0.99))
    plt.show()

def dipslay_length(values, labels, title_desc, colors):
    """The goal of this function is to show the barh plot of the count word repartition 

    Args:
        desc (str): name of the columns description
        title_desc (str): name for the title graph comprehension
        colors (str): the color we want to apply
        df_kpi_avi (pd.Dataframe): the dataframe that contains the data
    """

    tresh = 0.1*max(values)

    # Create a filtered_values list and filtered_labels list based on values > 0
    filtered_values = [value for value in values if value > tresh]
    filtered_labels = [label for value, label in zip(values, labels) if value > tresh]

    plot_barh_graph(filtered_values, filtered_labels,
                    'Count words', 'Test name',
                    color=colors, ax=None)
    plt.title(f'Amount of word for each {title_desc} data cleaning')


def get_length_description(project, df, color):
    """ 
        The goal of this function is to show the count word repartition for step, case and project
        before and after data cleaning

        Args:
            project (str): 
                name of the project

            df_kpi_avi (pd.dataframe): 
                the dataset which contains the description

        Returns:
            None
    """
    params = {
                'title' : [f'{project} project description before', f'{project} case description before', f'{project} step description before'],
                'desc' :  ['Description_project', 'Description_case', 'Description_step']}

    for i in range(3):
        desc, title_desc = params['desc'][i], params['title'][i]
        values = df[desc].apply(lambda x: len(x.split()) if isinstance(x, str) else 0).to_list()
        labels = df['Test'].to_list()
        dipslay_length(values, labels, title_desc, color)


def get_case_step_repartition(df, project, colors):
    """
        Total Step
    """
    values = df.Count_step.value_counts().to_list()
    labels =  df.Count_step.value_counts().index.to_list()

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title(f'Amount of tests depending on their number of steps {project}')

    ax = barh_sort_labels(values, 
                    labels,
                    'Number of test which have that many steps', 'Number of steps', 
                    color='GnBu_r', ax=ax, prefixe= 'steps')


    """
        Total Case
    """
    values = df.Count_case.value_counts().to_list()
    labels =  df.Count_case.value_counts().index.to_list()

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title(f'Amount of tests depending on their number of cases {project}')

    ax = barh_sort_labels(values, 
                    labels,
                    'Amount of test depending on the number of cases', 'Number of cases', 
                    color='GnBu_r', ax=ax, prefixe= 'cases')

# Bar graph visualization  : 
def barh_sort_labels(values, labels, name_value, name_labels, color, ax, prefixe):

    """ The goal of this function is to display a bar graph, with the option
    Args:
        values (list): list of value we want to display
        labels (list): list of the label's values
        name_value (str): label for the y axis
        name_labels (_type_): label for the x axis
        color (list): color description
        ax (plt): subplot where we want to display the fig

    Returns:
        fig : plt fig
        ax : plt axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 15))

    else:
        fig = ax.figure
        
    # Determine the colors for the bars
    if isinstance(color, str):
        try:
            # Try to convert web color name to hex
            color = webcolors.name_to_hex(color)
            colors = [color for _ in range(len(labels))]
            
        except ValueError:
            
            colormap = plt.get_cmap(color)
            colors = [colormap(i / len(values)) for i in range(len(values))]

    elif isinstance(color, (matplotlib.colors.LinearSegmentedColormap, matplotlib.colors.ListedColormap)):
        colors = [color(i / len(values)) for i in range(len(values))]

    else:
        colors = color

    data = pd.DataFrame({'values': values, 'labels': labels})
    
    
    # Sort the data
    data = data.sort_values('labels', ascending=True)  # Ascending for horizontal bars

    # add prefixe

    data.labels = data['labels'].apply(lambda x: str(x) +' '+ prefixe )
    
    # Plot with seaborn
    sns.barplot(x='values', y='labels', data=data, palette=colors, ax=ax, edgecolor='black', linewidth=1.5)

    # Add values at the end of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.2f'), 
                   (p.get_width(), p.get_y() + p.get_height() / 2.), 
                   ha='left', va='center', fontsize = 12,
                   xytext=(5, 0), 
                   textcoords='offset points')

    # Plot a dashed line for the maximum value
    max_value = max(values)
    ax.axvline(x=max_value, color='red', linestyle='--', label='Max Value')
    
    ax.legend(bbox_to_anchor=(0.95, 0.07))
    
    # Set xlabel and ylabel
    ax.set_xlabel(name_value)
    ax.set_ylabel(name_labels)
    
    plt.show()
    
    return ax, fig


def plot_anomalies_req_tst(projects, data_global, colors, on_req):    
    filters = ['tests without requierments', 
            'requierments without descriptions', 
            'descriptions without requierments', 
            'tests without descriptions & Requirements']
    prop = []

    list_data = {}

    for proj in projects:
        list_filters = {}

        sub_data_proj = data_global[data_global.index.get_level_values('Project') == proj]

        tst_no_req = sub_data_proj[sub_data_proj[on_req].isnull()]['Test'].unique()
        req_no_desc = sub_data_proj[(sub_data_proj[on_req].notnull() & data_global['Description_req_global'].isnull())][on_req].unique()
        desc_no_req = sub_data_proj[(sub_data_proj[on_req].isnull() & data_global['Description_req_global'].notnull())]["Description_req_global"]
        tst_no_req_desc = sub_data_proj[(sub_data_proj[on_req].isnull() & data_global['Description_req_global'].isnull())]['Test'].unique()

        if len(sub_data_proj) != 0:
            # number of tests without requierments
            tst_req = len(tst_no_req)/len(sub_data_proj.Test.unique())

            # number of requierments without description
            req_desc = len(req_no_desc)/len(sub_data_proj[on_req].unique())

            # number of description without requierments
            desc_req = len(desc_no_req)/len(sub_data_proj)

            # number of test without description and requierments
            without_desc_req = len(tst_no_req_desc)/len(sub_data_proj.Test.unique())

            prop.append([tst_req, req_desc, desc_req, without_desc_req])

        else:
            prop.append([0, 0, 0, 0])

        list_filters["tst_no_req"] = tst_no_req.tolist()
        list_filters["req_no_desc"] = req_no_desc.tolist()
        list_filters["desc_no_req"] = desc_no_req.tolist()
        list_filters["tst_no_req_desc"] = tst_no_req_desc.tolist()

        list_data[proj] = list_filters

    ################################################################### tot proportion ##########################################
    tst_no_req = len(data_global[(data_global[on_req].isnull())]['Test'].unique().tolist())/len(data_global.Test.unique())
    req_no_desc = len(data_global[(data_global[on_req].notnull() & data_global['Description_req_global'].isnull())][on_req].unique().tolist())/len(data_global[on_req].unique())
    desc_no_req = len(data_global[(data_global[on_req].isnull() & data_global['Description_req_global'].notnull())]["Description_req_global"].tolist())/len(data_global)
    tst_no_req_desc = len(data_global[(data_global[on_req].isnull() & data_global['Description_req_global'].isnull())]['Test'].unique().tolist())/len(data_global.Test.unique())

    tot= [tst_no_req, req_no_desc, desc_no_req, tst_no_req_desc]

    prop.append(tot)

    fig, ax= plt.subplots(figsize=(15, 8))
    #prop.append(portion_tot)
    #prop = [[i/len(data_global) for i in p] for p in prop]
    plt.title("Info about requirements and tests in %")
    plot_multiple_barv(prop, 
                        ["E10B", "KVHTS", "EGYPT", 'TOTAL'], filters, colors, ax)
    return list_data


def plot_pie_empty_cols_repartition(df, project, colors):
    # defined colors for pie : step, case, project

    empty_cols, values = utilities_avicenne.get_fully_na(df)

    plt.pie(values, 
                labels=["STEP", "CASE", "PROJECT"], colors=colors,
                autopct='%1.1f%%', startangle=140)
    plt.title(f'Fully empty columns in TCR dataset for {project}\n{len(empty_cols)} empty cols out of {len(df.columns)}')
    plt.show()


