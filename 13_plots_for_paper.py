"""
Yuli Tshuva
Plot results for paper
"""

# Imports
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from os.path import join
from tqdm import tqdm
from matplotlib import rcParams
from matplotlib.cm import get_cmap
import shap
import numpy as np
import pandas as pd
import json
import networkx as nx

# Set font
rcParams['font.family'] = 'Times New Roman'

# Constants
XGB_RESULTS_PATH = 'results_10'
LGB_RESULTS_PATH = 'LGB_results_10'
SAVE_PATH = "plots_for_paper"
XGB_SCORES = join(SAVE_PATH, "xgb_scores.csv")
LGB_SCORES = join(SAVE_PATH, "lgb_scores.csv")
BEST_YEAR = 2017

# Set colormaps

blue_cmap = [
    "#ADD8E6",  # Light Blue
    "#87CEEB",  # Sky Blue
    "#4682B4",  # Steel Blue
    "#1E90FF",  # Dodger Blue
    "#0000FF",  # Blue
    "#0000CD",  # Medium Blue
    "#00008B",  # Dark Blue
    "#000080",  # Navy
    "#191970"  # Midnight Blue
]

pink_cmap = [
    "#FFB6C1",  # Light Pink
    "#FF69B4",  # Hot Pink
    "#FF1493",  # Deep Pink
    "#DB7093",  # Pale Violet Red
    "#C71585",  # Medium Violet Red
    "#D02090",  # Violet Red
    "#FF007F",  # Bright Pink
    "#8B004F",  # Dark Pink
    "#660033"  # Very Dark Pink
]


def plot_roc_and_set_scores():
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set ranges
    roc_path = lambda year: f"roc_details_{year}.pkl"
    years = range(2011, 2020)
    cmaps = [blue_cmap, pink_cmap]

    # Read all rocs details and plot them
    for i, year in tqdm(enumerate(years)):
        for j, model_dir in enumerate([XGB_RESULTS_PATH, LGB_RESULTS_PATH]):
            with open(join(model_dir, roc_path(year)), 'rb') as f:
                roc_details = pickle.load(f)
            fpr, tpr, _ = roc_curve(roc_details['labels'], roc_details['scores'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{year} AUC: {roc_auc:.3f}', color=cmaps[j][i])

    # Set title and labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')

    # Set legend
    ax.legend(loc="lower right")

    # Save and show
    plt.savefig(join(SAVE_PATH, "roc_curve.png"))
    plt.show()

    file_names = ["xgb_scores.csv", "lgb_scores.csv"]
    for i, results_folder in enumerate([XGB_RESULTS_PATH, LGB_RESULTS_PATH]):
        results_lst = []

        # Load the results
        for year in range(2011, 2020):
            file = f"scores_{year}.json"
            with open(join(results_folder, file), 'r') as f:
                results = json.load(f)
                results_lst.append(pd.DataFrame(results, index=[year]))

        # Concatenate the results
        results_df = pd.concat(results_lst)
        results_df.to_csv(join(SAVE_PATH, file_names[i]))


def plot_stacked_feature_importance_and_shap_of_best_year():
    """This function is really self-explanatory"""
    # model path helper function
    model_path = lambda year: join(XGB_RESULTS_PATH, f"model_for_{year}.pkl")

    # Create figure
    fig = plt.figure(figsize=(20, 10))

    fig.suptitle("Feature Importance and Shap Values", fontsize=22, x=0.6)

    # Initialize an empty dictionary to hold cumulative importance values
    cumulative_importance = {}
    all_labels = set()

    # Number of years
    years = list(range(2011, 2020))

    # Get a colormap and generate colors
    cmap = get_cmap('tab10')  # You can choose any other colormap
    colors = [cmap(i % cmap.N) for i in range(len(years))]

    # Iterate over the years to collect all possible feature names
    for year in years:
        with open(model_path(year), 'rb') as f:
            model = pickle.load(f)
        importance = model.get_booster().get_score(importance_type="weight", fmap="")
        all_labels.update(importance.keys())

    # Convert to a sorted list to ensure consistent ordering
    all_labels = sorted(all_labels)

    # Initialize cumulative importance for each label to zero
    cumulative_importance = {label: 0 for label in all_labels}

    # Dictionary to store year-wise importance for each feature
    yearly_importance = {label: [0] * len(years) for label in all_labels}

    # Iterate over the years again to gather the importance values
    for idx, year in enumerate(years):
        with open(model_path(year), 'rb') as f:
            model = pickle.load(f)
        importance = model.get_booster().get_score(importance_type="weight", fmap="")

        # Fill missing values with zero
        for label in all_labels:
            yearly_importance[label][idx] = importance.get(label, 0)

    # Calculate cumulative importance and sort features accordingly
    cumulative_importance = {label: sum(yearly_importance[label]) for label in all_labels}
    sorted_labels = sorted(all_labels, key=lambda label: cumulative_importance[label], reverse=True)

    plt.subplot(1, 2, 1)

    sorted_labels = sorted_labels[:20]

    # Plot the data
    for idx, year in enumerate(years):
        values = [yearly_importance[label][idx] for label in sorted_labels]
        bottom_values = [sum(yearly_importance[label][:idx]) for label in sorted_labels]
        pretty_labels = [label.replace("_", " ").capitalize() for label in sorted_labels]
        pretty_labels[3] = "Head net\nincome loss"
        pretty_labels[6] = "Head stock\nholders equity"
        pretty_labels[9] = "Head square\nclustering coefficient"
        pretty_labels[11] = "Head operating\nincome loss"
        pretty_labels[12] = "Head common stock\nshares outstanding"
        pretty_labels[14] = "Tail eigenvector\ncentrality"
        pretty_labels[15] = "Tail square\nclustering coefficient"
        pretty_labels[16] = "Head eigenvector\ncentrality"
        pretty_labels[18] = "Head clustering\ncoefficient"
        plt.barh(pretty_labels, values, left=bottom_values, align="center", label=f"Year {year}", color=colors[idx])

    # Set the labels and limits
    plt.xlabel('F-score', fontdict={"size": 16})
    plt.ylabel('Features', fontdict={"size": 16})
    plt.title('Feature Importance', fontdict={"size": 19})
    plt.tick_params(axis="y", labelsize=14)

    # Clip empty edges and reverse the y-axis
    plt.xlim(left=0)
    plt.gca().invert_yaxis()  # Reverse the order of the labels

    # SHAP
    with open(model_path(2017), "rb") as file:
        model = pickle.load(file)

    # Initialize the SHAP explainer
    explainer = shap.Explainer(model)

    # Compute SHAP values
    # X_test = pd.read_csv(join(XGB_RESULTS_PATH, "X_test_2017.csv"), index_col=0)
    X_train = pd.read_csv(join(XGB_RESULTS_PATH, "X_train_2017.csv"), index_col=0)
    shap_values = explainer(X_train)

    # Create SHAP beeswarm plot on ax[0]
    plt.subplot(1, 2, 2)
    plt.title('Beeswarm Plot', fontdict={"size": 19})

    # Define the function to modify labels
    def modify_label(label):
        return label.replace("_", " ").capitalize()

    # Extract and modify the labels
    feature_names = shap_values.feature_names
    modified_feature_names = [modify_label(name) for name in feature_names]

    change_dict = {
        "Tail stockholdersequity": "Tail stock\nholders equity",
        "Tail netincomeloss": "Tail net\nincome loss",
        "Head stockholdersequity": "Head stock\nholders equity",
        "Tail commonstocksharesoutstanding": "Tail common stock\nshares outstanding",
        "Tail eigenvector centrality": "Tail eigenvector\ncentrality",
        "Head clustering coefficient": "Head clustering\ncoefficient",
        "Tail closeness centrality": "Tail closeness\ncentrality",
    }

    modified_feature_names = [change_dict.get(name, name) for name in modified_feature_names]

    # Update shap_values with modified labels
    shap_values.feature_names = modified_feature_names

    # Plot with modified labels
    shap.plots.beeswarm(shap_values, max_display=20, color_bar=False, show=False, plot_size=(10, 10))
    plt.xlabel("SHAP Value", fontdict={"size": 16})
    plt.tick_params(axis="y", labelsize=14)
    plt.rcParams['font.family'] = 'Times New Roman'

    # Show the plot
    plt.tight_layout()
    plt.savefig(join("plots_for_paper", "shap_and_feature_importance.png"), dpi=900)
    plt.show()


def lgb_plot_stacked_feature_importance_and_shap_of_best_year():
    """This function is really self-explanatory"""
    # model path helper function
    model_path = lambda year: join(LGB_RESULTS_PATH, f"model_for_{year}.pkl")

    # Create figure
    fig = plt.figure(figsize=(30, 30))

    fig.suptitle("Feature Importance and Shap Values", fontsize=30)

    # Initialize an empty dictionary to hold cumulative importance values
    cumulative_importance = {}
    all_labels = set()

    # Number of years
    years = list(range(2011, 2020))

    # Get a colormap and generate colors
    cmap = get_cmap('tab10')  # You can choose any other colormap
    colors = [cmap(i % cmap.N) for i in range(len(years))]

    # Iterate over the years to collect all possible feature names
    for year in tqdm(years):
        with open(model_path(year), 'rb') as f:
            model = pickle.load(f)
        importance = model.get_booster().get_score(importance_type="weight", fmap="")
        all_labels.update(importance.keys())

    # Convert to a sorted list to ensure consistent ordering
    all_labels = sorted(all_labels)

    # Initialize cumulative importance for each label to zero
    cumulative_importance = {label: 0 for label in all_labels}

    # Dictionary to store year-wise importance for each feature
    yearly_importance = {label: [0] * len(years) for label in all_labels}

    # Iterate over the years again to gather the importance values
    for idx, year in enumerate(tqdm(years)):
        with open(model_path(year), 'rb') as f:
            model = pickle.load(f)
        importance = model.get_booster().get_score(importance_type="weight", fmap="")

        # Fill missing values with zero
        for label in all_labels:
            yearly_importance[label][idx] = importance.get(label, 0)

    # Calculate cumulative importance and sort features accordingly
    cumulative_importance = {label: sum(yearly_importance[label]) for label in all_labels}
    sorted_labels = sorted(all_labels, key=lambda label: cumulative_importance[label], reverse=True)

    plt.subplot(1, 7, 3)

    # Plot the data
    for idx, year in enumerate(years):
        values = [yearly_importance[label][idx] for label in sorted_labels]
        bottom_values = [sum(yearly_importance[label][:idx]) for label in sorted_labels]
        plt.barh(sorted_labels, values, left=bottom_values, align="center", label=f"Year {year}", color=colors[idx])

    # Set the labels and limits
    plt.xlabel('F-score')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.legend(loc="lower right")

    # Clip empty edges and reverse the y-axis
    plt.xlim(left=0)
    plt.gca().invert_yaxis()  # Reverse the order of the labels

    # SHAP
    with open(model_path(2017), "rb") as file:
        model = pickle.load(file)

    # Initialize the SHAP explainer
    explainer = shap.Explainer(model)

    # Compute SHAP values
    X_test = pd.read_csv(join(LGB_RESULTS_PATH, "X_test_2017.csv"), index_col=0)
    # X_train = pd.read_csv(join(LGB_RESULTS_PATH, "X_train_2017.csv"), index_col=0)
    shap_values = explainer(X_test)

    # Create SHAP beeswarm plot on ax[0]
    plt.subplot(1, 7, 7)
    plt.title('Shap Values - Beeswarm Plot')
    shap.plots.beeswarm(shap_values, max_display=35)

    # Show the plot
    plt.tight_layout(pad=1.5)
    plt.show()


def plot_aucs():
    plt.figure(figsize=(11, 7))

    # Set ranges
    roc_path = lambda dir, year: join(dir, f"roc_details_{year}.pkl")
    years = range(2011, 2020)

    aucs_list = [[],  # XGB
                 []]  # LGB

    # Read all rocs details and plot them
    for i, year in tqdm(enumerate(years)):
        for j, model_dir in enumerate([XGB_RESULTS_PATH, LGB_RESULTS_PATH]):
            with open(roc_path(model_dir, year), 'rb') as f:
                roc_details = pickle.load(f)
            aucs_list[j].append(roc_auc_score(roc_details['labels'], roc_details['scores']))

    # Number of bars
    x = np.arange(len(years))

    # Width of a bar
    width = 0.35

    # Create the plot
    plt.bar(x - width / 2, aucs_list[0], width, label='XGBoost', color="dodgerblue", edgecolor="black")
    plt.bar(x + width / 2, aucs_list[1], width, label='LightGBM', color="hotpink", edgecolor="black")

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('AUC')
    plt.title("Models' AUCs")
    plt.xticks(x, years, rotation=30)
    plt.ylim(0, 1)
    plt.legend()

    # Save and show
    plt.savefig(join(SAVE_PATH, "aucs_barplot.png"))
    plt.show()


def hits_plot():
    """Plot graph of evaluation measures for each model in both sides"""
    fig = plt.figure(figsize=(16, 12))

    fig.suptitle("Models' Evaluation Measures", fontsize=40, y=0.96)

    # Define the grid structure
    grid = (2, 2)
    padding = 2
    width = 0.8
    mr_width = 2
    bar_colors = ["skyblue", "salmon", "mediumaquamarine", "lightpink"]
    xtick_labels = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]

    ax1 = plt.subplot2grid(grid, (0, 0))

    side = "head"
    df1, df2 = pd.read_csv(XGB_SCORES), pd.read_csv(LGB_SCORES)

    # Drop 2019
    df1.drop(index=8, inplace=True)
    df2.drop(index=8, inplace=True)

    df_side = df1[[f"hits_1_{side}", f"hits_3_{side}", f"hits_5_{side}", f"hits_10_{side}", f"mr_{side}"]]
    df_side.columns = ["Hits@1", "Hits@3", "Hits@5", "Hits@10", 'MR']

    df_side[["Hits@1", "Hits@3", "Hits@5", "Hits@10"]]. \
        plot(kind='bar', ax=ax1, width=width, color=bar_colors,
             label=["Hits@1", "Hits@3", "Hits@5", "Hits@10"])

    # Create a secondary y-axis for the line plot
    ax2 = ax1.twinx()

    # Plot the line on the secondary y-axis with a legend label
    line, = ax2.plot(df_side.index, df_side["MR"], color="black", linewidth=mr_width, label="MR")

    ax1.set_xticks(df_side.index)
    ax1.set_xticklabels(xtick_labels, rotation=45)

    # Set axis labels
    # ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Hits@k Values', fontsize=19)
    # ax2.set_ylabel('MR Values', fontsize=14)

    # Get handles and labels for both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Add a single legend for both axes
    ax1.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc="upper left")

    # Add a title
    ax1.set_title("XGBoost Head Prediction", fontsize=22)

    ax3 = plt.subplot2grid(grid, (0, 1))

    side = "tail"
    df_side = df1[[f"hits_1_{side}", f"hits_3_{side}", f"hits_5_{side}", f"hits_10_{side}", f"mr_{side}"]]
    df_side.columns = ["Hits@1", "Hits@3", "Hits@5", "Hits@10", 'MR']

    df_side[["Hits@1", "Hits@3", "Hits@5", "Hits@10"]]. \
        plot(kind='bar', ax=ax3, width=width, color=bar_colors,
             label=["Hits@1", "Hits@3", "Hits@5", "Hits@10"])

    # Create a secondary y-axis for the line plot
    ax4 = ax3.twinx()

    # Plot the line on the secondary y-axis with a legend label
    line, = ax4.plot(df_side.index, df_side["MR"], color="black", linewidth=mr_width, label="MR")

    ax3.set_xticks(df_side.index)
    ax3.set_xticklabels(xtick_labels, rotation=45)

    # Set axis labels
    # ax3.set_xlabel('Year', fontsize=14)
    # ax3.set_ylabel('Hits@k Values', fontsize=14)
    ax4.set_ylabel('MR Values', fontsize=19)

    # Get handles and labels for both axes
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()

    # Add a single legend for both axes
    ax3.legend(handles=handles3 + handles4, labels=labels3 + labels4, loc="upper left")

    # Add a title
    ax3.set_title("XGBoost Tail Prediction", fontsize=22)

    ax5 = plt.subplot2grid(grid, (1, 0))

    side = "head"

    df_side = df2[[f"hits_1_{side}", f"hits_3_{side}", f"hits_5_{side}", f"hits_10_{side}", f"mr_{side}"]]
    df_side.columns = ["Hits@1", "Hits@3", "Hits@5", "Hits@10", 'MR']

    df_side[["Hits@1", "Hits@3", "Hits@5", "Hits@10"]].plot(kind='bar', ax=ax5, width=width, color=bar_colors,
                                                            label=["Hits@1", "Hits@3", "Hits@5", "Hits@10"])

    # Create a secondary y-axis for the line plot
    ax6 = ax5.twinx()

    # Plot the line on the secondary y-axis with a legend label
    line, = ax6.plot(df_side.index, df_side["MR"], color="black", linewidth=mr_width, label="MR")

    ax5.set_xticks(df_side.index)
    ax5.set_xticklabels(xtick_labels, rotation=45)

    # Set axis labels
    ax5.set_xlabel('Year', fontsize=19)
    ax5.set_ylabel('Hits@k Values', fontsize=19)
    # ax6.set_ylabel('MR Values', fontsize=14)

    # Get handles and labels for both axes
    handles5, labels5 = ax5.get_legend_handles_labels()
    handles6, labels6 = ax6.get_legend_handles_labels()

    # Add a single legend for both axes
    ax5.legend(handles=handles5 + handles6, labels=labels5 + labels6, loc="upper left")

    # Add a title
    ax5.set_title("LightGBM Head Prediction", fontsize=22)

    ax7 = plt.subplot2grid(grid, (1, 1))

    side = "tail"

    df_side = df2[[f"hits_1_{side}", f"hits_3_{side}", f"hits_5_{side}", f"hits_10_{side}", f"mr_{side}"]]
    df_side.columns = ["Hits@1", "Hits@3", "Hits@5", "Hits@10", 'MR']

    df_side[["Hits@1", "Hits@3", "Hits@5", "Hits@10"]].plot(kind='bar', ax=ax7, width=width, color=bar_colors,
                                                            label=["Hits@1", "Hits@3", "Hits@5", "Hits@10"])

    # Create a secondary y-axis for the line plot
    ax8 = ax7.twinx()

    # Plot the line on the secondary y-axis with a legend label
    line, = ax8.plot(df_side.index, df_side["MR"], color="black", linewidth=mr_width, label="MR")

    ax7.set_xticks(df_side.index)
    ax7.set_xticklabels(xtick_labels, rotation=45)

    # Set axis labels
    ax7.set_xlabel('Year', fontsize=19)
    # ax7.set_ylabel('Hits@k Values', fontsize=14)
    ax8.set_ylabel('MR Values', fontsize=19)

    # Get handles and labels for both axes
    handles7, labels7 = ax7.get_legend_handles_labels()
    handles8, labels8 = ax8.get_legend_handles_labels()

    # Add a single legend for both axes
    ax7.legend(handles=handles7 + handles8, labels=labels7 + labels8, loc="upper left")

    # Add a title
    ax7.set_title("LightGBM Tail Prediction", fontsize=22)

    # Add a single legend for the entire figure above the subplots
    handles, labels = handles1 + handles2, labels1 + labels2
    fig.legend(handles, labels, loc="lower center", ncol=6)

    for ax in [ax1, ax3, ax5, ax7]:
        ax.legend().set_visible(False)

    # Adjust layout to prevent clipping of titles and labels
    plt.subplots_adjust(bottom=0.6)
    plt.tight_layout(pad=padding)

    # Show the plot
    plt.savefig(join(SAVE_PATH, "hits_mr_models_comparison.pdf"))
    plt.show()


def collapse_graph():
    def test():
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3, 4, 5, 6, 7], type='Director')
        G.add_nodes_from(['A', 'B', 'C', 'D'], type='Company')
        G.add_edges_from(
            [(1, 'A'), (2, 'A'), (3, 'A'), (2, 'B'), (5, 'B'), (4, 'C'), (5, 'C'), (5, 'D'), (6, 'D'), (7, 'D')])

        officer_nodes = {n for n, d in G.nodes(data=True) if d["type"] == 'Officer'}
        company_nodes = set(G) - officer_nodes

        GP = nx.bipartite.projection.projected_graph(G, officer_nodes)
        GC = nx.bipartite.projection.projected_graph(G, company_nodes)

        plt.figure(1)
        pos = nx.bipartite_layout(G, officer_nodes)
        nx.draw_networkx_labels(G, pos=pos)
        nx.draw_networkx_nodes(G, nodelist=officer_nodes, pos=pos, node_color='orange')
        nx.draw_networkx_nodes(G, nodelist=company_nodes, pos=pos, node_shape='s')
        nx.draw_networkx_edges(G, pos=pos)

        pos = nx.spring_layout(G, seed=0)
        plt.figure(2)
        nx.draw_networkx_nodes(G, pos=pos, nodelist=officer_nodes, node_color='orange')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=company_nodes, node_shape='s')
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_labels(G, pos=pos)

        plt.figure(3)
        nx.draw(GP, pos=pos, node_color='orange', with_labels=True)

        plt.figure(4)
        nx.draw(GC, pos=pos, node_shape='s', with_labels=True)

        GC = nx.MultiDiGraph(GC)
        nx.set_edge_attributes(GC, 'ID', 'type')
        GC.add_edges_from([('A', 'C'), ('B', 'D')], type='M&A')
        GC.add_nodes_from([11, 22], type='SIC')
        GC.add_edges_from([('A', 11), ('B', 11), ('C', 11), ('D', 22)], type='SIC')

        ID_edges = [(u, v) for u, v, e in GC.edges(data=True) if e['type'] == 'ID']
        MA_edges = [(u, v) for u, v, e in GC.edges(data=True) if e['type'] == 'M&A']
        SIC_nodes = [n for n, d in GC.nodes(data=True) if d['type'] == 'SIC']
        SIC_edges = [(u, v) for u, v, e in GC.edges(data=True) if e['type'] == 'SIC']

        plt.figure(5)
        nx.draw_networkx_nodes(GC, pos=pos, nodelist=company_nodes, node_shape='s')
        nx.draw_networkx_labels(GC, pos=pos, labels={n: n for n in company_nodes})

    test()
    plt.show()


def main():
    collapse_graph()


if __name__ == "__main__":
    main()
