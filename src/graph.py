import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_heatmap(correlation_matrix, features, annot = False, name = "Pearson_heatmap.png"):
    matrix_df = pd.DataFrame(
        correlation_matrix,
        index = features,
        columns = features
    )
    plt.figure(figsize=(14, 10))
    sns.heatmap(matrix_df, annot=annot, vmin=-1, vmax=1)
    plt.savefig("../outputs/" + name, bbox_inches="tight")
    plt.close()
