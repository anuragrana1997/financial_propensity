import pandas as pd
import helperFunctions
import graph

df = pd.read_csv("../data/german_credit.csv")
data = df.values.tolist()

for i in range(5):
    print(data[i])

''' target distribution '''
targetDistribution = {}
for row in data:
    target = row[-1]
    if target not in targetDistribution:
        targetDistribution[target] = 0
    targetDistribution[target] += 1
# print(targetDistribution)

features = df.columns[:-1]
means = helperFunctions.mean_of_features(df, features)

''' pearson correlation matrix formation '''
print("______________Pearson________________")
pearson_cor_matrix = []
for key_X in features:
    pearson_cor_matrix_row = []
    for key_Y in features:
        covariance_XY = helperFunctions.covariance(df, means, key_X, key_Y)
        variance_X = helperFunctions.covariance(df, means, key_X, key_X)
        variance_y = helperFunctions.covariance(df, means, key_Y, key_Y)
        correlation_value = helperFunctions.pearson_correlation(covariance_XY, variance_X, variance_y)
        pearson_cor_matrix_row.append(round(float(correlation_value), 2))
    pearson_cor_matrix.append(pearson_cor_matrix_row)

print(pearson_cor_matrix)

''' spearman correlation matrix formation '''
print("______________Spearman________________")
ranked_df = {}
for key in features:
    ranked_df[key] = df[key].rank(method = "average")

ranked_means = helperFunctions.mean_of_features(ranked_df, features)

spearman_cor_matrix = []
for key_X in features:
    spearman_cor_matrix_row = []
    for key_Y in features:
        covariance_XY = helperFunctions.covariance(ranked_df, ranked_means, key_X, key_Y)
        variance_X = helperFunctions.covariance(ranked_df, ranked_means, key_X, key_X)
        variance_y = helperFunctions.covariance(ranked_df, ranked_means, key_Y, key_Y)
        correlation_value = helperFunctions.pearson_correlation(covariance_XY, variance_X, variance_y)
        spearman_cor_matrix_row.append(round(float(correlation_value), 2))
    spearman_cor_matrix.append(spearman_cor_matrix_row)

print(spearman_cor_matrix) 

''' kendall correlation '''
print("______________Kendall________________")
kendall_cor_matrix = []
for i in range(len(features)):
    kendall_cor_matrix_row = []
    for j in range(len(features)):
        if i == j:
            correlation_value = 1
        elif j < i:
            correlation_value = kendall_cor_matrix[j][i]
        else:
            correlation_value = helperFunctions.kendall_correlation(df, features[i], features[j])
        kendall_cor_matrix_row.append(round(float(correlation_value), 2))
    kendall_cor_matrix.append(kendall_cor_matrix_row)

print(kendall_cor_matrix)

''' heatmaps ''' 

pearson_matrix_df = pd.DataFrame(
    pearson_cor_matrix,
    index = features,
    columns = features
)

graph.create_heatmap(pearson_cor_matrix, features, False, "pearson_heatmap.png")
graph.create_heatmap(spearman_cor_matrix, features, False, "spearman_heatmap.png")
graph.create_heatmap(kendall_cor_matrix, features, False, "kendall_heatmap.png")