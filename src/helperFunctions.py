import math

def pearson_correlation(covariance_XY,  variance_X, variance_Y):
    denominator = (math.sqrt(variance_X) * math.sqrt(variance_Y))
    if denominator == 0:
        return 0
    return covariance_XY / denominator

def covariance(df, means, feature_X, feature_Y):
    total = 0
    n = len(df[feature_X])
    for i in range(n):
        sum_diff_X = df[feature_X][i] - means[feature_X]
        sum_diff_Y = df[feature_Y][i] - means[feature_Y]
        total += sum_diff_X * sum_diff_Y
    return total / n

def mean_of_features(df, features):
    means = {}
    for key in features:
        means[key] = sum(df[key]) / len(df[key])
    return means
