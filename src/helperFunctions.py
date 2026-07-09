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

def kendall_correlation(df, feature_X, feature_Y):
    concordant = 0
    discordant = 0
    tie_X = 0
    tie_Y = 0
    for i in range(len(df[feature_X])):
        for j in range(i+1, len(df[feature_X])):
            diff_X = df[feature_X].iloc[j] - df[feature_X].iloc[i]
            diff_Y = df[feature_Y].iloc[j] - df[feature_Y].iloc[i]

            if diff_X == 0 and diff_Y == 0:
                continue
            elif diff_X == 0:
                tie_X += 1
            elif diff_Y == 0:
                tie_Y += 1
            elif diff_X * diff_Y > 0:
                concordant += 1
            else:
                discordant += 1 
    denominator = math.sqrt(
        (concordant + discordant + tie_X) *
        (concordant + discordant + tie_Y)
    )

    if denominator == 0:
        return 0
    return (concordant - discordant) / denominator