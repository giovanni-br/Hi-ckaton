import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



def visualize_pca_variance(X_train_scaled, max_components=50):
    """
    Visualizes the variance explained by PCA components.
    """
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), cumulative_variance, marker='o', linestyle='--', label='Cumulative Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% Variance')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Variance Explained')
    plt.legend()
    plt.grid()
    plt.show()

    return cumulative_variance


def evaluate_reduction_methods(X_train_scaled, y_train, X_valid_scaled, y_valid, n_features_list):
    """
    Evaluates filter, embedded, and PCA methods for feature reduction.
    """
    results = []

    for n_features in n_features_list:
        # Filter Method: SelectKBest with mutual information
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        X_train_filter = selector.fit_transform(X_train_scaled, y_train)
        X_valid_filter = selector.transform(X_valid_scaled)

        # Embedded Method: RFECV with RandomForest
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        selector_embedded = RFECV(estimator=model, step=1, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        selector_embedded.fit(X_train_scaled, y_train)
        X_train_embedded = selector_embedded.transform(X_train_scaled)[:, :n_features]
        X_valid_embedded = selector_embedded.transform(X_valid_scaled)[:, :n_features]

        # Unsupervised Method: PCA
        pca = PCA(n_components=n_features, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_valid_pca = pca.transform(X_valid_scaled)

        # Evaluate performance
        methods = {
            'filter': X_train_filter,
            'embedded': X_train_embedded,
            'pca': X_train_pca
        }

        for method, X_train_reduced in methods.items():
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train_reduced, y_train)
            X_valid_reduced = locals()[f'X_valid_{method}']
            predictions = model.predict(X_valid_reduced)
            mse = mean_squared_error(y_valid, predictions)
            results.append({'method': method, 'n_features': n_features, 'mse': mse})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    for method in results_df['method'].unique():
        subset = results_df[results_df['method'] == method]
        plt.plot(subset['n_features'], subset['mse'], marker='o', linestyle='--', label=f'{method} method')
    
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Squared Error')
    plt.title('Feature Reduction Methods Comparison')
    plt.legend()
    plt.grid()
    plt.show()

    return results_df