import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def process_data(file_path):
    """
    Read, clean, and transpose the data.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV file.
    - clustering_data (pd.DataFrame): Cleaned and processed data for clustering.
    - normalized_data (pd.DataFrame): Normalized version of the cleaned data.
    - transposed_data (pd.DataFrame): Transposed version of the cleaned data.
    - scaler (StandardScaler): Scaler used for normalization.
    """
    # Load the original data
    original_data = pd.read_csv(file_path)

    # Select relevant columns for clustering
    columns_for_clustering = ['Urban population growth (annual %) [SP.URB.GROW]' ,
                              'Rural population growth (annual %) [SP.RUR.TOTL.ZG]' ,
                              'Population in the largest city (% of urban population) [EN.URB.LCTY.UR.ZS]' ,
                              'Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]' ,
                              'Combustible renewables and waste (% of total energy) [EG.USE.CRNW.ZS]']

    # Extract the relevant columns
    clustering_data = original_data[columns_for_clustering]

    # Convert data to numeric, handling errors by setting them to NaN
    clustering_data = clustering_data.apply(pd.to_numeric , errors='coerce')

    # Replace missing values with the mean of each column
    clustering_data = clustering_data.apply(lambda x: x.fillna(x.mean()))

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(clustering_data)

    # Transpose the cleaned data
    transposed_data = clustering_data.transpose()

    return original_data , clustering_data , normalized_data , transposed_data , scaler


def polynomial_model(x , a , b , c):
    """
        Compute the value of a quadratic polynomial model.

        Parameters:
        - x (float): Input variable.
        - a (float): Coefficient of the quadratic term.
        - b (float): Coefficient of the linear term.
        - c (float): Constant term.

        Returns:
        - float: Value of the quadratic polynomial at the given input x.
        """
    return a * x**2 + b * x + c


def err_ranges(x_values , params , cov_matrix):
    """
        Calculate the confidence interval bounds for each parameter of a model.

        Parameters:
        - x_values (array-like): Input values.
        - params (array-like): Fitted parameters of the model.
        - cov_matrix (2D array): Covariance matrix of the fitted parameters.

        Returns:
        - tuple: Lower and upper bounds for each parameter.
        """

    # Calculate standard errors for each parameter
    std_errors = np.sqrt(np.diag(cov_matrix))

    # Calculate margin of error for each parameter
    margin_of_error = 1.96 * std_errors  # 1.96 corresponds to a 95% confidence interval

    # Calculate lower and upper bounds for each parameter
    lower_bound = params - margin_of_error
    upper_bound = params + margin_of_error

    # Generate predictions for lower and upper bounds
    lower_bound_values = polynomial_model(x_values , *lower_bound)
    upper_bound_values = polynomial_model(x_values , *upper_bound)

    return lower_bound_values , upper_bound_values


# Example usage of the process_data function
file_path = "f7434561-256c-4cb5-ae89-4d7331842ed0_Data.csv"
original_data , clustering_data , normalized_data , transposed_data , scaler = process_data(file_path)

# Apply K-means clustering
kmeans = KMeans(n_clusters = 6 , random_state = 42)
original_data['Cluster'] = kmeans.fit_predict(normalized_data)

# Get cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Calculate Silhouette Score
silhouette_avg = silhouette_score(normalized_data , original_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the clusters and cluster centers
plt.scatter(original_data['Urban population growth (annual %) [SP.URB.GROW]'] ,
            original_data['Rural population growth (annual %) [SP.RUR.TOTL.ZG]'] ,
            c = original_data['Cluster'] , cmap = 'viridis' , label = 'Data Points')
plt.scatter(cluster_centers[: , 0] , cluster_centers[: , 1] , marker = 'X' , s = 100 ,
            c = 'red' , label = 'Cluster Centers')

# Set labels, title, and legend
plt.xlabel('Urban population growth (annual %)')
plt.ylabel('Rural population growth (annual %)')
plt.title('Clustering of Countries with Cluster Centers')
plt.legend()

# Display the plot
plt.show()

# Select relevant columns for modeling
modeling_data = original_data[['Time' , 'Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]']]

# Convert 'Year' column to numeric
modeling_data['Time'] = pd.to_numeric(modeling_data['Time'] , errors = 'coerce')

# Convert 'Oil rents (% of GDP)' column to numeric, dropping rows with non-numeric values
modeling_data['Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]'] = \
    pd.to_numeric(modeling_data['Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]'] , errors = 'coerce')

# Remove rows with missing values
modeling_data = modeling_data.dropna()

# Fit the model
popt , pcov = curve_fit(polynomial_model , modeling_data['Time'] ,
                       modeling_data['Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]'])

# Generate predictions for future years
future_years = np.arange(1990 , 2051 , 1)
predicted_values = polynomial_model(future_years , *popt)
print(predicted_values)

# Calculate confidence intervals
lower_bound , upper_bound = err_ranges(future_years , popt , pcov)

# Plot the data, best-fitting function, and confidence intervals
plt.scatter(modeling_data['Time'] , modeling_data['Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]'] ,
            label = 'Actual Data')
plt.plot(future_years , predicted_values , label = 'Best-Fitting Function' ,
         color = 'red')
plt.fill_between(future_years , lower_bound , upper_bound , color = 'gray' ,
                 alpha = 0.3 , label = 'Confidence Interval')

# Set labels, title, and legend
plt.xlabel('Year')
plt.ylabel('Oil rents (% of GDP)')
plt.title('Low-Order Polynomial Model with Confidence Intervals')
plt.legend()

# Display the plot
plt.show()
