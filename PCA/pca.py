import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_principal_components(X, components=None):
    if components is None:
        components = X.shape[1]
    
    # Calculate covariance matrix of rows of X
    covariance = np.cov(X, rowvar=False)
    
    # Compute eigenvalues and eignenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Compute principal components
    principal_components = eigenvectors[:, :components]
    
    # Compute variance of the principal components
    variance = np.square(eigenvalues)[:components]
    
    return principal_components, variance

def project_onto_components(X, components):
    return np.dot(X, components)
