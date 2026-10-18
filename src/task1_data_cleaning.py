"""
Task 1: Comprehensive Data Quality Assessment and Missing Value Analysis

This module implements various data cleaning techniques and missing value handling strategies,
including multiple imputation methods and comprehensive visualization of data quality issues.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import fetch_openml
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class DataQualityAnalyzer:
    """
    A comprehensive data quality analyzer that identifies and handles missing values,
    duplicates, and other data quality issues.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.imputed_datasets = {}
        self.missing_indicators = None
        
    def assess_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            Dict: Dictionary containing total_rows, total_columns, duplicates, 
                missing_values, missing_percentage, data_types, and memory_usage
        """
        return {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'duplicates': int(self.data.duplicated().sum()),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().mean() * 100).to_dict(),
            'data_types': self.data.dtypes.astype(str).to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / (1024 ** 2)
        }
    
    def identify_missing_pattern(self, column: str) -> str:
        """
        Attempt to identify the missing data pattern (MCAR, MAR, MNAR).
        
        Parameters:
            column (str): Column name to analyze
            
        Returns:
            str: String indicating likely missing pattern (MCAR, MAR, or MNAR)
        """
        if column not in self.data.columns:
            return "Column not found"

        indicator = self.data[column].isna().astype(int)

        if indicator.sum() == 0:
            return "No missing values"

        numeric = self.data.select_dtypes(include=[np.number])

        for other in numeric.columns:
            if other == column:
                continue

            observed = self.data[other].notna()

            if observed.sum() < 3 or indicator[observed].nunique() < 2:
                continue

            correlation = indicator[observed].corr(
                self.data.loc[observed, other]
            )

            if pd.notna(correlation) and abs(correlation) >= 0.2:
                return "Likely MAR (Missing at Random)"

        # MNAR cannot be established from observed data alone.
        return "Likely MCAR (Missing Completely at Random)"
    
    def create_missing_indicators(self) -> pd.DataFrame:
        """
        Create binary indicator variables for missing values.
        
        Returns:
            pd.DataFrame: DataFrame with binary indicators for missing values
        """
        missing = self.data.isna()

        self.missing_indicators = missing.loc[
            :, missing.any()
        ].astype(int)

        self.missing_indicators.columns = [
            f'{column}_missing'
            for column in self.missing_indicators.columns
        ]

        return self.missing_indicators.copy()
    
    def simple_imputation(self, strategy: str = 'mean') -> pd.DataFrame:
        """
        Perform simple imputation (mean, median, or mode).
        
        Parameters:
            strategy (str): Imputation strategy - 'mean', 'median', or 'most_frequent'
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        if strategy == 'mode':
            strategy = 'most_frequent'

        if strategy not in ('mean', 'median', 'most_frequent'):
            raise ValueError('Unsupported imputation strategy')

        result = self.original_data.copy()

        numeric_cols = result.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        categorical_cols = result.columns.difference(
            numeric_cols
        ).tolist()

        if numeric_cols:
            numeric_strategy = strategy

            valid_cols = [
                col for col in numeric_cols
                if result[col].notna().any()
            ]

            for col in valid_cols:
                imputer = SimpleImputer(strategy=numeric_strategy)
                result[col] = imputer.fit_transform(
                    result[[col]]
                ).ravel()

        for col in categorical_cols:
            if result[col].notna().any():
                imputer = SimpleImputer(strategy='most_frequent')
                result[col] = imputer.fit_transform(
                    result[[col]]
                ).ravel()

        self.imputed_datasets[f'simple_{strategy}'] = result.copy()

        return result
    
    def knn_imputation(self, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Perform KNN imputation on numeric columns.
        
        Parameters:
            n_neighbors (int): Number of neighbors to use for imputation
            
        Returns:
            pd.DataFrame: DataFrame with KNN imputed values
        """
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be positive')

        result = self.original_data.copy()

        numeric_cols = result.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        valid_cols = [
            col for col in numeric_cols
            if result[col].notna().any()
        ]

        if valid_cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)

            result[valid_cols] = imputer.fit_transform(
                result[valid_cols]
            )

        for col in result.columns.difference(numeric_cols):
            if result[col].notna().any():
                imputer = SimpleImputer(strategy='most_frequent')
                result[col] = imputer.fit_transform(
                    result[[col]]
                ).ravel()

        self.imputed_datasets['knn'] = result.copy()

        return result
    
    def iterative_imputation(self, max_iter: int = 10) -> pd.DataFrame:
        """
        Perform multivariate imputation using MICE algorithm.
        
        Parameters:
            max_iter (int): Maximum number of iterations
            
        Returns:
            pd.DataFrame: DataFrame with iteratively imputed values
        """
        if max_iter < 1:
            raise ValueError('max_iter must be positive')

        result = self.original_data.copy()

        numeric_cols = result.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        valid_cols = [
            col for col in numeric_cols
            if result[col].notna().any()
        ]

        if valid_cols:
            imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=42
            )

            result[valid_cols] = imputer.fit_transform(
                result[valid_cols]
            )

        for col in result.columns.difference(numeric_cols):
            if result[col].notna().any():
                imputer = SimpleImputer(strategy='most_frequent')
                result[col] = imputer.fit_transform(
                    result[[col]]
                ).ravel()

        self.imputed_datasets['iterative'] = result.copy()

        return result
    
    def compare_imputation_methods(self) -> pd.DataFrame:
        """
        Compare statistical properties across different imputation methods.
        
        Returns:
            DataFrame with comparison statistics
        """
        comparison = {}
        
        # Original data statistics (excluding missing values)
        comparison['original'] = self.original_data.describe().loc[['mean', 'std', 'min', 'max']]
        
        # Statistics for each imputed dataset
        for name, dataset in self.imputed_datasets.items():
            comparison[name] = dataset.describe().loc[
                ['mean', 'std', 'min', 'max']
            ]

        return pd.concat(comparison, axis=1)
    
    def visualize_missing_data(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive visualizations of missing data patterns.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        fig = plt.figure(figsize=figsize)
        
        # 1. Missing data heatmap
        plt.subplot(2, 2, 1)
        msno.matrix(self.data, ax=plt.gca(), sparkline=False)
        plt.title('Missing Data Pattern Heatmap', fontsize=12, fontweight='bold')
        
        # 2. Bar plot of missing percentages
        plt.subplot(2, 2, 2)
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            missing_pct.plot(kind='bar', color='coral')
            plt.title('Percentage of Missing Values by Column', fontsize=12, fontweight='bold')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            plt.title('Percentage of Missing Values by Column', fontsize=12, fontweight='bold')
        
        # 3. Correlation heatmap of missing indicators
        plt.subplot(2, 2, 3)
        if self.missing_indicators is not None and len(self.missing_indicators.columns) > 0:
            corr_matrix = self.missing_indicators.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Between Missing Value Patterns', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Missing Value Patterns', ha='center', va='center', fontsize=14)
            plt.title('Correlation Between Missing Value Patterns', fontsize=12, fontweight='bold')
        
        # 4. Missing data dendrogram
        plt.subplot(2, 2, 4)
        if self.data.isnull().sum().sum() > 0:
            msno.dendrogram(self.data, ax=plt.gca())
            plt.title('Missing Data Dendrogram', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            plt.title('Missing Data Dendrogram', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def visualize_imputation_comparison(self, column: str, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize distribution comparison for a specific column across imputation methods.
        
        Parameters:
            column (str): Column name to visualize
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        if column not in self.data.select_dtypes(include=[np.number]).columns:
            print(f"Column '{column}' is not numeric or does not exist")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data (with missing values removed for visualization)
        ax = axes[0]
        self.original_data[column].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(self.original_data[column].median(), color='green', linestyle='--', linewidth=2, label='Median')
        ax.set_title(f'Original Data (n={self.original_data[column].notna().sum()})', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Imputed datasets
        methods = list(self.imputed_datasets.keys())
        for idx, method in enumerate(methods[:5], start=1):
            ax = axes[idx]
            imputed_df = self.imputed_datasets[method]
            imputed_df[column].hist(bins=30, ax=ax, color='lightcoral', edgecolor='black', alpha=0.7)
            ax.axvline(imputed_df[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(imputed_df[column].median(), color='green', linestyle='--', linewidth=2, label='Median')
            ax.set_title(f'{method.upper()} Imputation', fontweight='bold')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplot if less than 5 methods
        if len(methods) < 5:
            axes[5].axis('off')
        
        plt.suptitle(f'Distribution Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_correlation_comparison(self, figsize: Tuple[int, int] = (15, 8)):
        """
        Compare correlation matrices before and after imputation.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation comparison")
            return
        
        n_methods = len(self.imputed_datasets) + 1
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_methods > 1 else [axes]
        
        # Original correlation (pairwise deletion of missing values)
        ax = axes[0]
        corr_original = self.original_data[numeric_cols].corr()
        sns.heatmap(corr_original, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Original Data Correlation', fontweight='bold')
        
        # Imputed correlations
        for idx, (method_name, imputed_df) in enumerate(self.imputed_datasets.items(), start=1):
            if idx < len(axes):
                ax = axes[idx]
                corr_imputed = imputed_df[numeric_cols].corr()
                sns.heatmap(corr_imputed, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title(f'{method_name.upper()} Imputation', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Correlation Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_missing() -> pd.DataFrame:
    """
    Create a sample dataset with various types of missing values for demonstration.
    
    Returns:
        pd.DataFrame: DataFrame with missing values
    """
    np.random.seed(42)
    n_samples = 200
    
    # Create base data
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'years_employed': np.random.exponential(5, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce MCAR: completely random missing values in age
    mcar_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[mcar_indices, 'age'] = np.nan
    
    # Introduce MAR: income missing when credit_score is low
    mar_condition = df['credit_score'] < 650
    mar_indices = df[mar_condition].sample(frac=0.4).index
    df.loc[mar_indices, 'income'] = np.nan
    
    # Introduce MNAR: high debt_ratio values are missing (people hide high debt)
    mnar_condition = df['debt_ratio'] > 0.6
    mnar_indices = df[mnar_condition].sample(frac=0.5).index
    df.loc[mnar_indices, 'debt_ratio'] = np.nan
    
    # Add some random missing in years_employed
    random_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[random_indices, 'years_employed'] = np.nan
    
    return df


def main():
    """
    Main function demonstrating the complete Task 1 workflow.
    
    Returns:
        DataQualityAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 1: Comprehensive Data Quality Assessment and Missing Value Analysis")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with missing values...")
    df = create_sample_dataset_with_missing()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = DataQualityAnalyzer(df)
    
    # Assess data quality
    print("1. DATA QUALITY ASSESSMENT")
    print("-" * 80)
    quality_report = analyzer.assess_data_quality()
    print(f"Total Rows: {quality_report['total_rows']}")
    print(f"Total Columns: {quality_report['total_columns']}")
    print(f"Duplicates: {quality_report['duplicates']}")
    print(f"Memory Usage: {quality_report['memory_usage']:.2f} MB")
    print()
    print("Missing Values Summary:")
    for col, count in quality_report['missing_values'].items():
        pct = quality_report['missing_percentage'][col]
        if count > 0:
            print(f"  {col}: {count} ({pct:.1f}%)")
    print()
    
    # Identify missing patterns
    print("2. MISSING DATA PATTERN IDENTIFICATION")
    print("-" * 80)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            pattern = analyzer.identify_missing_pattern(col)
            print(f"{col}: {pattern}")
    print()
    
    # Create missing indicators
    print("3. CREATING MISSING VALUE INDICATORS")
    print("-" * 80)
    indicators = analyzer.create_missing_indicators()
    print(f"Created {len(indicators.columns)} indicator variables")
    print()
    
    # Apply different imputation methods
    print("  a) Simple Imputation (Mean)...")
    analyzer.simple_imputation(strategy='mean')

    print("  b) Simple Imputation (Median)...")
    analyzer.simple_imputation(strategy='median')

    print("  c) Simple Imputation (Mode)...")
    analyzer.simple_imputation(strategy='most_frequent')

    print("  d) KNN Imputation...")
    analyzer.knn_imputation(n_neighbors=5)

    print("  e) Iterative Imputation (MICE)...")
    analyzer.iterative_imputation(max_iter=10)
    print()
    
    # Compare methods
    print("5. COMPARING IMPUTATION METHODS")
    print("-" * 80)
    comparison = analyzer.compare_imputation_methods()
    print(comparison)
    print()
    
    # Generate visualizations
    print("6. GENERATING VISUALIZATIONS")
    print("-" * 80)
    print("  Creating missing data visualizations...")
    analyzer.visualize_missing_data()
    plt.savefig('images/task1_missing_data_analysis.png', dpi=300, bbox_inches='tight')
    print("  Saved: task1_missing_data_analysis.png")
    
    print("  Creating imputation comparison for 'income'...")
    analyzer.visualize_imputation_comparison('income')
    plt.savefig('images/task1_imputation_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task1_imputation_comparison.png")
    
    print("  Creating correlation comparison...")
    analyzer.visualize_correlation_comparison()
    plt.savefig('images/task1_correlation_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task1_correlation_comparison.png")
    print()
    
    print("=" * 80)
    print("Task 1 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
