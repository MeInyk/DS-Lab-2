"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}
        
    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        z_scores = np.abs(stats.zscore(numeric_data, nan_policy='omit'))

        outliers = pd.DataFrame(
            z_scores > threshold,
            index=numeric_data.index,
            columns=numeric_data.columns
        )

        self.outlier_masks['zscore'] = outliers

        return outliers
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
            multiplier (float): IQR multiplier for boundary calculation
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        q1 = numeric_data.quantile(0.25)
        q3 = numeric_data.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outliers = (numeric_data < lower_bound) | (numeric_data > upper_bound)

        self.outlier_masks['iqr'] = outliers

        return outliers
    
    def detect_outliers_isolation_forest(self, contamination: float = 0.1, 
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.empty or len(numeric_data.columns) == 0:
            return pd.DataFrame(False, index=self.data.index, columns=numeric_data.columns)

        filled_data = numeric_data.fillna(numeric_data.median()).fillna(0)

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

        predictions = model.fit_predict(filled_data)

        outliers = pd.DataFrame(
            np.repeat((predictions == -1)[:, np.newaxis], len(numeric_data.columns), axis=1),
            index=numeric_data.index,
            columns=numeric_data.columns
        )

        self.outlier_masks['isolation_forest'] = outliers

        return outliers
    
    def detect_outliers_lof(self, n_neighbors: int = 20, 
                           contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        
        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.empty or len(numeric_data.columns) == 0:
            outliers = pd.DataFrame(
                False,
                index=self.data.index,
                columns=numeric_data.columns
            )
            self.outlier_masks['lof'] = outliers
            return outliers

        filled_data = numeric_data.fillna(numeric_data.median()).fillna(0)

        if len(filled_data) < 3:
            outliers = pd.DataFrame(
                False,
                index=numeric_data.index,
                columns=numeric_data.columns
            )
            self.outlier_masks['lof'] = outliers
            return outliers

        model = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(filled_data) - 1),
            contamination=contamination
        )

        predictions = model.fit_predict(filled_data)

        outliers = pd.DataFrame(
            np.repeat(
                (predictions == -1)[:, np.newaxis],
                len(numeric_data.columns),
                axis=1
            ),
            index=numeric_data.index,
            columns=numeric_data.columns
        )

        self.outlier_masks['lof'] = outliers

        return outliers
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        
        Returns:
            DataFrame with outlier counts for each method
        """
        summary = {}

        for method, mask in self.outlier_masks.items():
            summary[method] = mask.sum()

        return pd.DataFrame(summary)
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters:
            method (str): Detection method to use
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        if method not in self.outlier_masks:
            if method == 'iqr':
                self.detect_outliers_iqr()
            elif method == 'zscore':
                self.detect_outliers_zscore()
            elif method == 'isolation_forest':
                self.detect_outliers_isolation_forest()
            elif method == 'lof':
                self.detect_outliers_lof()
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

        outlier_mask = self.outlier_masks[method]

        rows_with_outliers = outlier_mask.any(axis=1)

        cleaned_data = self.data.loc[~rows_with_outliers].copy()

        self.treated_datasets[f'removed_{method}'] = cleaned_data

        return cleaned_data
    
    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.
        
        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds
            
        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        capped_data = self.data.copy()
        numeric_cols = capped_data.select_dtypes(include=[np.number]).columns

        if method == 'iqr':
            q1 = capped_data[numeric_cols].quantile(0.25)
            q3 = capped_data[numeric_cols].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

        elif method == 'zscore':
            mean = capped_data[numeric_cols].mean()
            std = capped_data[numeric_cols].std(ddof=0)

            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

        else:
            raise ValueError(f"Unsupported capping method: {method}")

        capped_data[numeric_cols] = capped_data[numeric_cols].clip(
            lower=lower_bound,
            upper=upper_bound,
            axis=1
        )

        self.treated_datasets['capped'] = capped_data

        return capped_data
    
    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        
        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        transformed_data = self.data.copy()

        numeric_cols = transformed_data.select_dtypes(
            include=[np.number]
        ).columns

        for column in numeric_cols:
            values = transformed_data[column]

            if values.dropna().empty:
                continue

            min_value = values.min()

            shift = -min_value if min_value < 0 else 0

            transformed_data[column] = np.log1p(values + shift)

        self.treated_datasets['log_transform'] = transformed_data

        return transformed_data
    
    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        
        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        transformed_data = self.data.copy()

        numeric_cols = transformed_data.select_dtypes(
            include=[np.number]
        ).columns

        for column in numeric_cols:
            values = transformed_data[column]

            valid_values = values.dropna()

            if len(valid_values) < 2 or valid_values.nunique() < 2:
                continue

            transformer = PowerTransformer(
                method='yeo-johnson',
                standardize=True
            )

            transformed_values = transformer.fit_transform(
                valid_values.to_numpy().reshape(-1, 1)
            )

            transformed_data.loc[valid_values.index, column] = (
                transformed_values.ravel()
            )

        self.treated_datasets['boxcox_transform'] = transformed_data

        return transformed_data
    
    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        
        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)
            
        Returns:
            Dictionary with performance metrics
        """
        if dataset_name == 'original':
            dataset = self.original_data.copy()
        elif dataset_name in self.treated_datasets:
            dataset = self.treated_datasets[dataset_name].copy()
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        numeric_data = dataset.select_dtypes(include=[np.number])

        if target_col not in numeric_data.columns:
            raise ValueError("Target column must be numeric")

        numeric_data = numeric_data.dropna(subset=[target_col])

        X = numeric_data.drop(columns=[target_col])
        y = numeric_data[target_col]

        if X.empty or len(X) < 5:
            raise ValueError("Not enough data for model evaluation")

        X = X.fillna(X.median()).fillna(0)

        model = LinearRegression()

        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring='r2'
        )

        performance = {
            'r2_score': float(np.mean(scores)),
            'r2_std': float(np.std(scores)),
            'mean_cv_score': float(np.mean(scores))
        }

        return performance
    
    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        numeric_cols = self.data.select_dtypes(
            include=[np.number]
        ).columns[:6]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx, column in enumerate(numeric_cols):
            ax = axes[idx]

            sns.boxplot(
                y=self.data[column],
                ax=ax,
                color='skyblue'
            )

            ax.set_title(
                f'Boxplot: {column}',
                fontweight='bold'
            )

            ax.set_ylabel(column)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            'Outlier Detection Using Boxplots',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()
        
        return fig
    
    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx, column in enumerate(numeric_cols):
            ax = axes[idx]

            sns.violinplot(
                y=self.data[column].dropna(),
                ax=ax,
                color='lightblue',
                inner='box'
            )

            ax.set_title(
                f'Violin Plot: {column}',
                fontweight='bold'
            )

            ax.set_ylabel(column)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            'Data Distribution Using Violin Plots',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()
        
        return fig
    
    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """

        if x_col not in self.data.columns:
            print(f"Column '{x_col}' not found")
            return None

        if y_col not in self.data.columns:
            print(f"Column '{y_col}' not found")
            return None

        methods = ['zscore', 'iqr', 'isolation_forest', 'lof']

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        for idx, method in enumerate(methods):
            ax = axes[idx]

            if method not in self.outlier_masks:
                if method == 'zscore':
                    self.detect_outliers_zscore()
                elif method == 'iqr':
                    self.detect_outliers_iqr()
                elif method == 'isolation_forest':
                    self.detect_outliers_isolation_forest()
                elif method == 'lof':
                    self.detect_outliers_lof()

            mask = self.outlier_masks[method]

            outlier_rows = mask.any(axis=1)

            normal_data = self.data.loc[~outlier_rows]
            outlier_data = self.data.loc[outlier_rows]

            ax.scatter(
                normal_data[x_col],
                normal_data[y_col],
                color='blue',
                alpha=0.5,
                label='Normal'
            )

            ax.scatter(
                outlier_data[x_col],
                outlier_data[y_col],
                color='red',
                alpha=0.8,
                label='Outliers'
            )

            ax.set_title(method.replace('_', ' ').title())
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle(
            'Outlier Detection Comparison',
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout()
                                       
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        
        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        if method not in self.outlier_masks:
            print(f"Outlier detection results for '{method}' not found")
            return None

        mask = self.outlier_masks[method]
        outlier_rows = mask.any(axis=1)

        normal_data = self.data.loc[~outlier_rows]
        outlier_data = self.data.loc[outlier_rows]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            normal_data[x_col],
            normal_data[y_col],
            normal_data[z_col],
            color='blue',
            alpha=0.5,
            label='Normal'
        )

        ax.scatter(
            outlier_data[x_col],
            outlier_data[y_col],
            outlier_data[z_col],
            color='red',
            alpha=0.8,
            label='Outliers'
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)

        ax.set_title(
            f'3D Outlier Detection: {method.replace("_", " ").title()}',
            fontweight='bold'
        )

        ax.legend()
        plt.tight_layout()
                                 
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
