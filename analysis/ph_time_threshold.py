import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def load_and_prepare_data(filepath):
    """Load and prepare data for analysis."""
    data = pd.read_csv(filepath)
    data = data[data['Overlap'] != 0]
    data = data.drop(['Overlap', 'Biodegradability_yes'], axis=1)

    # Melt to long format
    data_long = data.melt(
        id_vars=['Lab book Number', 'Polymer Name', 'Trimer', 'PNAS number', 'PNAS biodeg'],
        var_name='Time',
        value_name='pH'
    )

    # Clean and convert data types
    data_long['Time'] = pd.to_numeric(data_long['Time'], errors='coerce')
    data_long = data_long.dropna(subset=['Time', 'pH'])  # Drop rows with NaN in Time or pH
    data_long['PNAS biodeg'] = data_long['PNAS biodeg'].map({'yes': 1, 'no': 0})

    return data_long


def evaluate_threshold(data, threshold):
    """Evaluate a pH threshold for biodegradability prediction."""
    data['pred'] = (data['pH'] < threshold).astype(int)
    y_pred = data['pred']
    y_true = data['PNAS biodeg']

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    balance = fp - fn

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'balance': balance
    }


def find_optimal_threshold(data_long, ph_range=(4.0, 6.0), step=0.01, balance_tolerance=7, max_time=1200):
    """Find optimal pH threshold and time point."""
    thresholds = np.arange(ph_range[0], ph_range[1] + step, step)
    unique_time_points = sorted(data_long[data_long['Time'] <= max_time]['Time'].unique())
    results = []

    # Test each time point and threshold combination
    for time_point in unique_time_points:
        data_at_time = data_long[data_long['Time'] == time_point].copy()

        # Skip if not enough samples
        if len(data_at_time) < 10:
            continue

        for threshold in thresholds:
            metrics = evaluate_threshold(data_at_time, threshold)

            # Only keep results with acceptable balance
            if abs(metrics['balance']) <= balance_tolerance:
                results.append({
                    'Time': time_point,
                    'Threshold': round(threshold, 2),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1_Score': metrics['f1_score'],
                    'TP': metrics['tp'],
                    'TN': metrics['tn'],
                    'FP': metrics['fp'],
                    'FN': metrics['fn'],
                    'Balance': metrics['balance']
                })

    return pd.DataFrame(results)


def plot_results(results_df):
    """Plot heatmap of accuracy across time and pH thresholds."""
    pivot_accuracy = results_df.pivot_table(
        values='Accuracy',
        index='Threshold',
        columns='Time',
        aggfunc='mean'
    )
    im = plt.imshow(
        pivot_accuracy.values,
        cmap='RdYlGn',
        aspect='auto',
        origin='lower',
        vmin=pivot_accuracy.min().min(),
        vmax=pivot_accuracy.max().max()
    )
    plt.ylabel('pH', fontsize=13)
    plt.xlabel('Time (min)', fontsize=13)

    # Set tick labels
    y_ticks = np.linspace(0, len(pivot_accuracy.index) - 1, 5, dtype=int)
    plt.yticks(y_ticks, labels=[f"{pivot_accuracy.index[i]:.2f}" for i in y_ticks], fontsize=12)
    x_ticks = np.linspace(0, len(pivot_accuracy.columns) - 1, 5, dtype=int)
    plt.xticks(x_ticks, labels=[f"{pivot_accuracy.columns[i]:.0f}" for i in x_ticks], fontsize=12)

    cb = plt.colorbar(im)
    cb.set_label(label='Accuracy', fontsize=11)

    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    CSV_FILE = '../data/full_pH_data.csv'
    PH_RANGE = (4.0, 6.0)
    STEP = 0.01
    BALANCE_TOLERANCE = 7
    MAX_TIME = 1200  # Only use data up to 1200 minutes

    print("Loading and preparing data...")
    data_long = load_and_prepare_data(CSV_FILE)
    print(f"Loaded {len(data_long)} measurements across {data_long['Time'].nunique()} time points")
    print(f"Filtering to time points ≤ {MAX_TIME} minutes...")

    print("\nFinding optimal threshold...")
    results_df = find_optimal_threshold(data_long, PH_RANGE, STEP, BALANCE_TOLERANCE, MAX_TIME)

    if results_df.empty:
        print(f"No combination found with balance within ±{BALANCE_TOLERANCE}")
        return

    # Find best result by accuracy
    best_idx = results_df['Accuracy'].idxmax()
    best_result = results_df.loc[best_idx]

    # Print results
    print("\n" + "=" * 70)
    print("OPTIMAL PREDICTION MODEL")
    print("=" * 70)
    print(f"Best Time Point:        {best_result['Time']:.3f}")
    print(f"Best pH Threshold:      {best_result['Threshold']:.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:             {best_result['Accuracy'] * 100:.2f}%")
    print(f"  Precision:            {best_result['Precision'] * 100:.2f}%")
    print(f"  Recall (Sensitivity): {best_result['Recall'] * 100:.2f}%")
    print(f"  F1 Score:             {best_result['F1_Score']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {int(best_result['TP'])}")
    print(f"  True Negatives (TN):  {int(best_result['TN'])}")
    print(f"  False Positives (FP): {int(best_result['FP'])}")
    print(f"  False Negatives (FN): {int(best_result['FN'])}")
    print(f"  Balance (FP-FN):      {int(best_result['Balance'])}")
    print("=" * 70)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_results(results_df)

    # Show top 10 alternatives
    print("\n" + "=" * 70)
    print("TOP 10 ALTERNATIVE MODELS")
    print("=" * 70)
    top_10 = results_df.nlargest(10, 'Accuracy')[['Time', 'Threshold', 'Accuracy', 'F1_Score', 'Balance']]
    print(top_10.to_string(index=False))


if __name__ == "__main__":
    main()