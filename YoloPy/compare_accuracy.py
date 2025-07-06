import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import os

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections(detections1: pd.DataFrame, detections2: pd.DataFrame, 
                    iou_threshold: float = 0.5, class_match: bool = True) -> Tuple[List, List, List]:
    """
    Match detections between two sets based on IoU and class.
    
    Returns:
        matched_pairs: List of (idx1, idx2, iou) for matched detections
        unmatched1: List of indices from detections1 that weren't matched
        unmatched2: List of indices from detections2 that weren't matched
    """
    matched_pairs = []
    used1 = set()
    used2 = set()
    
    # Reset indices to ensure we work with clean integer indices
    det1_reset = detections1.reset_index(drop=True)
    det2_reset = detections2.reset_index(drop=True)
    
    # Sort detections by confidence (highest first) but keep track of original indices
    det1_sorted = det1_reset.sort_values('confidence', ascending=False).reset_index()
    det1_sorted.rename(columns={'index': 'original_idx1'}, inplace=True)
    
    det2_sorted = det2_reset.sort_values('confidence', ascending=False).reset_index()
    det2_sorted.rename(columns={'index': 'original_idx2'}, inplace=True)
    
    for i, row1 in det1_sorted.iterrows():
        best_iou = 0
        best_match = -1
        best_original_idx2 = -1
        
        box1 = [row1['x1'], row1['y1'], row1['x2'], row1['y2']]
        
        for j, row2 in det2_sorted.iterrows():
            original_idx2 = row2['original_idx2']
            if original_idx2 in used2:
                continue
                
            # Check class match if required
            if class_match and row1['class'] != row2['class']:
                continue
                
            box2 = [row2['x1'], row2['y1'], row2['x2'], row2['y2']]
            iou = calculate_iou(box1, box2)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = j
                best_original_idx2 = original_idx2
        
        if best_match != -1:
            original_idx1 = row1['original_idx1']
            matched_pairs.append((original_idx1, best_original_idx2, best_iou))
            used1.add(original_idx1)
            used2.add(best_original_idx2)
    
    # Find unmatched detections
    unmatched1 = [i for i in range(len(det1_reset)) if i not in used1]
    unmatched2 = [i for i in range(len(det2_reset)) if i not in used2]
    
    return matched_pairs, unmatched1, unmatched2

def calculate_metrics(detections1: pd.DataFrame, detections2: pd.DataFrame, 
                     iou_threshold: float = 0.5) -> Dict:
    """
    Calculate comprehensive accuracy metrics between two detection sets.
    """
    matched_pairs, unmatched1, unmatched2 = match_detections(
        detections1, detections2, iou_threshold
    )
    
    # Basic counts
    num_matched = len(matched_pairs)
    num_det1 = len(detections1)
    num_det2 = len(detections2)
    
    # Calculate metrics
    precision = num_matched / num_det1 if num_det1 > 0 else 0
    recall = num_matched / num_det2 if num_det2 > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average IoU for matched detections
    avg_iou = np.mean([iou for _, _, iou in matched_pairs]) if matched_pairs else 0
    
    # Confidence statistics for matched vs unmatched
    if matched_pairs:
        matched_conf1 = [detections1.iloc[int(i)]['confidence'] for i, _, _ in matched_pairs]
        matched_conf2 = [detections2.iloc[int(j)]['confidence'] for _, j, _ in matched_pairs]
    else:
        matched_conf1 = []
        matched_conf2 = []
    
    unmatched_conf1 = [detections1.iloc[int(i)]['confidence'] for i in unmatched1]
    unmatched_conf2 = [detections2.iloc[int(i)]['confidence'] for i in unmatched2]
    return {
        'num_detections_1': num_det1,
        'num_detections_2': num_det2,
        'num_matched': num_matched,
        'num_unmatched_1': len(unmatched1),
        'num_unmatched_2': len(unmatched2),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'average_iou': avg_iou,
        'matched_pairs': matched_pairs,
        'unmatched_1': unmatched1,
        'unmatched_2': unmatched2,
        'matched_conf_1': matched_conf1,
        'matched_conf_2': matched_conf2,
        'unmatched_conf_1': unmatched_conf1,
        'unmatched_conf_2': unmatched_conf2
    }

def analyze_class_distribution(detections1: pd.DataFrame, detections2: pd.DataFrame) -> Dict:
    """Analyze class distribution differences between two detection sets."""
    class_counts1 = detections1['class'].value_counts().sort_index()
    class_counts2 = detections2['class'].value_counts().sort_index()
    
    all_classes = sorted(set(class_counts1.index) | set(class_counts2.index))
    
    class_comparison = {}
    for cls in all_classes:
        count1 = class_counts1.get(cls, 0)
        count2 = class_counts2.get(cls, 0)
        class_comparison[cls] = {
            'count_1': count1,
            'count_2': count2,
            'difference': count1 - count2,
            'ratio': count1 / count2 if count2 > 0 else float('inf') if count1 > 0 else 1.0
        }
    
    return class_comparison

def create_visualizations(metrics: Dict, detections1: pd.DataFrame, detections2: pd.DataFrame, 
                         class_comparison: Dict, output_dir: str = "comparison_results"):
    """Create comprehensive visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Metrics Summary Bar Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Metrics comparison
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Avg IoU']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['average_iou']]
    
    bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax1.set_title('Accuracy Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Detection counts comparison
    counts = [metrics['num_detections_1'], metrics['num_detections_2'], metrics['num_matched']]
    labels = ['C++ Detections', 'Python Detections', 'Matched Detections']
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    bars2 = ax2.bar(labels, counts, color=colors)
    ax2.set_title('Detection Counts Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Detections')
    
    for bar, count in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Confidence distribution comparison
    all_conf1 = detections1['confidence'].values
    all_conf2 = detections2['confidence'].values
    
    ax3.hist(all_conf1, bins=20, alpha=0.7, label='C++ Confidence', color='lightblue', density=True)
    ax3.hist(all_conf2, bins=20, alpha=0.7, label='Python Confidence', color='lightcoral', density=True)
    ax3.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Class distribution comparison
    classes = list(class_comparison.keys())
    counts1 = [class_comparison[cls]['count_1'] for cls in classes]
    counts2 = [class_comparison[cls]['count_2'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax4.bar(x - width/2, counts1, width, label='C++', color='lightblue')
    ax4.bar(x + width/2, counts2, width, label='Python', color='lightcoral')
    ax4.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Class ID')
    ax4.set_ylabel('Number of Detections')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. IoU Distribution for matched detections
    if metrics['matched_pairs']:
        plt.figure(figsize=(10, 6))
        ious = [iou for _, _, iou in metrics['matched_pairs']]
        plt.hist(ious, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean IoU: {np.mean(ious):.3f}')
        plt.title('IoU Distribution for Matched Detections', fontsize=14, fontweight='bold')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/iou_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_detailed_report(metrics: Dict, class_comparison: Dict, 
                           detections1: pd.DataFrame, detections2: pd.DataFrame,
                           output_file: str = "accuracy_report.txt"):
    """Generate a detailed text report of the comparison."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO DETECTION ACCURACY COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"C++ Implementation Detections: {metrics['num_detections_1']}\n")
        f.write(f"Python Implementation Detections: {metrics['num_detections_2']}\n")
        f.write(f"Matched Detections: {metrics['num_matched']}\n")
        f.write(f"Unmatched C++ Detections: {metrics['num_unmatched_1']}\n")
        f.write(f"Unmatched Python Detections: {metrics['num_unmatched_2']}\n\n")
        
        f.write("ACCURACY METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision (C++ vs Python): {metrics['precision']:.4f}\n")
        f.write(f"Recall (C++ vs Python): {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Average IoU for Matched Detections: {metrics['average_iou']:.4f}\n\n")
        
        f.write("CLASS DISTRIBUTION ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<8} {'C++':<8} {'Python':<8} {'Diff':<8} {'Ratio':<8}\n")
        f.write("-" * 40 + "\n")
        
        for cls, data in class_comparison.items():
            ratio_str = f"{data['ratio']:.2f}" if data['ratio'] != float('inf') else "∞"
            f.write(f"{cls:<8} {data['count_1']:<8} {data['count_2']:<8} {data['difference']:<8} {ratio_str:<8}\n")
        
        f.write("\nCONFIDENCE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        if metrics['matched_conf_1']:
            f.write(f"Average Confidence (Matched C++): {np.mean(metrics['matched_conf_1']):.4f}\n")
            f.write(f"Average Confidence (Matched Python): {np.mean(metrics['matched_conf_2']):.4f}\n")
        
        if metrics['unmatched_conf_1']:
            f.write(f"Average Confidence (Unmatched C++): {np.mean(metrics['unmatched_conf_1']):.4f}\n")
        
        if metrics['unmatched_conf_2']:
            f.write(f"Average Confidence (Unmatched Python): {np.mean(metrics['unmatched_conf_2']):.4f}\n")
        
        f.write("\nDETAILED MATCHED PAIRS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'C++ Idx':<8} {'Py Idx':<8} {'IoU':<8}\n")
        f.write("-" * 24 + "\n")
        
        for i, j, iou in metrics['matched_pairs']:
            f.write(f"{i:<8} {j:<8} {iou:.4f}\n")

def main():
    """Main function to run the accuracy comparison."""
    print("Loading detection results...")
    
    # Load CSV files
    try:
        cpp_detections = pd.read_csv('output/output_cpp.csv')
        print(f"✓ Loaded C++ detections from output/output_cpp.csv")
    except FileNotFoundError:
        print("✗ Could not find output/output_cpp.csv")
        print("Please run the C++ implementation first")
        return
    
    try:
        python_detections = pd.read_csv('YoloPy/output/output_py.csv')
        print(f"✓ Loaded Python detections from YoloPy/output/output_py.csv")
    except FileNotFoundError:
        print("✗ Could not find YoloPy/output/output_py.csv")
        print("Please run the Python implementation first")
        return
    
    print(f"\nDataset Summary:")
    print(f"C++ detections: {len(cpp_detections)}")
    print(f"Python detections: {len(python_detections)}")
    
    # Show sample data to verify format
    print(f"\nC++ CSV columns: {list(cpp_detections.columns)}")
    print(f"Python CSV columns: {list(python_detections.columns)}")
    
    # Calculate metrics with different IoU thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    
    print("\nCalculating accuracy metrics...")
    
    for iou_thresh in iou_thresholds:
        print(f"\n--- IoU Threshold: {iou_thresh} ---")
        
        metrics = calculate_metrics(cpp_detections, python_detections, iou_thresh)
        
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Average IoU: {metrics['average_iou']:.4f}")
        print(f"Matched detections: {metrics['num_matched']}")
    
    # Use IoU threshold of 0.5 for detailed analysis
    metrics = calculate_metrics(cpp_detections, python_detections, 0.5)
    class_comparison = analyze_class_distribution(cpp_detections, python_detections)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_visualizations(metrics, cpp_detections, python_detections, class_comparison)
    
    # Generate detailed report
    print("Generating detailed report...")
    generate_detailed_report(metrics, class_comparison, cpp_detections, python_detections)
    
    print("\nAccuracy comparison completed!")
    print("Check 'accuracy_report.txt' for detailed results")
    print("Check 'comparison_results/' folder for visualization plots")

if __name__ == "__main__":
    main() 