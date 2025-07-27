"""
Baseline Comparison Script for RetroReader
Thực hiện so sánh với các baseline khác nhau:
1. RoBERTa vanilla (không fine-tune)
2. Chỉ train Sketch module
3. Chỉ train Intensive module  
4. RetroReader đầy đủ (cả 2 modules)
"""

import os
import json
import argparse
import datasets
from retro_reader import RetroReader
from retro_reader.metrics import compute_squad_v2
import pandas as pd
from datetime import datetime

def evaluate_model(config_path, eval_dataset, output_name):
    """
    Evaluate một model configuration trên eval dataset
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {output_name}")
    print(f"Config: {config_path}")
    print(f"{'='*50}")
    
    try:
        # Load model với config
        retro_reader = RetroReader.load(
            eval_examples=eval_dataset,
            config_file=config_path,
            device="cuda" if not os.path.exists("/.dockerenv") else "cpu"  # Auto detect device
        )
        
        # Evaluate trên dev set
        results = retro_reader.evaluate(eval_dataset)
        
        # Save detailed results
        results_dir = f"outputs/baseline_comparison/{output_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f"{results_dir}/metrics.json", "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results for {output_name}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {output_name}: {str(e)}")
        return {"error": str(e)}

def run_baseline_comparison():
    """
    Chạy tất cả baseline experiments
    """
    print("Starting Baseline Comparison for RetroReader")
    print("Loading SQuAD v2.0 validation dataset...")
    
    # Load SQuAD v2.0 validation set
    squad_v2 = datasets.load_dataset("squad_v2")
    eval_dataset = squad_v2["validation"]
    
    # Để test nhanh, có thể giới hạn số samples
    # eval_dataset = eval_dataset.select(range(100))  # Uncomment để test với 100 samples
    
    print(f"Loaded {len(eval_dataset)} examples for evaluation")
    
    # Define all baseline configurations
    experiments = [
        {
            "name": "RoBERTa_Vanilla",
            "config": "configs/baseline_roberta_vanilla.yaml",
            "description": "RoBERTa base model without any fine-tuning"
        },
        {
            "name": "Sketch_Only", 
            "config": "configs/baseline_sketch_only.yaml",
            "description": "Only Sketch module trained"
        },
        {
            "name": "Intensive_Only",
            "config": "configs/baseline_intensive_only.yaml", 
            "description": "Only Intensive module trained"
        },
        {
            "name": "RetroReader_Full",
            "config": "configs/train_roberta_base_finetune.yaml",
            "description": "Full RetroReader with both modules trained"
        }
    ]
    
    # Store all results
    all_results = {}
    comparison_data = []
    
    # Run each experiment
    for exp in experiments:
        if os.path.exists(exp["config"]):
            results = evaluate_model(exp["config"], eval_dataset, exp["name"])
            all_results[exp["name"]] = {
                "results": results,
                "description": exp["description"],
                "config": exp["config"]
            }
            
            # Extract key metrics for comparison
            if "error" not in results:
                comparison_data.append({
                    "Model": exp["name"],
                    "Description": exp["description"], 
                    "EM": results.get("eval_exact", results.get("exact", "N/A")),
                    "F1": results.get("eval_f1", results.get("f1", "N/A")),
                    "HasAns_EM": results.get("eval_HasAns_exact", results.get("HasAns_exact", "N/A")),
                    "HasAns_F1": results.get("eval_HasAns_f1", results.get("HasAns_f1", "N/A")),
                    "NoAns_EM": results.get("eval_NoAns_exact", results.get("NoAns_exact", "N/A")),
                    "NoAns_F1": results.get("eval_NoAns_f1", results.get("NoAns_f1", "N/A"))
                })
        else:
            print(f"Warning: Config file {exp['config']} not found. Skipping {exp['name']}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/baseline_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save all results as JSON
    with open(f"{results_dir}/all_results_{timestamp}.json", "w", encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Create comparison table
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Save as CSV
        csv_path = f"{results_dir}/comparison_table_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Print comparison table
        print(f"\n{'='*80}")
        print("BASELINE COMPARISON RESULTS")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        # Calculate improvements
        if len(comparison_data) >= 2:
            print(f"\n{'='*50}")
            print("IMPROVEMENT ANALYSIS")
            print(f"{'='*50}")
            
            # Find best baseline và full model
            baseline_scores = []
            full_model_score = None
            
            for row in comparison_data:
                if row["Model"] == "RetroReader_Full":
                    full_model_score = row
                else:
                    baseline_scores.append(row)
            
            if full_model_score and baseline_scores:
                print(f"Full RetroReader vs Baselines:")
                for baseline in baseline_scores:
                    if isinstance(baseline["F1"], (int, float)) and isinstance(full_model_score["F1"], (int, float)):
                        f1_improvement = full_model_score["F1"] - baseline["F1"]
                        print(f"  vs {baseline['Model']}: +{f1_improvement:.2f} F1 points")
                    
                    if isinstance(baseline["EM"], (int, float)) and isinstance(full_model_score["EM"], (int, float)):
                        em_improvement = full_model_score["EM"] - baseline["EM"] 
                        print(f"    EM improvement: +{em_improvement:.2f} points")
        
        print(f"\nResults saved to:")
        print(f"  - {csv_path}")
        print(f"  - {results_dir}/all_results_{timestamp}.json")
        
        return df
    else:
        print("No valid results to compare")
        return None

def create_baseline_report(comparison_df):
    """
    Tạo báo cáo chi tiết về baseline comparison
    """
    if comparison_df is None:
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"outputs/baseline_comparison/baseline_report_{timestamp}.md"
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Baseline Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n")
        f.write("This report compares different configurations of the RetroReader model:\n\n")
        
        f.write("1. **RoBERTa Vanilla**: Base RoBERTa model without fine-tuning\n")
        f.write("2. **Sketch Only**: Only the sketch module is trained\n") 
        f.write("3. **Intensive Only**: Only the intensive module is trained\n")
        f.write("4. **RetroReader Full**: Both modules trained together\n\n")
        
        f.write("## Results\n\n")
        f.write("| Model | EM | F1 | HasAns EM | HasAns F1 | NoAns EM | NoAns F1 |\n")
        f.write("|-------|----|----|-----------|-----------|----------|----------|\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['Model']} | {row['EM']:.3f} | {row['F1']:.3f} | "
                   f"{row['HasAns_EM']:.3f} | {row['HasAns_F1']:.3f} | "
                   f"{row['NoAns_EM']:.3f} | {row['NoAns_F1']:.3f} |\n")
        
        f.write("\n## Key Insights\n\n")
        f.write("### Performance Analysis\n")
        f.write("- **Exact Match (EM)**: Percentage of predictions that match ground truth exactly\n")
        f.write("- **F1 Score**: Token-level F1 score considering partial matches\n") 
        f.write("- **HasAns**: Performance on answerable questions\n")
        f.write("- **NoAns**: Performance on unanswerable questions\n\n")
        
        f.write("### Module Contributions\n")
        f.write("- Compare Sketch Only vs Full model to see Intensive module contribution\n")
        f.write("- Compare Intensive Only vs Full model to see Sketch module contribution\n")
        f.write("- Compare both single modules vs Full model to see synergistic effects\n\n")
        
        f.write("## Configuration Details\n\n")
        f.write("All experiments used the same evaluation dataset (SQuAD v2.0 validation set) ")
        f.write("and consistent hyperparameters where applicable.\n")
    
    print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline comparison experiments')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run with limited samples for quick testing')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running in quick test mode with limited samples...")
    
    # Run the comparison
    comparison_df = run_baseline_comparison()
    
    # Generate detailed report
    create_baseline_report(comparison_df)
    
    print("\nBaseline comparison completed!")
