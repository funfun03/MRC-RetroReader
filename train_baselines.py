"""
Training Script cho Baseline Experiments
Trains individual modules hoặc combinations của RetroReader
"""

import os
import argparse
import datasets
import torch
from retro_reader import RetroReader

def train_baseline_model(config_path, module_type, output_name):
    """
    Train một baseline model với specific config
    
    Args:
        config_path: đường dẫn đến config file
        module_type: "sketch", "intensive", hoặc "all" 
        output_name: tên để save results
    """
    print(f"\n{'='*50}")
    print(f"Training {output_name}")
    print(f"Config: {config_path}")
    print(f"Module: {module_type}")
    print(f"{'='*50}")
    
    # Load SQuAD v2.0 dataset
    print("Loading SQuAD v2.0 dataset...")
    squad_v2 = datasets.load_dataset("squad_v2")
    
    # Có thể giảm data size để test nhanh
    # squad_v2["train"] = squad_v2["train"].select(range(1000))  # Uncomment để test
    # squad_v2["validation"] = squad_v2["validation"].select(range(200))
    
    print(f"Train examples: {len(squad_v2['train'])}")
    print(f"Validation examples: {len(squad_v2['validation'])}")
    
    try:
        # Load RetroReader
        print("Loading RetroReader...")
        retro_reader = RetroReader.load(
            train_examples=squad_v2["train"],
            eval_examples=squad_v2["validation"], 
            config_file=config_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Train specified modules
        print(f"Starting training for module: {module_type}")
        retro_reader.train(module=module_type)
        
        # Evaluate after training
        print("Evaluating trained model...")
        eval_results = retro_reader.evaluate(squad_v2["validation"])
        
        # Save results
        results_dir = f"outputs/baseline_training/{output_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        with open(f"{results_dir}/final_metrics.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Training completed for {output_name}")
        print("Final Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
        return eval_results
        
    except Exception as e:
        print(f"Error training {output_name}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--baseline', type=str, choices=['sketch', 'intensive', 'full'], 
                       required=True, help='Which baseline to train')
    parser.add_argument('--quick-test', action='store_true',
                       help='Use reduced dataset for quick testing')
    
    args = parser.parse_args()
    
    # Define baseline configurations
    baselines = {
        'sketch': {
            'config': 'configs/baseline_sketch_only.yaml',
            'module': 'sketch',
            'name': 'Sketch_Only_Trained'
        },
        'intensive': {
            'config': 'configs/baseline_intensive_only.yaml', 
            'module': 'intensive',
            'name': 'Intensive_Only_Trained'
        },
        'full': {
            'config': 'configs/train_roberta_base_finetune.yaml',
            'module': 'all', 
            'name': 'RetroReader_Full_Trained'
        }
    }
    
    baseline_config = baselines[args.baseline]
    
    if args.quick_test:
        print("Running in quick test mode...")
        # Modify dataset size ở đây
    
    # Train the specified baseline
    results = train_baseline_model(
        config_path=baseline_config['config'],
        module_type=baseline_config['module'],
        output_name=baseline_config['name']
    )
    
    if results:
        print(f"\nTraining successful for {args.baseline} baseline!")
    else:
        print(f"\nTraining failed for {args.baseline} baseline!")

if __name__ == "__main__":
    main()
