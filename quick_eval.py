"""
Quick Evaluation Script 
Để evaluation các pre-trained models mà không cần training từ đầu
"""

import os
import json
import datasets
from retro_reader import RetroReader
import pandas as pd
from datetime import datetime

def quick_evaluate():
    """
    Nhanh chóng evaluate các pre-trained models
    """
    print("Loading SQuAD v2.0 validation dataset...")
    squad_v2 = datasets.load_dataset("squad_v2")
    eval_dataset = squad_v2["validation"]
    
    # Để test nhanh, giới hạn samples
    eval_dataset = eval_dataset.select(range(100))  # Chỉ test với 100 samples
    print(f"Using {len(eval_dataset)} examples for quick evaluation")
    
    # Evaluate pre-trained RetroReader
    print("\nEvaluating pre-trained RetroReader...")
    try:
        retro_reader = RetroReader.load(
            eval_examples=eval_dataset,
            config_file="configs/inference_en_roberta.yaml",
            device="cpu"  # Sử dụng CPU để tránh memory issues
        )
        
        results = retro_reader.evaluate(eval_dataset)
        
        print("Pre-trained RetroReader Results:")
        key_metrics = ['exact', 'f1', 'HasAns_exact', 'HasAns_f1', 'NoAns_exact', 'NoAns_f1']
        for metric in key_metrics:
            if metric in results:
                print(f"  {metric}: {results[metric]:.4f}")
        
        # Save results
        os.makedirs("outputs/quick_eval", exist_ok=True)
        with open("outputs/quick_eval/pretrained_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        return results
        
    except Exception as e:
        print(f"Error evaluating pre-trained model: {str(e)}")
        return None

def compare_with_simple_baseline():
    """
    So sánh với một baseline đơn giản sử dụng existing pre-trained model
    """
    print("\n" + "="*50)
    print("QUICK COMPARISON WITH SIMPLE BASELINE")
    print("="*50)
    
    # Load dataset
    squad_v2 = datasets.load_dataset("squad_v2")
    eval_dataset = squad_v2["validation"].select(range(100))
    
    results_comparison = []
    
    # 1. Pre-trained RetroReader
    print("\n1. Evaluating Pre-trained RetroReader...")
    try:
        retro_reader = RetroReader.load(
            eval_examples=eval_dataset,
            config_file="configs/inference_en_roberta.yaml",
            device="cpu"
        )
        retro_results = retro_reader.evaluate(eval_dataset)
        
        results_comparison.append({
            "Model": "RetroReader (Pre-trained)",
            "EM": retro_results.get('exact', 0),
            "F1": retro_results.get('f1', 0),
            "HasAns_EM": retro_results.get('HasAns_exact', 0),
            "HasAns_F1": retro_results.get('HasAns_f1', 0),
            "NoAns_EM": retro_results.get('NoAns_exact', 0),
            "NoAns_F1": retro_results.get('NoAns_f1', 0)
        })
        
    except Exception as e:
        print(f"Error with RetroReader: {e}")
    
    # 2. Try với baseline RoBERTa từ HuggingFace (nếu có thể)
    try:
        from transformers import pipeline
        print("\n2. Evaluating HuggingFace RoBERTa baseline...")
        
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        
        correct_em = 0
        total_f1 = 0
        
        for example in eval_dataset.select(range(20)):  # Chỉ test 20 examples
            question = example["question"]
            context = example["context"]
            
            try:
                result = qa_pipeline(question=question, context=context)
                predicted_answer = result["answer"]
                
                # Simple EM check
                true_answers = [ans["text"] for ans in example["answers"]]
                if any(predicted_answer.strip().lower() == ans.strip().lower() for ans in true_answers):
                    correct_em += 1
                    
            except:
                continue
        
        baseline_em = correct_em / 20
        
        results_comparison.append({
            "Model": "RoBERTa-base-squad2 (HF)",
            "EM": baseline_em,
            "F1": "N/A",
            "HasAns_EM": "N/A", 
            "HasAns_F1": "N/A",
            "NoAns_EM": "N/A",
            "NoAns_F1": "N/A"
        })
        
    except Exception as e:
        print(f"Could not evaluate HF baseline: {e}")
    
    # Display comparison
    if results_comparison:
        df = pd.DataFrame(results_comparison)
        print("\nQUICK COMPARISON RESULTS:")
        print(df.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"outputs/quick_eval/quick_comparison_{timestamp}.csv", index=False)
        print(f"\nResults saved to outputs/quick_eval/quick_comparison_{timestamp}.csv")
        
        return df
    
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick evaluation of models')
    parser.add_argument('--mode', choices=['eval', 'compare'], default='compare',
                       help='Mode: eval (just RetroReader) or compare (with baseline)')
    
    args = parser.parse_args()
    
    if args.mode == 'eval':
        quick_evaluate()
    else:
        compare_with_simple_baseline()
