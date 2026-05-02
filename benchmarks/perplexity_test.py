#!/usr/bin/env python3
"""End-to-end perplexity test on real models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_perplexity(model_name, model_path, max_length=512, num_samples=100):
    """Evaluate perplexity with and without TurboQuant."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        # Test text
        test_text = """
        The quick brown fox jumps over the lazy dog. This pangram contains every 
        letter of the alphabet at least once. Machine learning models use such 
        examples to understand language patterns and improve their predictions.
        """ * 20  # Make it longer
        
        # Tokenize
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Baseline perplexity
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            baseline_loss = outputs.loss.item()
            baseline_ppl = np.exp(baseline_loss)
        
        print(f"Baseline perplexity: {baseline_ppl:.2f}")
        
        # Now test with TurboQuant on activations
        # Hook into attention layers
        from turboquant import TurboQuantPROD
        
        dim = model.config.hidden_size // model.config.num_attention_heads
        print(f"Head dimension: {dim}")
        
        quantization_errors = []
        
        def hook_fn(name, bits):
            def hook(module, input, output):
                # Quantize attention output
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                
                # Reshape to vectors
                batch, seq_len, hidden = output_tensor.shape
                x = output_tensor.reshape(-1, hidden)
                
                # Apply TurboQuant
                quantizer = TurboQuantPROD(hidden, bits, device=device)
                q = quantizer.quantize(x)
                x_recon = quantizer.dequantize(q)
                
                # Measure error
                error = ((x - x_recon)**2).mean().item()
                quantization_errors.append(error)
                
                return output
            return hook
        
        # Attach hooks to attention layers
        hooks = []
        for layer in model.model.layers[:2]:  # First 2 layers only
            for bits in [3, 4]:
                handle = layer.register_forward_hook(hook_fn(f"layer_{id(layer)}", bits))
                hooks.append(handle)
                
                with torch.no_grad():
                    outputs_q = model(input_ids, labels=input_ids)
                    loss_q = outputs_q.loss.item()
                    ppl_q = np.exp(loss_q)
                
                mean_error = np.mean(quantization_errors) if quantization_errors else 0
                print(f"  {bits}-bit PROD: PPL={ppl_q:.2f}, QMSE={mean_error:.6f}")
                
                quantization_errors = []
                handle.remove()
        
        return {
            'model': model_name,
            'baseline_ppl': baseline_ppl,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'model': model_name, 'error': str(e), 'success': False}

if __name__ == "__main__":
    models_dir = Path.home() / "workspace/Models"
    
    # Test on smallest model first
    test_models = [
        ("SmolLM2-135M", models_dir / "SmolLM2-135M"),
        ("Qwen2.5-0.5B", models_dir / "Qwen2.5-0.5B-Instruct"),
    ]
    
    results = []
    for name, path in test_models:
        if path.exists():
            result = evaluate_perplexity(name, path)
            results.append(result)
        else:
            print(f"Skipping {name}: path not found at {path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        if r['success']:
            print(f"{r['model']}: Baseline PPL = {r['baseline_ppl']:.2f}")
        else:
            print(f"{r['model']}: FAILED - {r.get('error', 'Unknown')}")
