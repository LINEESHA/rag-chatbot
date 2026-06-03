#!/usr/bin/env python3
"""
Fine-Tuning Script for LLMs
Run this to train your custom model on domain-specific data
"""

import os
import sys
import argparse
from pathlib import Path
from llm_finetuner import LLMFineTuner
import json
from datetime import datetime

def create_sample_data():
    """Create sample training data if not exists"""
    sample_text = """
Machine Learning is a subset of Artificial Intelligence.
Deep Learning uses neural networks with multiple layers.
Natural Language Processing helps computers understand text.
Computer Vision enables machines to interpret images.
Transformers are the foundation of modern LLMs.
Fine-tuning allows models to learn domain-specific knowledge.
RAG combines retrieval with generation for better answers.
Vector databases store embeddings for semantic search.
Attention mechanisms allow models to focus on relevant parts.
Tokenization converts text into model-readable tokens.
Embeddings represent words as numerical vectors.
FAISS enables fast similarity search in high dimensions.
Ollama provides easy local LLM inference.
Streamlit makes ML demos interactive and shareable.
GitHub is essential for version control and collaboration.
Docker containerizes applications for easy deployment.
APIs allow different systems to communicate.
FastAPI provides fast web framework for ML models.
Monitoring tracks model performance in production.
Evaluation metrics measure model quality objectively.
"""
    
    with open("training_data.txt", "w") as f:
        f.write(sample_text)
    
    print("✓ Sample training_data.txt created")
    print("  Replace with your own domain-specific data for best results")

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs on custom data"
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-v0.1",
        help="Model name (mistral, llama2, etc.)"
    )
    parser.add_argument(
        "--data",
        default="training_data.txt",
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        default="./my_finetuned_model",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Create sample data if doesn't exist
    if not os.path.exists(args.data):
        print("⚠️  Training data not found. Creating sample...")
        create_sample_data()
    
    print("\n" + "="*50)
    print("🎓 LLM Fine-Tuning")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("="*50 + "\n")
    
    # Check if GPU available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()
    
    # Initialize
    finetuner = LLMFineTuner(
        model_name=args.model,
        output_dir=args.output
    )
    
    # Load model
    print("📥 Loading model and tokenizer...")
    finetuner.load_model_and_tokenizer()
    print("✓ Model loaded\n")
    
    # Train
    print("🔥 Starting fine-tuning...")
    print(f"   This may take 30 minutes to several hours depending on hardware\n")
    
    try:
        finetuner.train(
            data_file=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    
    # Save model
    print("\n💾 Saving fine-tuned model...")
    finetuner.save_model()
    print(f"✓ Model saved to: {args.output}\n")
    
    # Summary
    print("="*50)
    print("✅ Fine-tuning Complete!")
    print("="*50)
    print(f"\nModel saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"1. Test inference:")
    print(f"   python -c \"from llm_finetuner import InferenceAPI; api = InferenceAPI('{args.output}'); print(api.predict('Your prompt here'))\"")
    print(f"\n2. Deploy API:")
    print(f"   python inference_api.py")
    print(f"\n3. Push to HuggingFace Hub:")
    print(f"   model.push_to_hub('your-username/model-name')")
    print(f"\n4. Add to GitHub for portfolio")
    print("="*50)

if __name__ == "__main__":
    main()
