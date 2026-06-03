# 🎓 Fine-Tuned LLM on Custom Domain Data

Production-ready fine-tuning pipeline for open-source LLMs (Mistral, Llama2) on custom datasets. Train, evaluate, and deploy domain-specific language models without expensive APIs.

## Features

✅ **Easy Fine-Tuning** - Simple API for training custom models  
✅ **Multiple Models** - Support for Mistral 7B, Llama2, etc.  
✅ **Free** - No API costs, use open-source models  
✅ **Production Ready** - Export and deploy anywhere  
✅ **Inference API** - Easy prediction interface  
✅ **Performance Tracking** - Monitor training metrics  

## Tech Stack

- **Base Models:** Mistral 7B / Llama2
- **Framework:** Hugging Face Transformers
- **Training:** PyTorch with distributed support
- **Optimization:** LoRA/QLoRA (optional, for efficiency)
- **Deployment:** FastAPI + Uvicorn

## Installation

### Prerequisites
- Python 3.9+
- 16GB+ RAM (GPU strongly recommended)
- NVIDIA GPU with CUDA 11.8+ (for fast training)
- 20GB+ disk space (for model weights)

### Step 1: Clone & Setup
```bash
git clone <your-repo>
cd fine-tuned-llm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Training Data
Create `training_data.txt` with your domain-specific text:
```
Your training text here. One sentence per line or full documents.
Each line will be treated as training samples.
The model learns patterns from this data.
...
```

### Step 4: Train Model
```python
from llm_finetuner import LLMFineTuner

# Initialize
finetuner = LLMFineTuner(
    model_name="mistralai/Mistral-7B-v0.1",
    output_dir="./my_trained_model"
)

# Load model
finetuner.load_model_and_tokenizer()

# Train on your data
finetuner.train(
    data_file="training_data.txt",
    epochs=3,
    batch_size=4,
    learning_rate=5e-5
)

# Save model
finetuner.save_model()
```

### Step 5: Use Fine-Tuned Model for Inference
```python
from llm_finetuner import InferenceAPI

# Load your fine-tuned model
api = InferenceAPI("./my_trained_model")

# Generate predictions
prompt = "Your input text here"
response = api.predict(prompt, max_length=200)
print(response)
```

## Project Structure
```
fine-tuned-llm/
├── llm_finetuner.py          # Training & inference code
├── inference_api.py           # FastAPI server (optional)
├── training_data.txt          # Your training data
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── train.py                   # Training script
├── my_trained_model/          # Output (after training)
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer_config.json
└── .gitignore
```

## Requirements File (requirements.txt)
```
torch==2.0.1
transformers==4.34.0
datasets==2.14.0
accelerate==0.24.0
tensorboard==2.14.0
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
peft==0.7.0  # For LoRA
bitsandbytes==0.41.0  # For QLoRA
```

## Training Guide

### Dataset Preparation
```
1. Collect domain-specific text (minimum 10k tokens recommended)
2. Save as plain text file (training_data.txt)
3. One document/paragraph per sample OR one sentence per line
```

### Hyperparameter Tuning
```python
# For your domain, adjust these:
epochs = 3          # More epochs = more learning (but risk overfitting)
batch_size = 4      # Larger batch = stable training (but needs more RAM)
learning_rate = 5e-5 # Lower LR = safer, higher LR = faster convergence
```

### Training on Different Hardware
```python
# GPU (NVIDIA)
device = "cuda"  # Automatic

# CPU (slower, but works)
device = "cpu"

# Multiple GPUs
accelerate launch train.py
```

## Inference API (Optional FastAPI Server)

### Create inference_api.py
```python
from fastapi import FastAPI
from llm_finetuner import InferenceAPI

app = FastAPI()
api = InferenceAPI("./my_trained_model")

@app.post("/predict")
def predict(text: str):
    response = api.predict(text, max_length=200)
    return {"input": text, "output": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run API
```bash
python inference_api.py
# Server at http://localhost:8000
# Test: curl -X POST "http://localhost:8000/predict?text=Hello"
```

## Performance Benchmarks

| Metric | GPU (A100) | GPU (RTX 3090) | CPU |
|--------|-----------|----------------|-----|
| Training Time (1 epoch) | ~30 min | ~1.5 hours | 12+ hours |
| Inference Time | 100ms | 200ms | 2-5 seconds |
| Model Size | 13GB | 13GB | 13GB |
| VRAM Required | 20GB | 20GB | - |

## Tips for Better Results

1. **Data Quality** - Clean, domain-specific data is crucial
2. **Data Volume** - More data = better model (aim for 10k+ tokens)
3. **Learning Rate** - Start with 5e-5, adjust down if loss spikes
4. **Epochs** - 3-5 epochs typically optimal (monitor for overfitting)
5. **Validation** - Always test on unseen data

## Deployment

### Deploy to HuggingFace Hub
```python
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")
```

### Use Pre-Trained Model from Hub
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")
```

## Troubleshooting

**Error: CUDA out of memory**
```python
# Solution: Reduce batch size
batch_size = 2  # Instead of 4
# or use gradient accumulation
gradient_accumulation_steps = 2
```

**Error: Model too large**
```python
# Solution: Use LoRA quantization
from peft import get_peft_model, LoraConfig
# (Advanced - see PEFT docs)
```

**Model not improving**
```python
# Solutions:
1. Check data quality
2. Increase epochs
3. Lower learning rate (5e-6 to 5e-5)
4. Add more training data
```

## Model Comparison

| Model | Size | Speed | Quality | VRAM |
|-------|------|-------|---------|------|
| Mistral 7B | 13GB | Fast | Good | 16GB |
| Llama2 7B | 13GB | Medium | Good | 16GB |
| Llama2 13B | 26GB | Slower | Better | 32GB |

## Evaluation Metrics

After training, evaluate on held-out test set:
```python
from torch.utils.data import DataLoader

# Calculate perplexity, BLEU scores, etc.
# See evaluation.py for full implementation
```

## Next Steps

1. ✅ Prepare your data
2. ✅ Run training script
3. ✅ Evaluate on test set
4. ✅ Deploy API
5. ✅ Add to GitHub portfolio
6. ✅ Link to resume

## License
MIT License

## Author
Lineesha S | AI/ML Engineer

---

**For Job Applications:**
- Show training curves (loss decreasing)
- Compare with base model (show improvement)
- Share deployed API endpoint
- Include inference examples
