import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
from pathlib import Path

class LLMFineTuner:
    """Fine-tune open-source LLM on custom dataset"""
    
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", output_dir="./fine_tuned_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model and tokenizer loaded!")
    
    def prepare_dataset(self, data_file, train_size=0.9):
        """Prepare training dataset"""
        # Create train/eval split
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=data_file,
            block_size=512
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        return dataset, data_collator
    
    def train(self, data_file, epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune the model"""
        print("Preparing dataset...")
        dataset, data_collator = self.prepare_dataset(data_file)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            weight_decay=0.01,
            adam_epsilon=1e-8,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        print(f"Fine-tuning complete! Model saved to {self.output_dir}")
    
    def save_model(self):
        """Save fine-tuned model"""
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")
    
    def generate_text(self, prompt, max_length=200):
        """Generate text using fine-tuned model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class InferenceAPI:
    """API for inference using fine-tuned model"""
    
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def predict(self, text, max_length=200):
        """Generate prediction"""
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    finetuner = LLMFineTuner(
        model_name="mistralai/Mistral-7B-v0.1",
        output_dir="./my_finetuned_model"
    )
    
    finetuner.load_model_and_tokenizer()
    
    # Train on your data
    # finetuner.train("data.txt", epochs=3, batch_size=4)
    # finetuner.save_model()
    
    # Or use pre-trained
    print("Fine-tuning script ready. Update with your data path and run train()")
