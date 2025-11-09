from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

# Helper function to get the correct model path hash for shared models
def get_model_key(params):
    return (params.get('model_path'), params.get('tokenizer_path'))

class ModelScheduler:
    """Manages the lifecycle of models on the GPU."""
    def __init__(self):
        self.device = device
        # Stores currently loaded model/tokenizer instances: {key: (model_instance, tokenizer_instance)}
        self.loaded_models = {}
        self.active_model_key = None
        print(f"ModelScheduler initialized. Target device: {device}")

    def load_model(self, agent_key, model_path, tokenizer_path):
        """Loads the required model/tokenizer to GPU if not already loaded."""

        if self.active_model_key == agent_key:
            return self.loaded_models[agent_key]
        
        # 1. Unload any currently active model
        self.unload_active_model()
        
        # 2. Load the requested model
        print(f"Loading model: {agent_key}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto"
            ).to(self.device)
            
            self.loaded_models[agent_key] = (model, tokenizer)
            self.active_model_key = agent_key
            print(f"Successfully loaded {agent_key[0]} to GPU.")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model {agent_key[0]}: {e}")
            raise

    def unload_active_model(self):
        """Unloads the currently active model from GPU."""
        if not self.active_model_key:
            return
        
        key = self.active_model_key
        model, tokenizer = self.loaded_models.pop(key)
        
        # Explicitly clear objects and cache
        del model
        del tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect() # Garbage collection
        
        print(f"Successfully UNLOADED model key: {key[0]} from GPU.")
        self.active_model_key = None
        
    def get_active_model_and_tokenizer(self):
        """Returns the currently active model and tokenizer."""
        if not self.active_model_key:
            return None, None
        return self.loaded_models[self.active_model_key]
    

GLOBAL_SCHEDULER = ModelScheduler()