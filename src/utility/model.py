import importlib.util
import torch
class ModelUtils:

    def load_model(config):
        model_file = config["model"]["file"]
        model_class_name = config["model"]["class"]
        model_params = config["model"]["params"]

        # Dynamically load the model module
        spec = importlib.util.spec_from_file_location("custom_model", model_file)
        custom_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_model_module)

        # Dynamically load the class
        model_class = getattr(custom_model_module, model_class_name)

        # Initialize the model with parameters
        model = model_class(**model_params)

        # Load checkpoint if specified
        checkpoint_path = config.get("checkpoint_path", None)
        if checkpoint_path and torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path))
        elif checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

        return model
    
    
