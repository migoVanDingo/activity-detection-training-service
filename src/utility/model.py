import importlib.util
import torch
class ModelUtils:

    def load_model(config):
        model_file = config["prepare_environment"]["model"]["file"]
        model_class_name = config["prepare_environment"]["model"]["class"]
        model_params = config["prepare_environment"]["model"]["params"]

        # Dynamically load the model module
        spec = importlib.util.spec_from_file_location("custom_model", model_file)
        custom_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_model_module)

        # Dynamically load the class
        model_class = getattr(custom_model_module, model_class_name)

        # Initialize the model with parameters
        print(f"Initializing model: {model_params}")
        model = model_class(**model_params)

        # Load checkpoint if specified
        checkpoint_path = config["prepare_environment"]["model_summary"]
        if checkpoint_path and torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path))
        elif checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

        return model
    

