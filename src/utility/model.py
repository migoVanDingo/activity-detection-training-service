import importlib.util
import torch


class ModelUtils:

    def load_model(config):
        model_file = config["prepare_environment"]["model"]["file"]
        model_class_name = config["prepare_environment"]["model"]["class"]
        model_params = config["prepare_environment"]["model"]["params"]

        # Dynamically load the model module
        spec = importlib.util.spec_from_file_location(
            "custom_model", model_file)
        custom_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_model_module)

        # Dynamically load the class
        model_class = getattr(custom_model_module, model_class_name)

        # Initialize the model with parameters
        print(f"Initializing model: {model_params}")
        model = model_class(**model_params)

        # Load checkpoint if specified
        checkpoint_path = config['prepare_environment']['checkpoint']
        if checkpoint_path and torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        elif checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)

        # Extract the model's state dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # In case it's a direct state_dict

        return model
