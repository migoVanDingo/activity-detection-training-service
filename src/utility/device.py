import torch


class DeviceUtils:
    def get_cuda_device(self, cuda_device_id=0):
        """ Sets NVIDIA device with id `cuda_device_id` to
        be used for training.

        Parameters
        ----------
        net: Custom Network Instance
            Instance of pytorch custom network we will be training
        cuda_device_id: int
            CUDA device ID
        """
        if not torch.cuda.is_available():
            raise Exception("ERROR: Cuda is not found in this environment")

        num_cuda_devices = torch.cuda.device_count()
        cuda_devices = list(range(0,
                                  num_cuda_devices))  # cuda index start from 0

        if not (cuda_device_id <= num_cuda_devices - 1):
            raise Exception(
                f"ERROR: Cuda device {cuda_device_id} is not found.\n"
                f"ERROR: Found {num_cuda_devices}("
                f"{cuda_devices}), Cuda devices")

        # If no errors load network onto cuda device
        print("INFO: Network sucessfully loaded into "
              f"CUDA device {cuda_device_id}")
        return f"cuda:{cuda_device_id}"