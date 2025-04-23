import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
    
    
    
if torch.cuda.is_available():
    device_name = "cuda"
    print(f"CUDA is available on device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device_name = "mps"
    print("MPS is available (Apple Silicon GPU) with this version of PyTorch")
else:
    print("No GPU available. Using CPU instead.")