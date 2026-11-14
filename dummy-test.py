import torch
import torchsparse
import os

print("--- Environment Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n--- Checkpoint Test ---")
ckpt_path = "checkpoints/waymo.pth"
if os.path.exists(ckpt_path):
    # Load the checkpoint directly into the GPU
    try:
        checkpoint = torch.load(ckpt_path, map_location="cuda:0")
        print("✅ Successfully loaded Waymo checkpoint into GPU VRAM!")
        
        # Check what's inside the checkpoint (usually 'state_dict' or 'model')
        keys = list(checkpoint.keys())
        print(f"Checkpoint keys: {keys}")
        
        # If it contains a state_dict, show a few layer names
        state_dict_key = 'state_dict' if 'state_dict' in keys else keys[0]
        if isinstance(checkpoint[state_dict_key], dict):
            layers = list(checkpoint[state_dict_key].keys())
            print(f"Total model layers: {len(layers)}")
            print(f"First 3 layers: {layers[:3]}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
else:
    print(f"❌ Checkpoint not found at {ckpt_path}. Did you download and unzip it?")

print("\n--- Torchsparse GPU Test ---")
try:
    # Create a dummy 3D point cloud coordinate tensor (N, 4) -> (Batch, X, Y, Z)
    coords = torch.randint(0, 10, (100, 4), dtype=torch.int32).cuda()
    # Create random features for those 100 points
    feats = torch.rand((100, 32), dtype=torch.float32).cuda()
    
    # Initialize a SparseTensor (the core object used by PCAccumulation)
    sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
    print("✅ Torchsparse successfully created and mapped a 3D sparse tensor on the GPU!")
except Exception as e:
    print(f"❌ Torchsparse failed: {e}")