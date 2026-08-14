import torch


def test_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        
        # Test tensor creation on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        
        print("GPU computation successful!")
        print("Result shape:", z.shape)
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    test_cuda()