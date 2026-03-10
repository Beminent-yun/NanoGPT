from pathlib import Path
import torch
from nanoGPT import NanoGPT
from train import load_data

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using {device}')
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8  # What is the maximum contect length for predictions?
TRAIN_EPOCHS = 3000
EVAL_ITERS = 300
lr = 1e-3
embed_dim = 384

datasets_dir = Path('./datasets/')
file_path = datasets_dir / "input.txt"
save_path = 'nano_gpt_model.pth'

if __name__ == "__main__":
    
    dataset = load_data('./datasets/input.txt')
    
    model = NanoGPT(
        vocab_size=dataset['vocab_size'],
        block_size=block_size,
        embed_dim=embed_dim,
        n_head=4
    ).to(device)
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    model.eval()
    
    decode = dataset['decode']
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(idx=context, max_new_tokens=500).squeeze().tolist()))