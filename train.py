import requests
from pathlib import Path
import torch
import torch.nn.functional as F
from bigram import Bigram
from nanoGPT import NanoGPT

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

def load_data(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers 
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[ch] for ch in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return {
        'corpus': chars,
        'vocab_size': vocab_size,
        'encode': encode,
        'decode': decode,
        'train': train_data,
        'val': val_data
    }

# dataloader
def get_batch(data):
    # generate a small batch of inputs x and targets y
    random_start = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_start])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_start])   # right shift
    return {'x': x, 'y': y}

def train_one_epoch(model, data, optimizer, loss_fn, device):
    model.train()
    # get data
    batch = get_batch(data) # get one small batch
    x, y = batch['x'].to(device), batch['y'].to(device)
    # forward
    logits = model(x)   # [B, T, vocab_size]
    # calculate loss
    # logits: [B, T, vocab_size], y: [B, T]
    # F.cross_entropy要求：logits:[N,C], target:[N]
    # logits: [B, T, vocab_size] -> [B*T, vocab_size], y: [B*T,]
    B, T, C = logits.shape
    loss = loss_fn(logits.view(B*T, C), y.view(B*T))
    # zero gradient
    optimizer.zero_grad()
    # backward
    loss.backward()
    # gradient descent
    optimizer.step()
    
    return loss.item()
    

def train(model, dataset:dict, optimizer, loss_fn, device):
    total_loss = 0.0
    cnt = 0
    for epoch in range(TRAIN_EPOCHS):
        total_loss += train_one_epoch(model, dataset['train'], optimizer, loss_fn, device)
        cnt += 1
        if epoch % EVAL_ITERS == 0:
            train_loss = evaluate(model, dataset['train'], loss_fn, device)
            val_loss = evaluate(model, dataset['val'], loss_fn, device)
            print(f"step {epoch:4d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
            
    
@torch.inference_mode()
def evaluate(model, data, loss_fn, device):
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        batch = get_batch(data)
        x, y = batch['x'].to(device), batch['y'].to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = loss_fn(logits.view(B*T, C), y.view(B*T))
        losses.append(loss.item())
    
    return sum(losses)/len(losses)


if __name__ == "__main__":
    
    if not file_path.exists():
        datasets_dir.mkdir(parents=True, exist_ok=True)
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading {url} ...")
        response = requests.get(url)
        if response.status_code == 200:
            file_path.write_text(response.text, encoding='utf-8')
        else:
            raise Exception(f"下载失败，状态码：{response.status_code}")
    
    dataset = load_data(file_path)
    # model = Bigram(
    #     vocab_size=dataset['vocab']
    # ).to(device)

    model = NanoGPT(
        vocab_size=dataset['vocab_size'],
        block_size=block_size,
        embed_dim=embed_dim,
        n_head=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    loss_fn = F.cross_entropy
    
    train(model, dataset, optimizer, loss_fn, device)
    torch.save(model.state_dict(), save_path)
    print(f"Saving model parameters at: {save_path}")
    
    encode, decode = dataset['encode'], dataset['decode']
    context = torch.zeros((1, 1), dtype=torch.long, device=device) # [[0]] -> BOS
    print(decode(model.generate(idx=context, max_new_tokens=500).squeeze().tolist()))
    