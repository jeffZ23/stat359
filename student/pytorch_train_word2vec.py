import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.centers = torch.tensor(skipgram_df['center'].values, dtype = torch.long)
        self.contexts = torch.tensor(skipgram_df['context'].values, dtype = torch.long)

    def __len__(self):
        return self.centers.size(0)
    
    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context):
        v = self.in_embed(center)
        u = self.out_embed(context)
        return (v * u).sum(dim = -1)
    
    def neg_forward(self, center, neg_context):
        v = self.in_embed(center).unsqueeze(1)  
        u = self.out_embed(neg_context)         
        return (v * u).sum(dim=-1)              
    
    def neg_backward(self, center, neg_context):
        v = self.in_embed(center).unsqueeze(1)
        u = self.out_embed(neg_context)
        return (u * v).sum(dim = -1)
    
    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

skipgram_df = data['skipgram_df']
word2idx = data['word2idx']
idx2word = data['idx2word']
counter = data['counter']
vocab_size = len(word2idx)
   
# Precompute negative sampling distribution below
counts = torch.zeros(vocab_size, dtype = torch.float)
for i in range(vocab_size):
    w = idx2word[i]
    counts[i] = float(counter.get(w, 0))
neg_dist = counts.pow(0.75)
neg_dist = neg_dist / neg_dist.sum()


# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle = True, drop_last=True)


# Model, Loss, Optimizer
model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)

#def make_targets(center, context, vocab_size):
#pass
def sample_negatives(pos_context, num_neg, dist, max_resample_iters=10):
    bsz = pos_context.size(0)
    samples = torch.multinomial(dist, bsz * num_neg, replacement=True).view(bsz, num_neg)
    pos = pos_context.view(-1, 1)
    mask = samples.eq(pos)
    it = 0
    while mask.any() and it < max_resample_iters:
        num_bad = int(mask.sum().item())
        resampled = torch.multinomial(dist, num_bad, replacement=True)
        samples[mask] = resampled
        mask = samples.eq(pos)
        it += 1
    return samples

# Training loop
model.train()
pbar_outer = tqdm(range(EPOCHS), desc="Epochs")
for _ in pbar_outer:
    total_loss = 0.0
    num_batches = 0
    for center, context in tqdm(dataloader, leave=False, desc="Batches"):
        center = center.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)

        optimizer.zero_grad()

        pos_logits = model(center, context)
        pos_labels = torch.ones_like(pos_logits)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)

        neg_context = sample_negatives(context.cpu(), NEGATIVE_SAMPLES, neg_dist).to(device, non_blocking=True)
        neg_logits = model.neg_forward(center, neg_context)
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    pbar_outer.set_postfix({"loss": total_loss / max(1, num_batches)})



# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
