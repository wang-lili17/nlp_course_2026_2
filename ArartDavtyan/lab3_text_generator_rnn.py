# ============================================================
#  RNN Text Generator — Trained on Frankenstein (Project Gutenberg)
#  Demonstrates language modeling with a vanilla RNN,
#  early stopping, and text generation at multiple temperatures.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from collections import Counter


# ── 1. Web Scraping ───────────────────────────────────────────

url = "https://www.gutenberg.org/cache/epub/84/pg84-images.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

raw_text    = soup.get_text()
lines       = raw_text.splitlines()
clean_lines = [line.strip() for line in lines if line.strip()]
full_text   = " ".join(clean_lines)

# Lowercase and strip unwanted characters
full_text = full_text.lower()
for ch in ['"', '\u201c', '\u201d', '\u2018', '\u2019', '\n', '\r', '\t']:
    full_text = full_text.replace(ch, ' ')

tokens = full_text.split()
print(f"Tokens: {len(tokens)}")
print(' '.join(tokens[:20]))


# ── 2. Preprocessing ──────────────────────────────────────────

word_counts = Counter(tokens)
print(f"\nUnique words: {len(word_counts)}")

# Keep all words that appear at least once (min_freq = 1)
vocab_words = [w for w, c in word_counts.items() if c >= 1]
vocab       = {"<PAD>": 0, "<UNK>": 1}
for word in vocab_words:
    vocab[word] = len(vocab)

idx2word   = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

encoded = [vocab.get(word, 1) for word in tokens]
print(f"Encoded length: {len(encoded)}")

# Build sliding-window sequences of length `window_size`
WINDOW_SIZE = 100
data    = []
targets = []

for i in range(len(encoded) - WINDOW_SIZE):
    data.append(encoded[i : i + WINDOW_SIZE - 1])
    targets.append(encoded[i + WINDOW_SIZE - 1])

print(f"Sequences: {len(data)}")
print(f"Input length: {len(data[0])}")
print(f"Example target: '{idx2word[targets[0]]}'")


# ── 3. Dataset & DataLoader ───────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, data, targets):
        self.data    = torch.tensor(data,    dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


dataset      = TextDataset(data, targets)
train_size   = int(0.9 * len(dataset))
val_size     = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

print(f"\nTrain: {train_size} | Val: {val_size}")
x_batch, y_batch = next(iter(train_loader))
print(f"X shape: {x_batch.shape} | Y shape: {y_batch.shape}")


# ── 4. Model ──────────────────────────────────────────────────

class TextRNN(nn.Module):
    """Vanilla RNN-based language model for next-word prediction."""
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128,
                 num_layers=1, dropout=0.3):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded  = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        last_out  = rnn_out[:, -1, :]        # take only the last timestep
        out       = self.fc(self.dropout(last_out))
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

model        = TextRNN(vocab_size=vocab_size).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
print(model)


# ── 5. Training with Early Stopping ──────────────────────────

NUM_EPOCHS = 15
PATIENCE   = 3

optimizer        = optim.Adam(model.parameters(), lr=0.001)
criterion        = nn.CrossEntropyLoss()
best_val_loss    = float('inf')
patience_counter = 0
best_model_state = None

train_losses = []
val_losses   = []

for epoch in range(NUM_EPOCHS):
    # ── Training pass ──
    model.train()
    epoch_train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        predictions = model(x_batch)
        loss        = criterion(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        epoch_train_loss += loss.item()

    # ── Validation pass ──
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions      = model(x_batch)
            loss             = criterion(predictions, y_batch)
            epoch_val_loss  += loss.item()

    avg_train = epoch_train_loss / len(train_loader)
    avg_val   = epoch_val_loss   / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # ── Early stopping ──
    if avg_val < best_val_loss:
        best_val_loss    = avg_val
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

# Restore the best checkpoint
model.load_state_dict(best_model_state)
print(f"\nBest Val Loss: {best_val_loss:.4f}")

# ── Loss curve ──
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(train_losses)+1), train_losses,
         marker='o', label='Train Loss', color='steelblue')
plt.plot(range(1, len(val_losses)+1),   val_losses,
         marker='s', label='Val Loss',   color='coral')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_val_loss.png", dpi=150)
plt.show()
print("Loss curve saved to training_val_loss.png")


# ── 6. Text Generation ────────────────────────────────────────

def generate_text(model, seed_text, vocab, idx2word,
                  num_words=100, temperature=1.0):
    """Generate text autoregressively from a seed phrase."""
    model.eval()
    seed_tokens   = seed_text.lower().split()
    seed_encoded  = [vocab.get(word, 1) for word in seed_tokens]
    seq_len       = WINDOW_SIZE - 1

    # Left-pad with zeros if shorter than the required context window
    window = [0] * (seq_len - len(seed_encoded)) + seed_encoded
    window = window[-seq_len:]

    generated_words = seed_tokens.copy()

    with torch.no_grad():
        for _ in range(num_words):
            x        = torch.tensor([window], dtype=torch.long).to(device)
            output   = model(x) / temperature
            probs    = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated_words.append(idx2word.get(next_idx, '<UNK>'))
            window   = window[1:] + [next_idx]

    return " ".join(generated_words)


SEED = "the monster looked at me with"
print("=" * 60)
print(f"Seed: '{SEED}'")
print("=" * 60)

for temp in [0.5, 1.0, 1.5]:
    print(f"\nTemperature {temp}:")
    print(generate_text(model, SEED, vocab, idx2word,
                        num_words=100, temperature=temp))


# ── 7. Analysis ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

print(f"\nBest Train Loss:  {min(train_losses):.4f}")
print(f"Best Val Loss:    {best_val_loss:.4f}")
print(f"Train Perplexity: {np.exp(min(train_losses)):.1f}")
print(f"Val Perplexity:   {np.exp(best_val_loss):.1f}")

print("\n<UNK> statistics:")
for temp in [0.5, 1.0, 1.5]:
    text  = generate_text(model, SEED, vocab, idx2word,
                          num_words=100, temperature=temp)
    words = text.split()
    unk   = words.count('<UNK>')
    print(f"  Temperature {temp}: {unk} <UNK> ({unk / len(words) * 100:.1f}%)")

# ── Combined loss & perplexity plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, len(train_losses)+1), train_losses,
         marker='o', label='Train', color='steelblue')
ax1.plot(range(1, len(val_losses)+1),   val_losses,
         marker='s', label='Val',   color='coral')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)

train_perp = [np.exp(l) for l in train_losses]
val_perp   = [np.exp(l) for l in val_losses]
ax2.plot(range(1, len(train_perp)+1), train_perp,
         marker='o', label='Train', color='steelblue')
ax2.plot(range(1, len(val_perp)+1),   val_perp,
         marker='s', label='Val',   color='coral')
ax2.set_title('Perplexity')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis_loss_perplexity.png", dpi=150)
plt.show()
print("Analysis plots saved to analysis_loss_perplexity.png")
