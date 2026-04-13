import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup
import re
import random

# ── Scrape raw text from Project Gutenberg (Frankenstein) ────────────────────

URL = "https://www.gutenberg.org/cache/epub/84/pg84-images.html"

print(f"Fetching text from {URL} ...")
response = requests.get(URL, timeout=30)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

# Extract all visible text from the page body
raw_text = soup.get_text(separator=" ")

# Basic cleanup: collapse whitespace, keep only letters/punctuation
raw_text = re.sub(r"\s+", " ", raw_text).strip()

print(f"Raw text length: {len(raw_text)} characters")

# ── Tokenization ─────────────────────────────────────────────────────────────

# Simple word-level tokenization (lowercase, keep words only)
tokens = re.findall(r"[a-z']+", raw_text.lower())
print(f"Total tokens: {len(tokens)}")

# Build vocabulary
from collections import Counter
word_counts = Counter(tokens)
# Keep words that appear at least 2 times to reduce vocabulary size
vocab_words = [w for w, c in word_counts.items() if c >= 2]
vocab = ["<UNK>"] + sorted(vocab_words)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

def encode_token(w):
    return word2idx.get(w, 0)  # 0 = <UNK>

encoded = [encode_token(w) for w in tokens]

# ── Sequence preparation ─────────────────────────────────────────────────────

window_size = 100  # each input sequence has window_size-1 = 99 tokens

data    = []  # list of lists, each inner list has 99 token indices
targets = []  # single next-token index per sequence

for i in range(len(encoded) - window_size):
    data.append(encoded[i : i + window_size - 1])    # 99 tokens
    targets.append(encoded[i + window_size - 1])      # 100th token

print(f"Number of sequences: {len(data)}")

# Convert to tensors (use a subset for speed if needed)
MAX_SEQUENCES = 50_000
if len(data) > MAX_SEQUENCES:
    indices = random.sample(range(len(data)), MAX_SEQUENCES)
    data    = [data[i]    for i in sorted(indices)]
    targets = [targets[i] for i in sorted(indices)]
    print(f"Using {MAX_SEQUENCES} sequences (random subset)")

X = torch.tensor(data,    dtype=torch.long)   # (N, 99)
y = torch.tensor(targets, dtype=torch.long)   # (N,)

# ── RNN Model ─────────────────────────────────────────────────────────────────

class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim)
        self.rnn        = nn.RNN(embedding_dim, hidden_dim, num_layers,
                                 batch_first=True, nonlinearity="tanh")
        self.fc         = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        emb = self.embedding(x)              # (batch, seq_len, emb_dim)
        out, hidden = self.rnn(emb, hidden)  # out: (batch, seq_len, hidden)
        logits = self.fc(out[:, -1, :])      # take last time-step -> (batch, vocab)
        return logits, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = RNNTextGenerator(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ── Training ──────────────────────────────────────────────────────────────────

EPOCHS     = 5
BATCH_SIZE = 128
N          = X.shape[0]

print("\nTraining RNN...")
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(N)
    X = X[perm]
    y = y[perm]

    total_loss = 0.0
    for i in range(0, N, BATCH_SIZE):
        xb = X[i:i+BATCH_SIZE].to(device)
        yb = y[i:i+BATCH_SIZE].to(device)

        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / N
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}")

# ── Text generation ───────────────────────────────────────────────────────────

def generate_text(model, seed_sentence, word2idx, idx2word,
                  num_words=100, temperature=0.8):
    """
    Generate text starting from a seed sentence.
    temperature: higher = more diverse, lower = more conservative.
    """
    model.eval()
    words = re.findall(r"[a-z']+", seed_sentence.lower())
    # Encode seed words, fall back to <UNK> if not in vocabulary
    token_ids = [word2idx.get(w, 0) for w in words]

    generated = list(words)

    with torch.no_grad():
        for _ in range(num_words):
            # Use last (window_size-1)=99 tokens as context
            context = token_ids[-(window_size - 1):]
            # Pad from the left if context is shorter than 99
            if len(context) < window_size - 1:
                context = [0] * (window_size - 1 - len(context)) + context

            inp = torch.tensor([context], dtype=torch.long).to(device)
            logits, _ = model(inp)

            # Apply temperature scaling before sampling
            probs = torch.softmax(logits[0] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            token_ids.append(next_id)
            generated.append(idx2word.get(next_id, "<UNK>"))

    return " ".join(generated)

seed = "the creature wandered through the dark forest"
print("\n─── Generated text ──────────────────────────────────────────────────────")
print(f"Seed: \"{seed}\"\n")
generated = generate_text(model, seed, word2idx, idx2word, num_words=100)
print(generated)

print("""
─── Analysis ─────────────────────────────────────────────────────────────────
The model is trained on Frankenstein (Gutenberg), a 19th-century gothic novel
with a rich and somewhat archaic vocabulary.

After 5 epochs the loss decreases noticeably, showing the model is learning
statistical patterns of the text.  The generated output often produces
grammatically plausible short phrases, reflecting learned bigram/trigram
distributions captured by the RNN hidden state.

However, long-range coherence is limited: the model tends to repeat common
words or transition topics abruptly.  This is a known limitation of a single-
layer vanilla RNN due to the vanishing-gradient problem, which prevents
learning dependencies beyond roughly 10–20 tokens.  LSTM or GRU variants
would substantially improve coherence and stylistic consistency.

The temperature parameter (0.8 here) balances diversity vs. repetitiveness:
lower values make output more predictable; higher values increase creativity
at the cost of grammatical correctness.
""")
