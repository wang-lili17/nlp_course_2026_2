# ============================================================
#  Text Classifier — Sports vs Technology
#  A simple embedding-based neural network built with PyTorch
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# ── 1. Dataset (train + test) ────────────────────────────────

train_texts = [
    # Sports (100 samples) - Label 0
    "Lakers win championship after amazing comeback",
    "Football team scores three goals in final match",
    "Tennis star wins grand slam tournament",
    "Olympic athlete breaks world record",
    "Basketball game ends in dramatic overtime",
    "Soccer team advances to finals",
    "Runner wins marathon in record time",
    "Baseball team clinches playoff spot",
    "Swimmer wins gold medal at Olympics",
    "Hockey team defeats rival in shootout",
    "Gymnast performs perfect routine at competition",
    "Boxer wins title fight in final round",
    "Cycling champion wins Tour de France stage",
    "Golfer sinks incredible putt to win tournament",
    "Figure skater lands difficult jump perfectly",
    "Wrestling champion defends title successfully",
    "Cricket team wins test match series",
    "Rugby team scores last minute try",
    "Volleyball team wins championship match",
    "Badminton player wins international tournament",
    "Team celebrates victory after tough game",
    "Players train hard for upcoming season",
    "Coach announces new strategy for playoffs",
    "Stadium packed with fans for final game",
    "Athletes compete in regional championship",
    "Team captain leads squad to victory",
    "Young player shows great potential",
    "Fans celebrate team winning streak",
    "Manager signs new contract with club",
    "Training camp begins for new season",
    "Team wins away game against rivals",
    "Player receives award for performance",
    "Championship game draws huge crowd",
    "Team prepares for important match",
    "Coach praises players after victory",
    "Athletes set new records at event",
    "Team qualifies for next round",
    "Player scores hat trick in match",
    "Championship trophy awarded to winners",
    "Team celebrates historic achievement",
    "Players excited for tournament start",
    "Coach announces starting lineup today",
    "Team practices before big game",
    "Victory parade celebrates championship win",
    "Players sign autographs for fans",
    "Team mascot entertains crowd at game",
    "Championship banner raised at stadium",
    "Team dominates in playoff series",
    "Player injury sidelines star athlete",
    "Team rebuilds roster for season",
    "Coach implements new training program",
    "Players bond during team retreat",
    "Stadium renovations completed for season",
    "Team announces ticket prices for games",
    "Player traded to rival team",
    "Coach resigns after disappointing season",
    "Team practices penalty kicks for game",
    "Championship ring ceremony held today",
    "Team wins tournament in overtime",
    "Player breaks scoring record in game",
    "Coach gives motivational speech to team",
    "Team celebrates winning season finale",
    "Athletes train for upcoming competition",
    "Player signs endorsement deal today",
    "Team unveils new uniform design",
    "Championship parade draws massive crowd",
    "Coach analyzes game film with players",
    "Team defeats defending champions convincingly",
    "Player receives sportsmanship award today",
    "Team holds charity event for fans",
    "Athletes prepare for championship match",
    "Coach praises team effort after win",
    "Team advances in tournament bracket",
    "Player scores winning goal in match",
    "Championship celebrations continue all week",
    "Team announces preseason schedule today",
    "Players work hard during practice",
    "Coach happy with team performance",
    "Team wins decisive game at home",
    "Athletes excited for season opener",
    "Player makes incredible save in game",
    "Team rallies from behind for victory",
    "Coach confident before important match",
    "Championship run inspires young athletes",
    "Team prepares strategy for rivals",
    "Player demonstrates leadership on field",
    "Team celebrates milestone victory today",
    "Coach reviews tactics with players",
    "Athletes compete in regional finals",
    "Player achieves personal best in event",
    "Team practices formations for game",
    "Championship trophy displayed at stadium",
    "Coach motivates team before playoffs",
    "Team wins thrilling match in finale",
    "Player earns spot on national team",
    "Athletes train for international competition",
    "Team holds press conference today",

    # Technology (100 samples) - Label 1
    "New smartphone features advanced AI technology",
    "Tech company releases latest software update",
    "Scientists develop breakthrough quantum computer",
    "Artificial intelligence system improves healthcare",
    "New app helps users learn programming",
    "Electric vehicle company announces new model",
    "Researchers create faster internet connection",
    "Social media platform adds new features",
    "Cloud computing service expands globally",
    "Cybersecurity system protects against attacks",
    "Virtual reality headset launches next month",
    "Machine learning algorithm solves complex problem",
    "New programming language released by developers",
    "Tech startup raises millions in funding",
    "5G network expands to more cities",
    "Robotics company builds autonomous system",
    "Data center uses renewable energy",
    "Blockchain technology improves security",
    "New laptop features powerful processor",
    "Software update fixes major bugs",
    "Company develops innovative tech solution",
    "Algorithm improves search accuracy online",
    "Digital platform streamlines business operations",
    "Innovation drives tech industry forward",
    "Startup creates app for education",
    "Computer chip breakthrough announced today",
    "Technology advances medical research capabilities",
    "New software helps developers code faster",
    "Tech firm invests in AI research",
    "Digital transformation changes business landscape",
    "Scientists program robot for tasks",
    "Tech company expands into new markets",
    "Innovation lab opens in Silicon Valley",
    "New technology revolutionizes communication industry",
    "Software engineers develop better tools",
    "Tech conference showcases latest innovations",
    "Startup creates platform for collaboration",
    "Computer vision system recognizes objects accurately",
    "Technology enables remote work solutions",
    "Digital security measures protect data",
    "Algorithm optimizes supply chain operations",
    "Tech giant announces quarterly earnings today",
    "Innovation accelerates in artificial intelligence",
    "Software update enhances user experience",
    "Tech startup disrupts traditional industry",
    "Computer scientists breakthrough in research",
    "Digital platform connects developers worldwide",
    "Technology transforms healthcare delivery system",
    "New app simplifies complex tasks",
    "Innovation drives efficiency in business",
    "Tech company partners with university",
    "Software development tools improve productivity",
    "Algorithm processes data faster now",
    "Technology enables smart home systems",
    "Digital innovation changes customer experience",
    "Tech firm releases open source software",
    "Computer program automates repetitive work",
    "Innovation creates new tech opportunities",
    "Software engineers collaborate on project",
    "Technology advances renewable energy solutions",
    "Digital tools help students learn",
    "Tech startup solves infrastructure problem",
    "Algorithm improves recommendation system accuracy",
    "Innovation drives semiconductor industry growth",
    "Software platform integrates multiple services",
    "Technology enables precision medicine approach",
    "Digital assistant becomes more intelligent",
    "Tech company invests in quantum research",
    "Computer network expands bandwidth capacity",
    "Innovation transforms financial services industry",
    "Software developers build mobile applications",
    "Technology improves agricultural productivity significantly",
    "Digital marketplace connects buyers sellers",
    "Tech firm announces new partnership",
    "Algorithm detects patterns in data",
    "Innovation accelerates autonomous vehicle development",
    "Software update adds security features",
    "Technology enables virtual collaboration tools",
    "Digital platform supports remote learning",
    "Tech startup creates innovative solution",
    "Computer system processes information quickly",
    "Innovation drives biotechnology research forward",
    "Software engineers optimize application performance",
    "Technology transforms manufacturing processes today",
    "Digital tools enhance creative workflows",
    "Tech company expands research division",
    "Algorithm improves translation accuracy significantly",
    "Innovation creates sustainable technology solutions",
    "Software platform enables data analysis",
    "Technology advances space exploration capabilities",
    "Digital infrastructure supports cloud services",
    "Tech firm develops edge computing solution",
    "Computer scientists research neural networks",
    "Innovation drives telecommunications industry growth",
    "Software development becomes more accessible",
    "Technology enables personalized learning experiences",
    "Digital security protects online transactions",
]

train_labels = [0] * 100 + [1] * 100

test_texts = [
    # Sports (25 samples) - Label 0
    "Team wins game in final seconds",
    "Player scores amazing goal today",
    "Coach proud of team performance",
    "Athletes compete in championship finals",
    "Team defeats rivals in playoff",
    "Player breaks record in tournament",
    "Championship game ends in victory",
    "Team prepares for important match",
    "Athletes train for upcoming season",
    "Player receives award for excellence",
    "Team celebrates winning streak today",
    "Coach announces strategy for game",
    "Championship trophy presented to team",
    "Player scores in overtime win",
    "Team advances to next round",
    "Athletes perform well at competition",
    "Coach motivates players before match",
    "Team wins decisive playoff game",
    "Player demonstrates skill on field",
    "Championship victory celebrated by fans",
    "Team dominates in tournament play",
    "Athletes excited for season start",
    "Player achieves milestone in career",
    "Team practices before big match",
    "Championship parade honors winning team",

    # Technology (25 samples) - Label 1
    "Software company releases new product",
    "Algorithm improves system performance significantly",
    "Tech startup develops innovative platform",
    "Digital tools enhance productivity today",
    "Innovation drives technology sector forward",
    "Computer program solves difficult problem",
    "Technology transforms business operations completely",
    "Software engineers create better solutions",
    "Tech firm announces breakthrough research",
    "Digital platform connects users globally",
    "Algorithm processes information efficiently now",
    "Innovation accelerates in tech industry",
    "Software update improves functionality greatly",
    "Technology enables new capabilities today",
    "Tech company invests in development",
    "Digital system automates complex tasks",
    "Innovation creates technology opportunities now",
    "Software platform integrates services seamlessly",
    "Technology advances research capabilities significantly",
    "Tech startup solves industry challenge",
    "Algorithm optimizes operations effectively today",
    "Innovation drives digital transformation forward",
    "Software developers build applications efficiently",
    "Technology improves user experience greatly",
    "Digital innovation changes industry landscape",
]

test_labels = [0] * 25 + [1] * 25

print(f"Train samples: {len(train_texts)}")
print(f"Test samples:  {len(test_texts)}")
print(f"Sports train:  {train_labels.count(0)}")
print(f"Tech train:    {train_labels.count(1)}")


# ── 2. Vocabulary ─────────────────────────────────────────────

def tokenize(text):
    return text.lower().split()


all_words = []
for text in train_texts:
    all_words.extend(tokenize(text))

word_counts = Counter(all_words)
print(f"\nUnique words in train: {len(word_counts)}")

vocab = {"<PAD>": 0, "<UNK>": 1}
for word in word_counts:
    vocab[word] = len(vocab)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

for word in ["team", "game", "software", "algorithm"]:
    print(f"  '{word}' → index {vocab.get(word, 1)}")


# ── 3. Dataset & DataLoader ───────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=10):
        self.labels = labels[:len(texts)]
        self.max_len = max_len
        self.encoded = [self.encode(text) for text in texts]

    def encode(self, text):
        tokens = text.lower().split()
        indices = [vocab.get(word, 1) for word in tokens]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset  = TextDataset(test_texts,  test_labels,  vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

x_batch, y_batch = next(iter(train_loader))
print(f"\nX batch shape: {x_batch.shape}")
print(f"Y batch shape: {y_batch.shape}")
print(f"Train dataset size: {len(train_dataset)}")


# ── 4. Model ──────────────────────────────────────────────────

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64, num_classes=2):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden    = nn.Linear(embedding_dim, hidden_dim)
        self.relu      = nn.ReLU()
        self.output    = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled   = embedded.mean(dim=1)   # mean pooling over token dimension
        hidden   = self.relu(self.hidden(pooled))
        out      = self.output(hidden)
        return out


model = TextClassifier(vocab_size=vocab_size)
print(f"\n{model}")


# ── 5. Training ───────────────────────────────────────────────

NUM_EPOCHS = 20

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Save embeddings before training (epoch 0)
words_to_track = ["team", "game", "software"]
embeddings_epoch0 = {}
for word in words_to_track:
    idx = vocab[word]
    embeddings_epoch0[word] = model.embedding.weight[idx].detach().clone()

print(f"\nEmbeddings saved at epoch 0")
print(f"Tracked words: {words_to_track}\n")

train_losses = []
final_train_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    correct    = 0
    total      = 0

    for x_batch, y_batch in train_loader:
        predictions = model(x_batch)
        loss        = criterion(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss        += loss.item()
        predicted_labels   = predictions.argmax(dim=1)
        correct           += (predicted_labels == y_batch).sum().item()
        total             += y_batch.size(0)

    avg_loss             = epoch_loss / len(train_loader)
    accuracy             = correct / total * 100
    final_train_accuracy = accuracy
    train_losses.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.1f}%")

# Save embeddings after training (epoch 20)
embeddings_epoch_final = {}
for word in words_to_track:
    idx = vocab[word]
    embeddings_epoch_final[word] = model.embedding.weight[idx].detach().clone()

print(f"\nEmbeddings saved at epoch {NUM_EPOCHS}")


# ── 6. Evaluation & Analysis ──────────────────────────────────

# 6a. Test Accuracy
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        predictions      = model(x_batch)
        predicted_labels = predictions.argmax(dim=1)
        correct         += (predicted_labels == y_batch).sum().item()
        total           += y_batch.size(0)

test_accuracy = correct / total * 100
print(f"\nTest Accuracy:  {test_accuracy:.1f}%")
print(f"Train Accuracy: {final_train_accuracy:.1f}%")
print()

# 6b. Cosine Similarity between word embeddings
def cosine_similarity(v1, v2):
    v1    = v1.numpy()
    v2    = v2.numpy()
    dot   = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norms


pairs = [
    ("team",     "game"),      # Sports ↔ Sports  → should be close
    ("team",     "software"),  # Sports ↔ Tech    → should be distant
    ("game",     "software"),  # Sports ↔ Tech    → should be distant
]

print("─" * 55)
print(f"{'Word pair':<25} {'Epoch 0':>10} {f'Epoch {NUM_EPOCHS}':>10}")
print("─" * 55)
for w1, w2 in pairs:
    sim0   = cosine_similarity(embeddings_epoch0[w1],     embeddings_epoch0[w2])
    simF   = cosine_similarity(embeddings_epoch_final[w1], embeddings_epoch_final[w2])
    arrow  = "↑" if simF > sim0 else "↓"
    print(f"  {w1} ↔ {w2:<18} {sim0:>9.3f}  {simF:>9.3f}  {arrow}")
print("─" * 55)
print()

# 6c. Embedding shift (L2 norm of difference)
print("Embedding vector change (L2 norm):")
for word in words_to_track:
    diff = (embeddings_epoch_final[word] - embeddings_epoch0[word]).norm().item()
    print(f"  '{word}': {diff:.4f}")
print()

# 6d. Loss curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker="o", color="steelblue", linewidth=2)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()
print("Loss curve saved to training_loss.png")
