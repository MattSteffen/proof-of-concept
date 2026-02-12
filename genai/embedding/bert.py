"""
Fine-tuning BERT for Embeddings - Complete Example
Run: pip install sentence-transformers torch && python finetune_embeddings.py
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import torch.nn.functional as F
import random

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
TEMPERATURE = 0.05

# =============================================================================
# SYNTHETIC DATA GENERATION (Replace with your real data)
# =============================================================================

def generate_synthetic_data(num_samples=500):
    """
    Generates synthetic query-positive pairs.
    Replace this with your actual data loading logic.
    """
    topics = [
        ("machine learning", ["neural networks for prediction",
                              "training AI models with data",
                              "supervised learning algorithms",
                              "deep learning model optimization"]),
        ("python programming", ["coding in python language",
                                "python script development",
                                "writing python applications",
                                "python software engineering"]),
        ("data science", ["analyzing datasets for insights",
                          "statistical data analysis methods",
                          "data-driven decision making",
                          "exploratory data analysis techniques"]),
        ("web development", ["building websites and apps",
                             "frontend and backend development",
                             "creating web applications",
                             "internet software development"]),
        ("database systems", ["SQL database management",
                              "storing and querying data",
                              "relational database design",
                              "database administration tasks"]),
    ]

    pairs = []
    for _ in range(num_samples):
        topic, paraphrases = random.choice(topics)
        query = topic
        positive = random.choice(paraphrases)
        pairs.append({"query": query, "positive": positive})

    return pairs


# =============================================================================
# DATASET
# =============================================================================

class EmbeddingDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return pair["query"], pair["positive"]


def collate_fn(batch, tokenizer, max_length):
    """Tokenizes batch of query-positive pairs."""
    queries, positives = zip(*batch)

    q_encoded = tokenizer(
        list(queries),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    p_encoded = tokenizer(
        list(positives),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return q_encoded, p_encoded


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pooling(outputs, attention_mask)

    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


# =============================================================================
# LOSS FUNCTION: Multiple Negatives Ranking Loss
# =============================================================================

def multiple_negatives_ranking_loss(
    query_embeddings, doc_embeddings, temperature=0.05
):
    """
    In-batch negative loss.
    Each query should match its corresponding document (same index).
    All other documents in batch serve as negatives.
    """
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

    # Compute similarity matrix: (batch_size, batch_size)
    similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T) / temperature

    # Labels: diagonal elements are positives (query i matches doc i)
    batch_size = query_embeddings.size(0)
    labels = torch.arange(batch_size, device=query_embeddings.device)

    # Cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


# =============================================================================
# TRAINING
# =============================================================================

def train(model, dataloader, optimizer, device, temperature):
    model.train()
    total_loss = 0

    for batch_idx, (q_encoded, p_encoded) in enumerate(dataloader):
        # Move to device
        q_input_ids = q_encoded["input_ids"].to(device)
        q_attention_mask = q_encoded["attention_mask"].to(device)
        p_input_ids = p_encoded["input_ids"].to(device)
        p_attention_mask = p_encoded["attention_mask"].to(device)

        # Forward pass
        q_embeddings = model(q_input_ids, q_attention_mask)
        p_embeddings = model(p_input_ids, p_attention_mask)

        # Compute loss
        loss = multiple_negatives_ranking_loss(q_embeddings, p_embeddings, temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, tokenizer, device):
    """Simple evaluation: check if similar queries cluster together."""
    model.eval()

    test_pairs = [
        ("machine learning", "neural networks for AI"),
        ("machine learning", "baking a chocolate cake"),
        ("python programming", "writing code in python"),
        ("python programming", "growing tomatoes in garden"),
        ("data science", "analyzing datasets for insights"),
        ("data science", "playing basketball outdoors"),
    ]

    texts = list(set([p[0] for p in test_pairs] + [p[1] for p in test_pairs]))

    # Get embeddings
    with torch.no_grad():
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        embeddings = model(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarities
    print("\n" + "=" * 60)
    print("SIMILARITY SCORES (higher = more similar)")
    print("=" * 60)

    for text_a, text_b in test_pairs:
        idx_a = texts.index(text_a)
        idx_b = texts.index(text_b)
        similarity = torch.dot(embeddings[idx_a], embeddings[idx_b]).item()
        print(f"{similarity:.4f} | '{text_a}' <-> '{text_b}'")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmbeddingModel(MODEL_NAME).to(device)

    # Generate data
    print("\nGenerating synthetic training data...")
    train_pairs = generate_synthetic_data(num_samples=500)
    print(f"Generated {len(train_pairs)} training pairs")

    # Create dataloader
    dataset = EmbeddingDataset(train_pairs, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, MAX_LENGTH),
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        avg_loss = train(model, dataloader, optimizer, device, TEMPERATURE)
        print(f"  Average Loss: {avg_loss:.4f}")

    # Evaluate
    evaluate(model, tokenizer, device)

    # Save model
    save_path = "./fine_tuned_embedding_model"
    print(f"\nSaving model to {save_path}")
    model.bert.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")


if __name__ == "__main__":
    main()