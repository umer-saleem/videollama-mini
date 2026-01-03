import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from model import VideoLLaMAMini
from dataset import VideoCaptionDataset

device = "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = VideoCaptionDataset("data/captions.txt")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = VideoLLaMAMini(
    vocab_size=tokenizer.vocab_size
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for epoch in range(200):
    for frames, input_ids in loader:
        frames, input_ids = frames.to(device), input_ids.to(device)

        logits = model(input_ids[:, :-1], frames)
        loss = criterion(
            logits.reshape(-1, tokenizer.vocab_size),
            input_ids[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
