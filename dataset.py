import torch
from torch.utils.data import Dataset
import cv2
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class VideoCaptionDataset(Dataset):
    def __init__(self, captions_file, num_frames=8, frame_size=(64, 64), max_len=50):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.max_len = max_len
        self.samples = []

        with open(captions_file, "r") as f:
            video, caption = None, []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "|" in line:
                    if video:
                        self.samples.append((video, " ".join(caption)))
                    video, first = line.split("|", 1)
                    caption = [first]
                else:
                    caption.append(line)
            if video:
                self.samples.append((video, " ".join(caption)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, caption = self.samples[idx]
        cap = cv2.VideoCapture(f"data/{video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        idxs = torch.linspace(0, total - 1, self.num_frames).long().tolist()
        frames = []

        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        cap.release()
        frames = torch.stack(frames)

        tokens = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return frames, tokens.input_ids.squeeze(0)
