import torch
import torch.nn.functional as F

def generate_text_from_video(
    model, frames, tokenizer,
    max_new_tokens=50, top_k=5, temperature=1.0, device="cpu"
):
    model.eval()
    tokens = torch.tensor([[tokenizer.bos_token_id]], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens, frames)
            logits = logits[:, -1] / temperature

            topk_logits, topk_idx = torch.topk(logits, top_k)
            probs = F.softmax(topk_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token = topk_idx.gather(-1, next_idx)

            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokens
