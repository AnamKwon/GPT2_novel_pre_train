import torch
import torch.nn.functional as F
import sentencepiece as spm

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs >= top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tok, sent, num_token, temperature, top_p, top_k, device='cuda'):
    device = torch.device(device)

    toked = tok(sent)
    count = 0
    input_ids = torch.tensor(toked).unsqueeze(0)
    if len(toked) > 1024:
        return 0

    while count != num_token:
        input_ids = input_ids.to(device)
        model = model.to(device)
        predicts = model(input_ids)
        logits = predicts[0]
        logits = logits[:, -1, :] / temperature      
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p=top_p)
        log_probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1) 
        input_ids = torch.cat([input_ids[0],prev[0]]).unsqueeze(0)
        count += 1
    return input_ids[0]
