import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer, ExLlamaV2Lora
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from transformers import AutoTokenizer
import random

# 在模型初始化前添加
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision("medium")  # 嘗試 "high" 而非 "medium"

def generate(
        self,
        ids: torch.Tensor,
        num_tokens: int,
        seed: int | None = None,
        loras: ExLlamaV2Lora | list[ExLlamaV2Lora] | None = None,
    ):

    self.abort_event = None

    # Default stop token
    stop_token = self.tokenizer.eos_token_id

    # Accept LoRA or list of LoRAs
    if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]

    # Apply seed
    if seed is not None: random.seed(seed)

    # Truncate prompt if generation would cause cache overflow
    overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
    if overflow > 0: ids = ids[:, overflow:]
    else: overflow = 0

    mask = None

    first_token = max(-overflow, 0)

    # Process prompt and begin gen
    self._gen_begin_base(ids, mask, loras)

    # Generate tokens
    for i in range(num_tokens):
        logits = self.model.forward(self.sequence_ids[:, -1:],
                                    self.cache,
                                    loras = loras).float().cpu()

        token = torch.argmax(logits, dim=-1)


        if stop_token is not None:
            if token[0, 0].item() == stop_token:
                token[0, 0] = self.tokenizer.pad_token_id

        self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

    # Decode
    decode_ids = self.sequence_ids[:, first_token:]
    return decode_ids
ExLlamaV2BaseGenerator.generate = generate

# === 路徑與參數 ===
#model_dir = "/home/a313551050/final/Llama-3.2-3B-Instruct-exllama-default-head8b"
model_dir = "/home/a313551050/final/exllama_model_b4.0_hb6_hsol26"
#model_dir = "/home/a313551050/final/Llama-3.2-3B-Instruct-exllama-default-head8b"
#model_dir = "/home/a313551050/final/Llama-3.2-3B-Instruct"
prompt = "How to learn a new language?"
max_new_tokens = 256
device = "cuda:1"

# === 初始化模型 ===
print("Loading model...")
config = ExLlamaV2Config(model_dir)
config.arch_compat_overrides()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache_Q4(model, max_seq_len=2048)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2BaseGenerator(model=model, cache=cache, tokenizer=tokenizer)

# === Throughput 測試 ===
settings = ExLlamaV2Sampler.Settings()
settings.temperature = 1.0
settings.top_k = 1

tputs, time_record = [], []

input_ids = tokenizer.encode(prompt) 

#print("Warming up...")
for _ in range(3):
    output = generator.generate(input_ids, num_tokens=max_new_tokens)
    #print(f"Warm-up output: {output}")
    #print("Running benchmark...")

for _ in tqdm(range(10), desc="Test Inference"):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()    
    generated = generator.generate(input_ids, num_tokens=max_new_tokens)    
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    tput = max_new_tokens / (elapsed_ms / 1000)
    tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
    time_record.append(elapsed_ms / 1000)
    tputs.append(tput)
sorted_tputs = np.sort(tputs)[2:-2]
org_tput = np.mean(sorted_tputs)
response = tokenizer.decode(generated[0][input_ids.shape[1]:])
print(f'Prompt: {prompt}\nResponse: {response}\n')
print(f"Time Record (sec): {time_record}")
print(f"Throughput Record (tokens/s): {tputs}")
print(f"Average Throughput: {org_tput:.2f} tokens/s")


def evaluate_ppl_hf_encode_fixed(model, max_seq_len=2048):
    from transformers import AutoTokenizer
    import torch.nn.functional as F

    hf_tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-3B-Instruct-exllama-default-head8b")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    input_ids = hf_tokenizer.encode(text, add_special_tokens=False)
    print(f"[HF DEBUG] Encoded token count: {len(input_ids)}")

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    device = model.modules[0].device()
    input_ids = input_ids.to(device)

    nll_total = 0.0
    count = 0
    nsamples = (len(input_ids) - 1) // max_seq_len

    for i in tqdm(range(nsamples), desc="Evaluating PPL"):
        segment = input_ids[i * max_seq_len : (i + 1) * max_seq_len + 1]
        input_seq = segment[:-1].unsqueeze(0)
        target_seq = segment[1:]

        with torch.no_grad():
            logits = model.forward(input_seq)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs[0, torch.arange(target_seq.size(0)), target_seq]
            nll_total += -token_log_probs.sum().item()
            count += target_seq.size(0)

    ppl = torch.exp(torch.tensor(nll_total / count))
    return round(ppl.item(), 4)

def evaluate_ppl(model: ExLlamaV2, tokenizer: ExLlamaV2Tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer.encode("\n\n".join(test_dataset["text"]))
    model.seqlen = 2048
    test_enc = test_enc.to(model.modules[0].device())
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model.forward(batch)

        shift_logits = lm_logits[:, :-1, :].contiguous().float().to(model.modules[0].device())
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

# === 執行 PPL 測試 ===
print("\nEvaluating PPL on Wikitext-2...")
ppl = evaluate_ppl_hf_encode_fixed(model)
#ppl = evaluate_ppl(model, tokenizer)
print(f"Perplexity (PPL): {ppl}")

# Save results to CSV
import csv
rounded_tput = round(org_tput, 1)
ppl = round(ppl, 2)
with open("result.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "value"])
    writer.writerow([0, ppl])
    writer.writerow([1, rounded_tput])