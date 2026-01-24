import json
import struct
import regex as re
import numpy as np

from load import Load
from safetensors.numpy import load_file
from safetensors import safe_open

# https://huggingface.co/%7Bmodel_size%7D/resolve/main/pytorch_model.bin%20gpt2-medium

def load_safetensors(path):
    with open(path, 'rb') as f:
        length_bytes = f.read(8)
        header_size = struct.unpack('<Q', length_bytes)[0]
        
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        
        weights_data = f.read()
        
        tensors = {}
        for name, info in header.items():
            print(f'name: {name} | info: {info}')
            if name == "__metadata__": continue
            
            start, end = info['data_offsets']
            raw_buffer = weights_data[start:end]
            
            count = len(raw_buffer) // 4
            tensors[name] = struct.unpack(f'<{count}f', raw_buffer)
            
        return tensors

def load_gpt2_weights(path):
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        print(f.keys())
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

class Tokenizer():
    def __init__(self, merges_bytes: bytes, vocab: dict) -> None:
        self.vocab = vocab
        self.merges = self._parse_merges(merges_bytes)
        self.byte_encoder = self.get_byte_to_unicode_map()
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _parse_merges(self, merges_bytes: bytes):
        lines = merges_bytes.decode('utf-8').split('\n')
        lines = [l for l in lines if l and not l.startswith('#')]
        merges = {}
        for idx, line in enumerate(lines):
            pair = tuple(line.split())
            if len(pair) == 2:
                merges[pair] = idx
        return merges

    def get_byte_to_unicode_map(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def bpe(self, token):
        word = list(token)
        
        while len(word) >= 2:
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
            min_rank = float('inf')
            min_pair = None
            
            for p in pairs:
                rank = self.merges.get(p, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = p
            
            if min_pair is None or min_rank == float('inf'):
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == min_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            
        return word

    def __call__(self, text: str):
        token_chunks = re.findall(self.pat, text)

        final_tokens = []
        for chunk in token_chunks:
            encoded_chunk = "".join(self.byte_encoder[b] for b in chunk.encode('utf-8'))
            final_tokens.extend(self.bpe(encoded_chunk))
            
        token_ids = [self.vocab.get(t, self.vocab.get('<|endoftext|>')) for t in final_tokens]
        return token_ids



import numpy as np

class LayerNorm:
    def __init__(self, weight, bias):
        self.weight = weight  # np.array
        self.bias   = bias

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var  = np.var (x, axis=-1, keepdims=True)
        return self.weight * (x - mean) / np.sqrt(var + 1e-5) + self.bias



class GPT2Block:
    def __init__(self, weights: dict, n_embd: int, n_head: int):
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Attention
        self.c_attn_w = weights['attn_c_attn_w']   # [n_embd, 3*n_embd]
        self.c_attn_b = weights['attn_c_attn_b']   # [3*n_embd]
        self.c_proj_w = weights['attn_c_proj_w']   # [n_embd, n_embd]
        self.c_proj_b = weights['attn_c_proj_b']
        
        self.c_fc_w   = weights['mlp_c_fc_w']      # [n_embd, 4*n_embd]
        self.c_fc_b   = weights['mlp_c_fc_b']
        self.c_proj_w = weights['mlp_c_proj_w']    # [4*n_embd, n_embd]
        self.c_proj_b = weights['mlp_c_proj_b']

        # Attention
        self.attn_c_proj_w = weights['attn_c_proj_w']   # (768, 768)
        self.attn_c_proj_b = weights['attn_c_proj_b']

        # MLP
        self.mlp_c_fc_w    = weights['mlp_c_fc_w']      # (768, 3072)
        self.mlp_c_fc_b    = weights['mlp_c_fc_b']
        self.mlp_c_proj_w  = weights['mlp_c_proj_w']    # (3072, 768)
        self.mlp_c_proj_b  = weights['mlp_c_proj_b']
        
        # LayerNorms
        self.ln_1 = LayerNorm(weights['ln_1_w'], weights['ln_1_b'])
        self.ln_2 = LayerNorm(weights['ln_2_w'], weights['ln_2_b'])

    def __call__(self, x):
        batch, seq_len, _ = x.shape

        x_norm = self.ln_1(x)
        qkv = np.matmul(x_norm, self.c_attn_w) + self.c_attn_b
        q, k, v = np.split(qkv, 3, axis=-1)
        
        q = q.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0,2,1,3)
        k = k.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0,2,1,3)
        v = v.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0,2,1,3)
        
        att = np.matmul(q, k.transpose(0,1,3,2)) / np.sqrt(self.head_dim)
        
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        att = np.where(causal_mask[None,None,:,:], att, -1e9)
        
        att = np.exp(att) / (np.sum(np.exp(att), axis=-1, keepdims=True) + 1e-10)
        
        y = np.matmul(att, v)
        y = y.transpose(0,2,1,3).reshape(batch, seq_len, -1)
        
        y = np.matmul(y, self.attn_c_proj_w) + self.attn_c_proj_b
        x = x + y

        x_norm = self.ln_2(x)
        h = np.matmul(x_norm, self.mlp_c_fc_w) + self.mlp_c_fc_b          # → 3072
        
        h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
        
        y = np.matmul(h, self.mlp_c_proj_w) + self.mlp_c_proj_b          # 3072 → 768
        
        x = x + y

        return x


class GPT2Minimal:
    def __init__(self, weights: dict):
        self.n_embd   = 768
        self.n_head   = 12
        self.n_layer  = 12
        self.n_ctx    = 1024

        self.wte = weights['wte.weight']
        self.wpe = weights['wpe.weight']

        self.ln_f = LayerNorm(weights['ln_f.weight'],
                               weights['ln_f.bias'])

        self.blocks = []

        for i in range(self.n_layer):
            block_weights = {
                'attn_c_attn_w': weights[f'h.{i}.attn.c_attn.weight'],
                'attn_c_attn_b': weights[f'h.{i}.attn.c_attn.bias'],
                'attn_c_proj_w': weights[f'h.{i}.attn.c_proj.weight'],
                'attn_c_proj_b': weights[f'h.{i}.attn.c_proj.bias'],

                'mlp_c_fc_w'   : weights[f'h.{i}.mlp.c_fc.weight'],
                'mlp_c_fc_b'   : weights[f'h.{i}.mlp.c_fc.bias'],
                'mlp_c_proj_w' : weights[f'h.{i}.mlp.c_proj.weight'],
                'mlp_c_proj_b' : weights[f'h.{i}.mlp.c_proj.bias'],

                'ln_1_w'       : weights[f'h.{i}.ln_1.weight'],
                'ln_1_b'       : weights[f'h.{i}.ln_1.bias'],
                'ln_2_w'       : weights[f'h.{i}.ln_2.weight'],
                'ln_2_b'       : weights[f'h.{i}.ln_2.bias'],
            }
            self.blocks.append(GPT2Block(block_weights, self.n_embd, self.n_head))

    def forward(self, idx):
        batch, seq_len = idx.shape

        tok_emb = self.wte[idx]         # [1, seq, n_embd]
        pos_emb = self.wpe[np.arange(seq_len)]   # [seq, n_embd]
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = np.dot(x, self.wte.T)  # [1, seq, vocab]

        return logits

    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0):
        ids = np.array([prompt_ids])   # [1, len]

        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -self.n_ctx:])
            logits = logits[:, -1, :] / temperature

            next_id = np.argmax(logits, axis=-1)
            ids = np.concatenate([ids, next_id[:, None]], axis=-1)

        return ids.flatten().tolist()


if __name__ == '__main__':
    model_size = 'gpt2'
    url = f'https://huggingface.co/openai-community/{model_size}/resolve/main/model.safetensors'
    url_merges = f'https://huggingface.co/openai-community/{model_size}/resolve/main/merges.txt'
    url_vocab = f'https://huggingface.co/openai-community/{model_size}/resolve/main/vocab.json'
    url_config = f'https://huggingface.co/openai-community/{model_size}/resolve/main/config.json'
    p = Load.fetch_by_url(url=url, filename='model.safetensors')
    tensors = load_file(p).keys()
    for t in tensors:
        print(t)
    config_path = Load.fetch_by_url(url=url_config, filename='config.json')
    merges_path = Load.fetch_by_url(url=url_merges, filename='merges.txt')
    vocab_path = Load.fetch_by_url(url=url_vocab, filename='vocab.json')

    merges_file = None
    vocab_file = None

    with open(merges_path, 'rb') as mf:
        merges_file = mf.read()  
    
    with open(vocab_path, 'r') as vf:
        vocab_file = json.load(vf)

    if merges_file and vocab_file:
        print(type(merges_file), type(vocab_file))
        T = Tokenizer(merges_bytes=merges_file, vocab=vocab_file)       
        T('HELLO')
        text = "Hello"
        ids = T(text)
        print(ids)

        weights = load_gpt2_weights(p)

        model = GPT2Minimal(weights)

        text = "Hello world"
        ids = T(text)
        print("Input ids:", ids)

        generated_ids = model.generate(ids, max_new_tokens=40)
        print("Generated ids:", generated_ids)

        itos = {v: k for k, v in T.vocab.items()}
        print("".join(itos.get(i, '?') for i in generated_ids))