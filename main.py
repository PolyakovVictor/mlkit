import json
import struct
import regex as re

from load import Load
from safetensors.numpy import load_file

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
        
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
            
            print('hz', self.merges.get(pairs[1], float('inf')), self.merges.get(pairs[0], float('inf')) )
            print('hz2 :', min(pairs, key=lambda p: self.merges.get(p, float('inf'))))
            print('pairs: ', pairs)
            bigram = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            
            if bigram not in self.merges:
                break
            
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == bigram:
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



if __name__ == '__main__':
    model_size = 'gpt2'
    url = f'https://huggingface.co/openai-community/{model_size}/resolve/main/model.safetensors'
    url_merges = f'https://huggingface.co/openai-community/{model_size}/resolve/main/merges.txt'
    url_vocab = f'https://huggingface.co/openai-community/{model_size}/resolve/main/vocab.json'
    url_config = f'https://huggingface.co/openai-community/{model_size}/resolve/main/config.json'
    p = Load.fetch_by_url(url=url, filename='model.safetensors')
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
    # res = load_safetensors(path=p)
