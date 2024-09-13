import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import sentencepiece as spm
import numpy as np
import matplotlib.pyplot as plt

# model class + model helpers
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:x.shape[0]], requires_grad=False)
        return x

class Multi_Head_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, d_k, d_v, use_mask=False):
        super(Multi_Head_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k, self.d_v = d_k, d_v
        self.use_mask = use_mask

        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
    
    def scaled_dot_product_attention(self, q, k, v, use_mask=False):
        d_k = q.shape[-1]# q, k, v are shape: num_heads, num_tokens, d_qkv
        scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if use_mask:
            mask = torch.triu(torch.ones((scaled.shape), dtype=torch.float), diagonal=1).to(q.device)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            scaled = scaled + mask
        attention = F.softmax(scaled, dim=-1)
        out = torch.matmul(attention, v)
        return out, attention
            
    def forward(self, q_val, k_val, v_val):
        q = self.linear_q(q_val).view(-1, self.num_heads,  self.d_k).transpose(0, 1) # I think I have ruled out the entire multi head attention class as being wrong
        k = self.linear_k(k_val).view(-1, self.num_heads, self.d_k).transpose(0, 1)
        v = self.linear_v(v_val).view(-1, self.num_heads, self.d_v).transpose(0, 1)
        
        head_outs, head_attentions = self.scaled_dot_product_attention(q, k, v, use_mask=self.use_mask)
        attention_output = head_outs.transpose(0, 1).contiguous().view(-1, self.embed_dim)
        attention_output = self.linear_out(attention_output)
        return attention_output, head_attentions # can change this thing later

class Encoder_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, d_k, d_v):
        super(Encoder_Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k, self.d_v = d_k, d_v

        self.attention = Multi_Head_Attention(embed_dim=self.embed_dim, num_heads=self.num_heads, d_k=d_k, d_v=d_v, use_mask=False)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, encoder_input):
        attention_output, _ = self.attention(q_val=encoder_input, k_val=encoder_input, v_val=encoder_input)
        dropped_attention_output = self.dropout1(attention_output)
        add_and_norm_output1 = encoder_input + self.norm1(dropped_attention_output) # try encoder input in layernorm

        ff1_output = F.relu(self.ff1(add_and_norm_output1))
        ff2_output = self.ff2(ff1_output)
        dropped_ff2_output = self.dropout2(ff2_output)
        add_and_norm_output2 = add_and_norm_output1 + self.norm2(dropped_ff2_output)

        return add_and_norm_output2
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, ff_dim, dropout_rate, d_k, d_v):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim # dont really need all of these self.s because they are just params for other things
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.d_k, self.d_v = d_k, d_v
        
        self.blocks = nn.ModuleList([Encoder_Block(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout_rate=dropout_rate, d_k=self.d_k, d_v=self.d_v) for _ in range(num_blocks)])

    def forward(self, encoder_input):
        blocks_output = encoder_input
        for block in self.blocks:
            blocks_output = block(blocks_output)
        return blocks_output
    
class Decoder_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, d_k, d_v):
        super(Decoder_Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k, self.d_v = d_k, d_v

        self.masked_attention = Multi_Head_Attention(embed_dim=self.embed_dim, num_heads=self.num_heads, d_k=d_k, d_v=d_v, use_mask=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attention = Multi_Head_Attention(embed_dim=self.embed_dim, num_heads=self.num_heads, d_k=d_k, d_v=d_v, use_mask=False)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, encoder_output, decoder_input): # decoder_input is block output
        masked_attention_output, _ = self.masked_attention(q_val=decoder_input, k_val=decoder_input, v_val=decoder_input)
        dropped_masked_attention_output = self.dropout1(masked_attention_output)
        add_and_norm_output1 = decoder_input + self.norm1(dropped_masked_attention_output)

        attention_output, _ = self.cross_attention(q_val=add_and_norm_output1, k_val=encoder_output, v_val=encoder_output) # SHOULD BE TGT for V_VAL
        dropped_attention_output = self.dropout2(attention_output) # ^ USED TO BE decoder_input, SO IT WAS SKIPPING ALL BEFORE THIS LINE
        add_and_norm_output2 = add_and_norm_output1 + self.norm2(dropped_attention_output)

        ff1_output = F.relu(self.ff1(add_and_norm_output2))
        ff2_output = self.ff2(ff1_output)
        dropped_ff2_output = self.dropout3(ff2_output)
        add_and_norm_output3 = add_and_norm_output2 + self.norm3(dropped_ff2_output)

        return add_and_norm_output3
    
class Decoder(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, ff_dim, dropout_rate, d_k, d_v):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim # dont really need all of these self.s because they are just params for other things
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.d_k, self.d_v = d_k, d_v
        
        self.blocks = nn.ModuleList([Decoder_Block(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout_rate=dropout_rate, d_k=self.d_k, d_v=self.d_v) for _ in range(num_blocks)])

    def forward(self, encoder_output, decoder_input):
        blocks_output = decoder_input
        for block in self.blocks:
            blocks_output = block(encoder_output, blocks_output)
        return blocks_output
    
class Model(nn.Module):
    def __init__(self, embed_dim=512, num_blocks=6, num_heads=8, ff_dim=2048, dropout_rate=0.1, vocab_size=0):
        super(Model, self).__init__()
        self.embed_dim = embed_dim # dont really need all of these self.s because they are just params for other things
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.d_k, self.d_v = embed_dim // num_heads, embed_dim // num_heads
        self.embedding_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim=self.embed_dim, max_len=5000)

        self.encoder_dropout = nn.Dropout(p=dropout_rate)
        self.decoder_dropout = nn.Dropout(p=dropout_rate)

        self.encoder = Encoder(embed_dim=self.embed_dim, num_blocks=self.num_blocks, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout_rate=self.dropout_rate, d_k=self.d_k, d_v=self.d_v)
        self.decoder = Decoder(embed_dim=self.embed_dim, num_blocks=self.num_blocks, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout_rate=self.dropout_rate, d_k=self.d_k, d_v=self.d_v)

        self.Linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, en_tokens, de_tokens): # src, tgt

        en_embed = self.embedding_table(en_tokens)
        de_embed = self.embedding_table(de_tokens)

        en_embed = self.encoder_dropout(en_embed)
        de_embed = self.decoder_dropout(de_embed)

        encoder_input = self.positional_encoding(en_embed)
        decoder_input = self.positional_encoding(de_embed)

        encoder_output = self.encoder(encoder_input) # memory
        decoder_output = self.decoder(encoder_output, decoder_input) # memory, tgt

        logits = self.Linear(decoder_output)

        return logits
    
# training helpers:

def get_preprocessed_example(i, train_df, sp, device):
    en_input, de_input = train_df.iloc[i]['en'], train_df.iloc[i]['de']

    en_input = '[START]' + en_input + '[END]'
    de_input = '[START]' + de_input + '[END]'

    # src = sp.Encode(en_input)[1:]
    # tgt = sp.Encode(de_input)[1:]

    src = sp(en_input, max_length = 1024, truncation = True)['input_ids']
    tgt = sp(de_input, max_length = 1024, truncation = True)['input_ids']

    return torch.tensor(src).to(device), torch.tensor(tgt).to(device)

# testing helpers

def get_attention_matrix(src, tgt, loaded_model):
    src_embd = loaded_model.embedding_table(src)
    tgt_embd = loaded_model.embedding_table(tgt)
    src_embd = loaded_model.positional_encoding(src_embd)
    tgt_embd = loaded_model.positional_encoding(tgt_embd)
    
    encoder_attentions = []
    masked_attentions = []
    cross_attentions = []
    
    block_in_out = src_embd
    for block in loaded_model.encoder.blocks:
        _, head_attentions = block.attention(q_val=block_in_out, k_val=block_in_out, v_val=block_in_out)
        block_in_out = block(block_in_out)
        encoder_attentions.append(head_attentions.cpu().detach().numpy())
        
    memory = loaded_model.encoder(encoder_input=src_embd)

    block_in_out = tgt_embd
    for block in loaded_model.decoder.blocks:
        masked_attention_output, head_attentions_self = block.masked_attention(q_val=block_in_out, k_val=block_in_out, v_val=block_in_out)
        add_and_norm_output1 = block_in_out + block.norm1(masked_attention_output)
        _, head_attentions_cross = block.cross_attention(q_val=add_and_norm_output1, k_val=memory, v_val=memory)
        block_in_out = block(memory, block_in_out)
        
        masked_attentions.append(head_attentions_self.cpu().detach().numpy())
        cross_attentions.append(head_attentions_cross.cpu().detach().numpy())


     # Aggregate attention matrices (e.g., average over heads and layers)
    encoder_attention_matrix = np.mean(np.array(encoder_attentions), axis=(0, 1))  # average over layers and heads
    masked_attention_matrix = np.mean(np.array(masked_attentions), axis=(0, 1))# average over heads within each layer then 
    cross_attention_matrix = np.mean(np.array(cross_attentions), axis=(0, 1))# over layers within the model
    # cross_attention_matrix = np.mean(np.array(cross_attentions), axis=0)

    encoder_attention_matrix = np.clip((encoder_attention_matrix - encoder_attention_matrix.min()) / (encoder_attention_matrix.max() - encoder_attention_matrix.min()), 0, 1)
    masked_attention_matrix = np.clip((masked_attention_matrix - masked_attention_matrix.min()) / (masked_attention_matrix.max() - masked_attention_matrix.min()), 0, 1)
    cross_attention_matrix = np.clip((cross_attention_matrix - cross_attention_matrix.min()) / (cross_attention_matrix.max() - cross_attention_matrix.min()), 0, 1)

    return encoder_attention_matrix, masked_attention_matrix, cross_attention_matrix

def visualize_attention_matrix(encoder_attention_matrix, masked_attention_matrix, cross_attention_matrix, src, tgt, sp, attention_type):
    # for i in range(8):

    fig, ax = plt.subplots()

    if attention_type == 'encoder':
        im = ax.imshow(encoder_attention_matrix, cmap='viridis')
        ax.set_xticks(np.arange(len(src)))
        ax.set_yticks(np.arange(len(src)))
        ax.set_xticklabels([sp.decode([token.item()]) for token in src], rotation=45)
        ax.set_yticklabels([sp.decode([token.item()]) for token in src])
    if attention_type == 'masked':
        im = ax.imshow(masked_attention_matrix, cmap='viridis')
        ax.set_xticks(np.arange(len(tgt)))
        ax.set_yticks(np.arange(len(tgt)))
        ax.set_xticklabels([sp.decode([token.item()]) for token in tgt], rotation=45)
        ax.set_yticklabels([sp.decode([token.item()]) for token in tgt])
    if attention_type == 'cross':
        im = ax.imshow(cross_attention_matrix, cmap='viridis')
        # im = ax.imshow(cross_attention_matrix[i], cmap='viridis')
        ax.set_xticks(np.arange(len(src)))
        ax.set_yticks(np.arange(len(tgt)))
        ax.set_xticklabels([sp.decode([token.item()]) for token in src], rotation=45)
        ax.set_yticklabels([sp.decode([token.item()]) for token in tgt])
                            
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_title('attention scores')
    plt.show()