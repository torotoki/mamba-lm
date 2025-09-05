import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import Dataset

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import ModelOutput

class MambaConfig(PretrainedConfig):
    model_type = "mamba-impl"
    def __init__(
        self,
        d_model,
        vocab_size=30522,
        num_hidden_layers=4,
        d_ssm_state_size=16,
        d_conv=4,
        expand=2,
        time_step_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        if time_step_rank is "auto":
            time_step_rank = math.ceil(d_model / 16)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_hidden_layers = 4
        self.d_ssm_state_size = d_ssm_state_size
        self.d_conv = d_conv
        self.expand = expand
        # time-step-related arguments:
        self.time_step_rank = time_step_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor

class MambaBlock(GradientCheckpointingLayer):
    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_inner = int(config.expand * config.d_model)

        # convolution sequence transformation layer
        # (not explained in the paper)
        self.seq_conv1d = nn.Conv1d(
            in_channels=config.conv_dim,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # S4D initialization in continuous-time, will be discreted.
        # self.log_A: [config.d_state, config.d_model]
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        A = A.expand(config.d_model, -1).contiguous()
        self.log_A = nn.Parameter(torch.log(A))

        # selective projections for the input and the hidden layer
        # TODO: they might be the same linear layer
        self.B_layer = nn.Linear(config.d_model, self.d_inner)
        self.C_layer = nn.Linear(config.d_model, self.d_inner)

        # time-step projection (discretization)
        self.Delta_layer = nn.Linear(config.d_model, self.d_model)
        
        # Linear-recurrence-based State Space Model



    def forward(self, input_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        
        # 1. Gated MLP's linear projection
        B_states = self.B_layer(input_states)
        C_states = self.C_layer(input_states)
        Delta_states = self.Delta_layer(input_states)
        
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation (ナニコレ)
        hidden_states = self.activation(
            self.conv1d(hidden_states)[..., :seq_len]
        )
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation

        # 3.a. Selection: [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]

        # 3.b. Discretization: B and C to [batch, seq_len, self.d_inner, ssm_state_size]
        # A: [batch, d_model, seq_len, ssm_state_size]
        A = torch.exp(self.log_A)
        discrete_A = torch.exp(A[None, :, None]
        # 3.c perform the recurrence y <- SSM(A, B, C)(x)
        # 4. Final linear projection
        pass

class MambaOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensort | None = None
    hidden_states: torch.FloatTensor | None = None

class Mamba(PreTrainedModel):
    config_class = MambaConfig

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()  # Init weight (HF recommended)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        
    def get_input_embeddings(self):
        return self.emb
    
    def set_input_embeddings(self, new_emb):
        self.emb = new_emb
    
    # 語彙サイズが変わった場合
    def resize_token_embeddings(self, new_num_tokens: int):
        old_emb = self.get_input_embeddings()
        new_emb = nn.Embedding(new_num_tokens, self.config.d_model)
        new_emb.weight.data[:min(old_emb.num_embeddings, new_num_tokens)] = \
            old_emb.weight.data[:min(old_emb.num_embeddings, new_num_tokens)]
        self.set_input_embeddings(new_emb)
        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()


    
from datasets import Dataset

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "This message is hello world",
    "Goodbye world until the sun rises",
    "Smashing the mics in the bar",
    "custom transformer model works",
    "trainer rocks",
    "pytorch fun"
]

def encode(ex):
    out = tok(ex["text"], truncation=True, padding=False, max_length=64)
    return out
ds = Dataset.from_dict({"text": texts}).map(encode)
collator = DataCollatorWithPadding(tokenizer=tok)

config = MambaConfig(vocab_size=tok.vocab_size, d_model=128, nhead=4, num_layers=2, num_labels=2)
model = Mamba(config)

args = TrainingArguments(
    output_dir="out-custom",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds.select(range(4)),
    eval_dataset=ds.select(range(4, len(ds))),
    tokenizer=tok,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("out-custom") 