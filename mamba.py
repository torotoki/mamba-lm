import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import ModelOutput

class MambaConfig(PretrainedConfig):
    model_type = "mamba-impl"
    def __init__(
        self,
        d_model=128,
        vocab_size=30522,
        num_hidden_layers=4,
        d_ssm_state_size=16,
        d_conv=4,
        time_step_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        if time_step_rank == "auto":
            time_step_rank = math.ceil(d_model / 16)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.d_ssm_state_size = d_ssm_state_size
        self.d_conv = d_conv
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

        # convolution sequence transformation layer
        self.seq_conv1d = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.d_conv,
            groups=config.d_model,
            padding=config.d_conv - 1,
        )

        # selective projections for the input and the hidden layer
        self.to_params = nn.Linear(config.d_model, 3 * config.d_model)

        # The eigenvalues of A in continuous-time
        lam = torch.arange(1, config.d_model + 1, dtype=torch.float32)
        self.lam = nn.Parameter(lam)

        # Output projection
        self.norm = nn.RMSNorm(config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.LongTensor | None):
        # x: (B, T, D ( = config.d_model))
        # attention_mask: (B, T)
        B, T, D = x.shape
        x = self.norm(x)

        if attention_mask is not None:
            # Add dimention to attention_mask for broadcasting
            x = x * attention_mask.unsqueeze(2)
        
        # 1. Depth-wise convolution
        #   Swap the dimensions of x since
        #   self.seq_conv1d expects the dimension (B, D, T)
        hidden_states = self.seq_conv1d(x.transpose(1, 2)).transpose(1, 2)

        # 2. Gated MLP's linear projection
        dlt, Bt, Ct = self.to_params(hidden_states).chunk(3, dim=-1)

        # Stabilization using softplus
        dlt = F.softplus(dlt)         # \Delta_t > 0
        lam = -F.softplus(self.lam)   # \lambd a< 0  (D,)

        # 3. Autoregressive State-Space Models (SSM)
        ht = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            # NOTE: dlt and x can be transposed for faster computation
            dt = dlt[:, t, :]                        # (B,D)
            At_diag = torch.exp(dt * lam)            # (B,D) ← lam (D,) is broadcasted
            Bt_bar = torch.where(
                lam.abs() > 1e-4,
                ((At_diag - 1.0) / lam) * Bt[:, t, :],
                dt * Bt[:, t, :]
            )                                        # (B,D)
            ht = At_diag * ht + Bt_bar * x[:, t, :]  # (B,D)
            y = Ct[:, t, :] * ht
            ys.append(y)
        y = torch.stack(ys, dim=1)                   # (B,T,D)

        # 4. Final linear projection
        return self.out(y)

@dataclass
class MambaOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None

class Mamba(PreTrainedModel):
    config_class = MambaConfig

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()  # Init weight (HF recommended)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        x = self.emb(input_ids)
        for mamba_block in self.layers:
            x = mamba_block(x, attention_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        labels = input_ids
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),  # (B * T, vocab_size)
            shift_labels.view(-1)  # (B * T,)
        )

        print(loss)
        return MambaOutput(
            loss=loss,
            logits=logits,
        )
        
        
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
    save_strategy="no",  #TODO: epoch
    logging_steps=1,
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