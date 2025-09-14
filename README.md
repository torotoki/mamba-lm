# mamba-lm
A minimal implementation of Mamba in PyTorch and Hugging Face Transformers.

See also my blog articles: [English on Medium](https://medium.com/@torotoki0329soft/getting-started-with-mamba-implementing-mamba-in-pytorch-33d56ccd8393), [Japanese on Qiita](https://qiita.com/torotoki/items/97aae3116e8178851697).

## Architecture Diagram
<img src="images/architecture_overview.png" alt="architecture" style="border: 2px solid #333; border-radius: 8px;" />

## Experiments
The loss curve from an experiment using artificial data:
![loss curve](images/loss_v1.png)

# References

* Albert Gu and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces." arXiv preprint arXiv:2312.00752 (2023).
* An official implementation (highly optimized): https://github.com/state-spaces/mamba
