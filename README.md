# Arc GPT

I built Arc GPT step by step in PyTorch to better understand how GPT works.

In this project, I followed Andrej Karphathy's youtube video, "Let's build GPT: from scratch, in code, spelled out.": https://www.youtube.com/watch?v=kCc8FmEb1nY

I built a simple bigram model and gradually developed it into self-attention and a small Transformer from first principles.

The biggest thing I learned was how the individual pieces of GPT fit together, especially tokenization, attention, residual connections, and training.

RESULTS: 10.79M parameters | 6-layer, 6-head Transformer | Loss 4.36 → 1.89 train / 2.00 validation by step 500