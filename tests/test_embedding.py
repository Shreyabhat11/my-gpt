import torch

from model.embedding import GPTEmbedding


vocab_size = 65
embedding_dim = 128
context_length = 16

model = GPTEmbedding(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_length=context_length,
)

x = torch.randint(
    0,
    vocab_size,
    (4, context_length),
)

output = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)

print(model.token_embedding.weight.shape)
print(model.position_embedding.weight.shape)