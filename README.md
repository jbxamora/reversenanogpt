# Little Shakespeare Language Model
- **THIS MODEL IS BASED ON https://github.com/karpathy/nanoGPT/blob/master/model.py**

This repository contains a character-based language model trained on a dataset of Shakespeare's works. The model is built using the Transformer architecture, implemented in PyTorch.

## Overview

The model was trained on the "tinyshakespeare" dataset, which is a subset of Shakespeare's works. The dataset was split into 90% for training and 10% for validation.

The Transformer architecture is composed of a series of blocks that include multi-head self-attention and feedforward layers. The model also utilizes positional embeddings for each token, which allows the model to understand the order of the input characters.

Here are the key hyperparameters used for training the model:

- Batch size: 16
- Block size: 32
- Maximum iterations: 5000
- Learning rate: 1e-3
- Number of layers: 4
- Number of heads: 4
- Embedding dimension: 64
- Dropout rate: 0.0

The model was trained using the AdamW optimizer.

## Components

### Token and Position Embeddings
The model starts by converting the input characters into embeddings using a token embedding table. Additionally, position embeddings are added to the token embeddings to help the model understand the order of the characters. The combined embeddings are then passed through the Transformer blocks.

### Transformer Blocks
Each Transformer block consists of two main components:

1. Multi-head self-attention: This allows the model to weigh the importance of different characters in the input sequence when predicting the next character. The self-attention mechanism computes attention scores (weights) by comparing the query, key, and value vectors. These scores are then used to aggregate the value vectors, resulting in an output that emphasizes the most relevant characters.
  
2. Feedforward layers: Each Transformer block also contains a feedforward sub-layer that consists of two linear layers with a ReLU activation function in between. This sub-layer helps the model learn more complex relationships between the input characters.
  
The output of each Transformer block is then passed to the next block, allowing the model to learn increasingly abstract relationships between the characters.

### Final Layer Normalization and Linear Projection
After passing through all the Transformer blocks, the embeddings undergo a final layer normalization to ensure their values are on a similar scale. The normalized embeddings are then passed through a linear layer that projects them onto the output vocabulary size, resulting in logits for each possible character.

## Training
The model is trained using a batched approach, where it processes multiple independent sequences in parallel. At each iteration, a batch of input sequences (`xb`) and their corresponding target sequences (`yb`) are sampled from the training data. The model computes the logits for the input sequences, and the loss is calculated using cross-entropy between the logits and the target sequences.

The gradients of the loss with respect to the model's parameters are computed using backpropagation, and the optimizer updates the model's weights accordingly.

## Evaluation
During training, the model's performance is regularly evaluated on both the training and validation datasets. This helps to monitor the model's progress and detect any signs of overfitting. The evaluation process involves estimating the average loss over a fixed number of iterations for both the training and validation sets.

## Text Generation
The trained model can be used to generate text by providing a context as input and specifying the desired number of new tokens to generate. The model outputs logits for the next character, which are then converted to probabilities using the softmax function. A character is sampled from the probability distribution, and the process is repeated until the desired number of characters is generated.

## Usage

To use the model for text generation, provide a context as input, and specify the number of new tokens to generate. The model will return the generated text as output and into the output.txt for clear visual representation between both texts.

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=2000)[0].tolist())
```
  Replace `max_new_tokens` with the desired number of characters to generate.


## Lessons Learned

1. **Transformer architecture**: Implementing the Transformer architecture helped gain a deeper understanding of the inner workings of self-attention, multi-head attention, and positional embeddings.
2. **Character-level language modeling**: Working with a character-based dataset provided insights into the challenges and nuances of predicting the next character in a sequence, as opposed to word-level language modeling.
3. **Dataset splitting and evaluation**: Splitting the dataset into training and validation sets, and regularly evaluating the model's performance on both sets, helped monitor the model's generalization capabilities and prevent overfitting.
4. **Text generation**: Implementing the model's text generation functionality provided experience in working with the softmax function, sampling from a probability distribution, and concatenating the generated tokens to create the final output.
