import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class DependencyDataset(Dataset):

  def __init__(self, inputs_filename, output_filename):
    self.inputs = np.load(inputs_filename)
    self.outputs_onehot = np.load(output_filename)

    self.outputs = self.outputs_onehot.argmax(axis=1).astype(np.int64)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    x = torch.tensor(self.inputs[k], dtype=torch.long)
    y = torch.tensor(self.outputs[k], dtype=torch.long)
    return x, y
    # return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    # TODO: complete for part 3

    self.embedding = Embedding(num_embeddings=word_types, embedding_dim=128)
    self.hidden    = Linear(6 * 128, 128)
    self.output    = Linear(128, 91)


  def forward(self, inputs):

    # TODO: complete for part 3
    # return torch.zeros(inputs.shape(0), 91)  # replace this line
    emb = self.embedding(inputs)
    B = emb.shape[0]
    x = emb.view(B, 6 * 128)
    h = relu(self.hidden(x))
    logits = self.output(h)
    return logits


def train(model, loader): 

  loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  # put model in training mode
  model.train()
 

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch
 
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits = model(inputs)
    loss = loss_function(logits, targets)

    tr_loss += loss.item()

    #print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    # correct += sum(torch.argmax(logits, dim=1) == torch.argmax(targets, dim=1))
    # total += len(inputs)

    preds = torch.argmax(logits, dim=1)
    correct += (preds == targets).sum().item()
    total += inputs.size(0)

    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary file {}.".format(WORD_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f)


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
    
    model.to(device)

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 
