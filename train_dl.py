import torch
from torch.utils.data import Dataset, DataLoader
from torchtext import data, vocab
import torch.nn as nn
import torch.nn.functional as F
import spacy
import os
import numpy as np
import pandas as pd
from torchtext.data import get_tokenizer
import time
import shutil
import torch.optim as optim
import matplotlib.pyplot as plt


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        """
        x_flat = x.reshape(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha


class BilinearSeqAttn(nn.Module):
    """
    A bilinear attention layer over a sequence X w.r.t y:
    """

    def __init__(self, x_size, y_size):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size, )

    def forward(self, x, y, x_mask):
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        x_mask = x_mask.type(torch.float32)
        masked_logits = x_mask * xWy + (1 - x_mask) * -1e30
        softmax_fn = F.log_softmax
        probs = softmax_fn(masked_logits, dim=-1)

        return probs


class question_answering_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, padding_idx=1):
        """

        :param vocab_size:
        :param embedding_dim:
        :param hidden_size:
        :param num_layers:
        :param padding_idx:
        """
        super(question_answering_model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=True, batch_first=True)

        self.ques_rnn = nn.LSTM(input_size=embedding_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=True, batch_first=True)

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * hidden_size
        question_hidden_size = 2 * hidden_size

        self.self_attn = LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

        self.end_attn = BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

    def forward(self, context_idxs, question_idxs, c_mask, q_mask):
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        context_emb = self.embedding(context_idxs)
        question_emb = self.embedding(question_idxs)

        # Encode document with RNN
        c_output, _ = self.rnn(context_emb)
        # c_output =  c_output.permute(0, 2, 1)

        # Encode question with RNN
        q_output, (_, _) = self.ques_rnn(question_emb)

        q_merge_weights = self.self_attn(q_output, q_mask)
        q_hidden = weighted_avg(q_output, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(c_output, q_hidden, c_mask)
        end_scores = self.end_attn(c_output, q_hidden, c_mask)
        return start_scores, end_scores

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def save_ckp(state, checkpoint_dir):
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)


def plot_graph(dev, train, name, ep):
    """Plot accuracy"""
    plt.figure()
    x = range(1, ep + 1)
    plt.plot(x, dev, color='b', label='Dev')
    plt.plot(x, train, color='r', label='Train')
    plt.title("Question Answering")
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('{0}.png'.format(name))


def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0
    model.initHidden()
    loss_df = pd.DataFrame(columns=["loss", "y1", "y2"])

    # iterate over the training data
    counter_list = []
    counter = 0
    with torch.set_grad_enabled(True):
        for (question_idxs, context_idxs), (y1, y2) in data_loader:
            # send the input/labels to the GPU
            context_idxs = context_idxs.to(device)
            question_idxs = question_idxs.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            c_mask = torch.ones_like(context_idxs) != context_idxs
            batch_size = context_idxs.size(0)
            q_mask = torch.ones_like(question_idxs) != question_idxs
            optimizer.zero_grad()

            # forward
            # zero the parameter gradients
            score_s, score_e = model(context_idxs, question_idxs, c_mask, q_mask)

            _, prediction1 = torch.max(score_s, 1)
            _, prediction2 = torch.max(score_e, 1)

            l_y1 = loss_function(score_s, y1)
            l_y2 = loss_function(score_e, y2)
            loss = l_y1 + l_y2
            # backward
            loss.backward()
            optimizer.step()
            counter += 1

            # statistics
            current_loss += loss.item() * batch_size
            # current_acc += torch.sum(prediction1 == y1) + torch.sum(prediction2 == y2)
            current_acc += torch.sum(prediction1 == y1)

        total_loss = current_loss / len(data_loader.dataset)
        # total_loss = current_loss / batch_size
        total_acc = current_acc.double() / len(data_loader.dataset)
        # total_acc = current_acc.double() / batch_size

        print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))

    return total_loss, total_acc


def validation(model, loss_function, optimizer, data_loader):
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for (question_idxs, context_idxs), (y1, y2) in data_loader:
        # send the input/labels to the GPU
        context_idxs = context_idxs.to(device)
        question_idxs = question_idxs.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)

        c_mask = torch.ones_like(context_idxs) != context_idxs
        batch_size = context_idxs.size(0)
        q_mask = torch.ones_like(question_idxs) != question_idxs

        with torch.set_grad_enabled(False):
            # forward
            score_s, score_e = model(context_idxs, question_idxs, c_mask, q_mask)
            _, prediction1 = torch.max(score_s, 1)
            _, prediction2 = torch.max(score_e, 1)
            loss = loss_function(score_s, y1) + loss_function(score_e, y2)

        # statistics
        current_loss += loss.item() * batch_size
        # current_acc += torch.sum(prediction1 == y1) + torch.sum(prediction2 == y2)
        current_acc += torch.sum(prediction1 == y1)

    total_loss = current_loss / len(data_loader.dataset)
    # total_loss = current_loss / batch_size
    total_acc = current_acc.double() / len(data_loader.dataset)
    # total_acc = current_acc.double() / batch_size

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))

    return total_loss, total_acc


if __name__ == '__main__':
    TEXT = data.Field(use_vocab=True, fix_length=100, batch_first=True, lower=True, tokenize="spacy")
    TARGET = data.Field(sequential=False, use_vocab=False, is_target=True, batch_first=True)

    FIELD = [('question', TEXT),
             ('paragraph', TEXT),
             ('start_token', TARGET),
             ('end_token', TARGET)
             ]

    train, val = data.TabularDataset.splits(path='.\\data\\', train='preprocess_dl_train.csv',
                                            validation='preprocess_dl_dev.csv', format='csv',
                                            fields=FIELD)

    TEXT.build_vocab(train, vectors=vocab.GloVe(name='6B', dim=300))
    TEXT.build_vocab(val, vectors=vocab.GloVe(name='6B', dim=300))

    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_sizes=(64, 64), shuffle=True,
                                                      sort_key=lambda x: len(x.paragraph))

    model = question_answering_model(vocab_size=10000, embedding_dim=300, hidden_size=128, num_layers=1)
    print(model)
    model.embedding.weight.data = TEXT.vocab.vectors
    model.embedding.weight.requires_grad = True

    start_time = time.time()
    # select gpu 0, if available# otherwise fallback to cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_function = nn.NLLLoss()
    # transfer the model to the GPU
    model = model.to(device)
    # We'll optimize all parameters
    para = list(model.parameters())

    optimizer = optim.Adadelta(model.parameters(), lr=0.5)

    EPOCHS = 50
    dev_acc, train_acc = [], []
    dev_loss_values, train_loss_values = [], []
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_loss, train_accuracy = train_model(model, loss_function, optimizer, train_iter)
        dev_loss, dev_accuracy = validation(model, loss_function, optimizer, val_iter)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        checkpoint_dir = ".\checkpoints"
        save_ckp(checkpoint, checkpoint_dir)

        train_acc.append(train_accuracy)
        train_loss_values.append(train_loss)

        dev_acc.append(dev_accuracy)
        dev_loss_values.append(dev_loss)

    endtime = time.time() - start_time
    print("Endtime %s seconds", endtime)
    plot_graph(dev_acc, train_acc, "accuracy", EPOCHS)
    plot_graph(dev_loss_values, train_loss_values, "loss", EPOCHS)
