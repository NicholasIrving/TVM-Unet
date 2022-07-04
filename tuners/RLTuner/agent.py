import torch
import torch.nn as nn
from collections import deque

MEMORY_SIZE = 1000000
MEMORY_THRESHOLD = 10000
BATCH_SIZE = 1
GAMMA = 0.99
LR = 0.001
MOMENTUM = 0.9

class Agent(object):
    def __init__(self, input_size, hidden_size=64, num_steps=1, device='cpu'):
        super(Agent, self).__init__()
        self.memory = deque()
        self.device = device

        if self.device == 'cuda':
            self.network = network(input_size, hidden_size, num_steps, self.device).to(self.device)
        else:
            self.network = network(input_size, hidden_size, num_steps, self.device)
        # self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=LR, momentum=MOMENTUM)

    def learn(self, tune_number, state, action, reward, next_state, done):
        # if done:
        #     self.memory.append((state, action, reward, next_state, 0))
        # else:
        #     self.memory.append((state, action, reward, next_state, 1))
        # if len(self.memory) > MEMORY_SIZE:
        #     self.memory.popleft()
        # if len(self.memory) < MEMORY_THRESHOLD:
        #     return
        #
        # batch = random.sample(self.memory, BATCH_SIZE)
        # state = torch.FloatTensor([x[0] for x in batch])
        # action = torch.LongTensor([[x[1]] for x in batch])
        # reward = torch.FloatTensor([[x[2]] for x in batch])
        # next_state = torch.FloatTensor([x[3] for x in batch])
        # done = torch.FloatTensor([[x[4]] for x in batch])

        for i in range(tune_number):
            eval_q = self.network(torch.tensor([[state[i]]], device=self.device))
            next_q = self.network(torch.tensor([[next_state[i]]], device=self.device))
            # eval_q = self.network(torch.tensor([[state[i]]])).gather(1, torch.tensor(action))
            # next_q = self.network(torch.tensor([[next_state[i]]])).detach()
            target_q = reward + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done
            loss = self.loss_func(eval_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class network(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_steps=1, device='cpu'):
        super(network, self).__init__()

        self.DEVICE = device
        self.num_filter_option = 2
        self.filter_size_option = 3

        self.lstm1 = nn.LSTMCell(input_size, hidden_size).to(self.DEVICE)
        # May be could just use different decoder if these two numbers are the same, not sure
        self.decoder = nn.Linear(hidden_size, self.num_filter_option).to(self.DEVICE)
        # self.decoder2 = nn.Linear(hidden_size, self.filter_size_option)

        # num_steps = max_layer * 2 # two conv layer * 2 h-parameters (kernel size and number of kernels)
        self.num_steps = num_steps
        self.nhid = hidden_size
        self.hidden = self.init_hidden()

        self.init_weights(self.lstm1)
        self.init_weights(self.decoder)

    def forward(self, input):
        outputs = []
        h_t, c_t = self.hidden

        for i in range(self.num_steps):
            # input_data = self.embedding(step_data)
            h_t, c_t = self.lstm1(input.float(), (h_t, c_t))
            # Add drop out
            # h_t = self.drop(h_t)
            output = self.decoder(h_t)
            input = output
            outputs += [output]

        outputs = torch.stack(outputs).squeeze(1)

        return outputs

    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)

        return (h_t, c_t)

    def init_weights(self, m):
        if type(m) == nn.LSTMCell:
            m.weight_hh.data.normal_(0.0, 0.02)
            m.weight_ih.data.normal_(0.0, 0.02)
            m.bias_hh.data.fill_(0.01)
            m.bias_ih.data.fill_(0.01)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

