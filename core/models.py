from torch import nn
import torch.nn.functional as F


MAX_LENGTH = 16


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

class FullyConnectedNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim=100):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class CifarNet(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear((input_dim // 4)**2 * hidden_channels, 384, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.flatten(x)
        x = self.mlp(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size=ENC_VOCAB_LEN, hid_size=512):
        super().__init__()

        self.hid_size = hid_size
        self.emb = nn.Embedding(input_size, hid_size)
        self.lstm = nn.LSTM(input_size=hid_size, hidden_size=hid_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        embedded = self.dropout(self.emb(input))
        output, hidden = self.lstm(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, output_size=DEC_VOCAB_LEN, hid_size=512):
        super().__init__()

        self.hid_size = hid_size
        self.emb = nn.Embedding(output_size, hid_size)
        self.lstm = nn.LSTM(input_size=hid_size, hidden_size=hid_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25) # Dropout helps with overfitting. It "drops out" random features from inputs (25% features in our case)

        self.classification_out = nn.Linear(hid_size * 2, output_size)

    def forward_step(self, input, hidden): # Predicts the next token for the input
        output = self.dropout(self.emb(input))
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.classification_out(output)
        return output, hidden

    def forward(self, encoder_outputs, encoder_hidden, targets=None, teacher_forcing_ratio=None):
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(en_token_to_id[BOS_TOKEN])
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        # During training, we will pass the length of the correct answer to model
        length = targets.shape[1] if targets is not None else MAX_LENGTH
        for i in range(length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # In 20% of cases we will help our model continue its prediction by applying teacher forcing
            # We simply pass the correct answers to reinforce prediction of the next steps

            # Keep in mind that we pass the correct answers only in decoder_input, so
            # the loss function will not take into account these values
            if teacher_forcing_ratio is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, i].unsqueeze(1) # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden

class EncDec(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, targets=None, teacher_forcing_ratio=None):
        encoder_outputs, encoder_hiddens = self.encoder(inputs)
        decoder_outputs, decoder_hiddens = self.decoder(encoder_outputs, encoder_hiddens, targets, teacher_forcing_ratio)
        return decoder_outputs
