import torch
from torch import nn
from torch.nn import functional as F


class OCRModel(nn.Module):
    def __init__(self, num_chars):
        super(self.__class__, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(768, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(64, 32, bidirectional=True, batch_first=True, dropout=0.25)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        batch_size, channels_size, height, width = images.size()
        # print(bs, c, h, w)
        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.max_pool_1(x)
        # print(x.size())
        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.max_pool_2(x)
        # print(x.size())
        x = x.permute(0, 3, 1, 2)
        # print(x.size())
        x = x.view(batch_size, x.size(1), -1)
        # print(x.size())
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        # print("before lstm:", x.size())
        x, _ = self.lstm(x)
        # print("after lstm", x.size())
        x = self.output(x)
        # print(x.size())

        x = x.permute(1, 0, 2)
        # print(x.size())

        if targets is not None:
            log_softmax_values = F.log_softmax(x, dim=2)
            # print(f"{log_softmax_values.shape=}")
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=log_softmax_values.size(0),
                dtype=torch.int32,
            )
            # print(input_lengths)
            target_lengths = torch.full(
                size=(batch_size,), fill_value=targets.size(1), dtype=torch.int32
            )
            # print(target_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths
            )
            return x, loss
        return x, None
