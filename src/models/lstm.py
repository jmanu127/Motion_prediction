# import torch.nn as nn
#
#
# class LSTM(nn.module):
#     def __init__(self, input_size=11, hidden_size=80, batch_first=True, device="cpu"):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size,
#                             hidden_size=hidden_size,
#                             batch_first=batch_first,
#                             device=device)
