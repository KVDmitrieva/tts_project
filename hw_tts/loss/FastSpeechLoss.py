import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, alignment, **batch):
        print("MEL", mel.shape, mel)
        print("MEL TARGET", mel_target.shape, mel_target)
        print("DUR PRED", duration_predicted.shape, duration_predicted)
        print("AL", alignment.shape, alignment)
        mel_loss = self.mse_loss(mel, mel_target)
        duration_predictor_loss = self.l1_loss(duration_predicted, alignment.float())
        return mel_loss, duration_predictor_loss
