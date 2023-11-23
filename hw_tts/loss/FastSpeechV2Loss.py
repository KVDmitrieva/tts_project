import torch.nn as nn


class FastSpeechV2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, pitch_predicted, energy_predicted,
                mel_target, alignment, pitch, energy, **batch):
        mel_loss = self.mse_loss(mel, mel_target)
        duration_predictor_loss = self.l1_loss(duration_predicted, alignment.float())
        pitch_loss = self.mse_loss(pitch_predicted, pitch.float())
        energy_loss = self.mse_loss(energy_predicted, energy.float())
        return {
            "loss": mel_loss + duration_predictor_loss + pitch_loss + energy_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_predictor_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss
        }
