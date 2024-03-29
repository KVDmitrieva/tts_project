import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

import waveglow

from hw_tts.trainer.base_trainer import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metrics, optimizer, config, device, dataloaders,
                 waveglow_model, lr_scheduler=None, len_epoch=None, skip_oom=True):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50
        self.waveglow_model = waveglow_model
        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss",
            "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss",
            *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["alignment", "mel_target", "src_pos", "mel_pos", "text_encoded", "pitch", "energy"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_spectrogram(batch["mel_target"], "target")
                self._log_spectrogram(batch["mel"].detach(), "prediction")
                self._log_waveglow_audio(batch["mel_target"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        outputs = self.model(**batch)

        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["mel"] = outputs

        loss_out = self.criterion(**batch)
        batch.update(loss_out)

        if is_train:
            with torch.autograd.set_detect_anomaly(True):
                batch["loss"].backward(retain_graph=True)
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for key in loss_out.keys():
            metrics.update(key, batch[key].item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            self._log_spectrogram(batch["mel_target"], "target")
            self._log_spectrogram(batch["mel"], "prediction")
            self._log_waveglow_audio(batch["mel_target"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(self, text_encoded, src_pos, mel_target, mel_len, examples_to_log=3, *args, **kwargs):
        if self.writer is None:
            return
        is_training = self.model.training
        self.model.eval()
        for i in range(examples_to_log):
            txt, pos, mel_src, length = text_encoded[i].detach(), src_pos[i].detach(), mel_target[i].detach(), mel_len[i]

            mel_pred = self.model.inference(txt.unsqueeze(0), pos.unsqueeze(0)).detach()

            mel_pred = mel_pred[:length, :]
            mel_src = mel_src[:length, :]

            wav = waveglow.inference.get_wav(mel_pred.contiguous().transpose(1, 2), self.waveglow_model)
            pred = PIL.Image.open(plot_spectrogram_to_buf(mel_pred.T.cpu()))
            target = PIL.Image.open(plot_spectrogram_to_buf(mel_src.T.cpu()))

            self.writer.add_image("mel prediction example", ToTensor()(pred))
            self.writer.add_image("mel target example", ToTensor()(target))
            self.writer.add_audio("audio example", wav.detach().cpu().short(), self.config["preprocessing"]["sr"])

        if is_training:
            self.model.train()

    def _log_spectrogram(self, spectrogram_batch, name="spectrogram"):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.T))
        self.writer.add_image(name, ToTensor()(image))

    def _log_audio(self, audio_batch, name="audio"):
        audio = random.choice(audio_batch.cpu())
        self.writer.add_audio(name, audio, self.config["preprocessing"]["sr"])

    def _log_waveglow_audio(self, spectrogram_batch, name="waveglow audio"):
        mel = random.choice(spectrogram_batch)
        audio = waveglow.inference.get_wav(mel.contiguous().unsqueeze(0).transpose(1, 2), self.waveglow_model)
        self.writer.add_audio(name, audio.cpu().short(), self.config["preprocessing"]["sr"])

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
