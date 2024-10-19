import torch
# import wandb
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from torch.utils.data import DataLoader
from TTS.tts.layers.xtts.trainer.dvae_dataset import DVAEDataset
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig

from dataclasses import dataclass, field
from typing import Optional
import os
import datetime
from transformers import HfArgumentParser

@dataclass
class DVAETrainerArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    output_path: str = field(
        metadata={"help": "Path to pretrained + checkpoint model"}
    )
    train_csv_path: str = field(
        metadata={"help": "Path to train metadata file"},
    )
    eval_csv_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to eval metadata file"},
    )
    language: Optional[str] = field(
        default="en",
        metadata={"help": "The language you want to train (language in your dataset)"},
    )
    lr: Optional[float] = field(
        default=5e-6,
        metadata={"help": "Learning rate"},
    )
    num_epochs: Optional[int] = field(
        default=5,
    )
    batch_size: Optional[int] = field(
        default=512,
    )



def train(output_path, train_csv_path, eval_csv_path="", language="en", lr=5e-6, num_epochs=5, batch_size=512):
    dvae_pretrained = os.path.join(output_path, 'XTTS_v2.0_original_model_files/dvae.pth')
    mel_norm_file = os.path.join(output_path, 'XTTS_v2.0_original_model_files/mel_stats.pth')

    now = datetime.datetime.now()
    now_without_ms = now.replace(microsecond=0)
    # CHECKPOINTS_OUT_PATH = os.path.join(output_path, f"DVAE_checkpoint_{now_without_ms}/")
    # os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="large",
        path=os.path.dirname(train_csv_path),
        meta_file_train=os.path.basename(train_csv_path),
        meta_file_val=os.path.basename(eval_csv_path),
        language=language,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]
    GRAD_CLIP_NORM = 0.5
    LEARNING_RATE = lr

    dvae = DiscreteVAE(
                channels=80,
                normalization=None,
                positional_dims=1,
                num_tokens=1024,
                codebook_dim=512,
                hidden_dim=512,
                num_resnet_blocks=3,
                kernel_size=3,
                num_layers=2,
                use_transposed_convs=False,
            )

    dvae.load_state_dict(torch.load(dvae_pretrained), strict=False)
    dvae.cuda()
    opt = Adam(dvae.parameters(), lr = LEARNING_RATE)
    torch_mel_spectrogram_dvae = TorchMelSpectrogram(
                mel_norm_file=mel_norm_file, sampling_rate=22050
            ).cuda()

    train_samples, eval_samples = load_tts_samples(
            DATASETS_CONFIG_LIST,
            eval_split=True,
            eval_split_max_size=256,
            eval_split_size=0.01,
        )

    eval_dataset = DVAEDataset(eval_samples, 22050, True, max_wav_len=15*22050)
    train_dataset = DVAEDataset(train_samples, 22050, False, max_wav_len=15*22050)

    eval_data_loader = DataLoader(
                        eval_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=eval_dataset.collate_fn,
                        num_workers=0,
                        pin_memory=False,
                    )

    train_data_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=False,
                    )

    torch.set_grad_enabled(True)
    dvae.train()

    # wandb.init(project = 'train_dvae')
    # wandb.watch(dvae)

    def to_cuda(x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.contiguous()
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
        return x

    @torch.no_grad()
    def format_batch(batch):
        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = to_cuda(v)
        elif isinstance(batch, list):
            batch = [to_cuda(v) for v in batch]

        try:
            batch['mel'] = torch_mel_spectrogram_dvae(batch['wav'])
            # if the mel spectogram is not divisible by 4 then input.shape != output.shape 
            # for dvae
            remainder = batch['mel'].shape[-1] % 4
            if remainder:
                batch['mel'] = batch['mel'][:, :, :-remainder]
        except NotImplementedError:
            pass
        return batch

    best_loss = 1e6

    for i in range(num_epochs):
        dvae.train()
        for cur_step, batch in enumerate(train_data_loader):
            opt.zero_grad()
            batch = format_batch(batch)
            recon_loss, commitment_loss, out = dvae(batch['mel'])
            recon_loss = recon_loss.mean()
            total_loss = recon_loss + commitment_loss
            # print(f"commitment_loss shape: {commitment_loss.shape}")
            # print(f"recon_loss shape: {recon_loss.shape}")
            # print(f"total_loss shape: {total_loss.shape}")
            total_loss.backward()
            clip_grad_norm_(dvae.parameters(), GRAD_CLIP_NORM)
            opt.step()

            log = {'epoch': i,
                'cur_step': cur_step,
                'loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'commit_loss': commitment_loss.item()}
            print(f"epoch: {i}", print(f"step: {cur_step}"), f'loss - {total_loss.item()}', f'recon_loss - {recon_loss.item()}', f'commit_loss - {commitment_loss.item()}')
            # wandb.log(log)
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            dvae.eval()
            eval_loss = 0
            for cur_step, batch in enumerate(eval_data_loader):
                batch = format_batch(batch)
                recon_loss, commitment_loss, out = dvae(batch['mel'])
                recon_loss = recon_loss.mean()
                eval_loss += (recon_loss + commitment_loss).item()
            eval_loss = eval_loss/len(eval_data_loader)
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(dvae.state_dict(), dvae_pretrained)
            print(f"#######################################\nepoch: {i}\tEVAL loss: {eval_loss}\n#######################################")

    print(f'Checkpoint saved at {dvae_pretrained}')


if __name__ == "__main__":
    parser = HfArgumentParser(DVAETrainerArgs)

    args = parser.parse_args_into_dataclasses()[0]

    trainer_out_path = train(
        language=args.language,
        train_csv_path=args.train_csv_path,
        eval_csv_path=args.eval_csv_path,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )