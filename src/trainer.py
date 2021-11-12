import os

import torch
import wandb
from transformers import AdamW, get_linear_schedule_with_warmup

from src.data_loader import load_train_dataset
from src.loss_function import compute_kl_loss
from src.model import Bert2Bert


class Trainer:
    def __init__(self, cfg):
        self.model = Bert2Bert(cfg.model)
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.trainer.lr)
        self.loader = load_train_dataset(cfg.data.train_path, cfg.model.model_name,
                                         cfg.trainer.batch_size, cfg.trainer.num_workers)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, cfg.trainer.warm_up_steps,
                                                            15 * len(self.loader))
        self.epochs = cfg.trainer.epochs
        self.project_name = cfg.project_name
        self.save_path = self.get_save_path(cfg.data.save_path)

    def forward(self):
        total_len = len(self.loader)
        global_step = 0
        best_loss = 10000
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.loader):
                batch = {k: v.cuda() for k, v in batch.items()}
                pad_mask = batch['decoder_attention_masks'].clone().unsqueeze(-1).expand(-1, -1, 42000)
                del batch['decoder_attention_masks']
                self.optimizer.zero_grad()
                output = self.model.forward(batch)
                output2 = self.model.forward(batch)

                loss = 0.5 * (output.loss + output2.loss)
                kl_loss = compute_kl_loss(output.logits, output2.logits, pad_mask.le(0))
                loss = loss + 1 * kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()

                if i % 50 == 0:
                    print(f"epoch: {epoch}, steps: {i}/{total_len}, loss: {loss.item()}")

                if i % 100 == 0:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_path, f"step_{str(global_step).zfill(6)}_loss_{round(best_loss, 5)}.pth")
                    )
                global_step += 1
                wandb.log({
                    "Loss": loss.item()
                })
                if best_loss > loss.item():
                    best_loss = loss.item()
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_path, f"{self.project_name}_best.pth")
                    )

    def get_save_path(self, save_folder_path: str):
        save_path = os.path.join(save_folder_path, "train", self.project_name)
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(save_path)
        return save_path



