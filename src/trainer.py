import torch
import os
from tqdm import tqdm

class SSLTrainer:
    def __init__(
        self,
        cnn,
        transformer,
        optimizer,
        dataloader,
        device="cpu",
        checkpoint_dir="checkpoints"
    ):
        self.cnn = cnn.to(device)
        self.transformer = transformer.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")

        torch.save({
            "cnn": self.cnn.state_dict(),
            "transformer": self.transformer.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

        print(f"Saved checkpoint → {path}")

    def train_epoch(self, loss_fn, mask_fn, apply_mask_fn):

        self.cnn.train()
        self.transformer.train()

        total_loss = 0

        for wave in tqdm(self.dataloader):

            wave = wave.to(self.device)

            # ---------------------------
            # Forward pass
            # ---------------------------
            features = self.cnn(wave)
            out = self.transformer(features)

            B, T, C = out.shape

            # ---------------------------
            # Mask
            # ---------------------------
            mask = mask_fn(B, T).to(self.device)
            masked = apply_mask_fn(out, mask)

            # ---------------------------
            # Loss
            # ---------------------------
            loss = loss_fn(masked, out)

            # ---------------------------
            # Backprop
            # ---------------------------
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def train(
        self,
        epochs,
        loss_fn,
        mask_fn,
        apply_mask_fn,
        save_every=1
    ):
        for epoch in range(1, epochs+1):

            loss = self.train_epoch(
                loss_fn,
                mask_fn,
                apply_mask_fn
            )

            print(f"Epoch {epoch} | Loss = {loss:.4f}")

            if epoch % save_every == 0:
                self.save_checkpoint(epoch)