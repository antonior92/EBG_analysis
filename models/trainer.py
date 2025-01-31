import torch
from torch import nn
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast


class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            weights: torch.Tensor,
            n_epochs: int,
            save_path, device='cuda'
    ):

        self.device = device

        self.model = model.to(device)
        self.loss_cls = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)
        self.accuracy = Accuracy(task='binary').to(self.device)
        self.auroc = AUROC(task="binary")
        self.optimizer = optimizer

        self.epochs = n_epochs
        self.mixed_precision = True

        self.save_path = save_path
        self.patience = 2000
        self.log_freq = 50

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        scaler = GradScaler(enabled=self.mixed_precision)

        best_model = None
        best_loss = 10000000
        best_acc = 0.0
        patience = self.patience
        print("Training Started...")
        for epoch in range(self.epochs):
            # print(f"Epoch {epoch}/{self.epochs}.")
            steps = 0
            loss_epoch = []
            y_true = []
            y_pred = []
            self.model.train()
            progress_bar = tqdm(train_data_loader)
            for x, y in progress_bar:
                self.optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)

                with autocast(enabled=self.mixed_precision):
                    preds = self.model(x)
                    preds = preds.squeeze()
                    loss_classification = self.loss_cls(preds, y)

                loss_epoch.append(loss_classification.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds) >= 0.5)

                scaler.scale(loss_classification).backward()
                scaler.step(self.optimizer)
                scaler.update()

                steps += 1

            with torch.no_grad():
                # train_avg_acc = self.accuracy(torch.stack(y_pred), torch.stack(y_true))
                train_loss = np.mean(loss_epoch)
                train_auroc = self.auroc(torch.stack(y_pred).detach().cpu().float(), torch.stack(y_true).detach().cpu())
                train_balanced_acc = balanced_accuracy_score(torch.stack(y_true).detach().cpu(), torch.stack(y_pred).
                                                             detach().cpu())

            val_loss, val_balanaced_acc, val_auroc = self.evaluate(self.model, val_data_loader)
            # if val_loss < best_loss:
            if val_balanaced_acc > best_acc:
                # best_loss = val_loss
                best_acc = val_balanaced_acc
                patience = self.patience
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'acc': val_balanaced_acc,
                    'auroc': val_auroc
                }
            else:
                patience -= 1

            # if patience == 0:
            #     break

            if epoch % self.log_freq == 0 or epoch == self.epochs - 1:
                print(f'---------------------- Epoch: {epoch} ----------------------')
                print(f'Training AUROC: {train_auroc} | Training Acc.: {train_balanced_acc} | Training Loss: {train_loss}')
                print(f'Validation AUROC: {val_auroc} | Validation Acc: {val_balanaced_acc} | Validation Loss: {val_loss}')

            wandb.log({
                "train_acc": train_balanced_acc,
                "train_loss": train_loss,
                "train_auroc": train_auroc,
                "val_acc": val_balanaced_acc,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "epoch": epoch
            })

        print("Finished training.")
        print("Creating checkpoint.")

        if best_model is None:
            best_model = {
                'epoch': self.epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_balanaced_acc,
                'auroc': val_auroc
            }

        print(f"Best Validation Acc. = {best_model['acc']} (Epoch = {best_model['epoch']})")
        filename = f'{self.save_path}/checkpoint.pth.tar'
        torch.save(best_model, filename)
        print("Finished creating checkpoint.")

        return best_model

    def evaluate(self, model, data_loader):

        model.eval()
        y_true = []
        y_pred = []
        loss_epoch = []
        progress_bar = tqdm(data_loader)
        with torch.no_grad():
            for x, y in progress_bar:
                x = x.to(self.device)
                y = y.to(self.device)

                with autocast(enabled=self.mixed_precision):
                    preds = model(x)
                    preds = preds.squeeze()
                    loss_classification = self.loss_cls(preds, y)

                loss_epoch.append(loss_classification.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds) >= 0.5)

            mean_loss_epoch = np.mean(loss_epoch)
            # avg_acc = self.accuracy(torch.stack(y_pred), torch.stack(y_true))
            auroc = self.auroc(torch.stack(y_pred).detach().cpu().float(), torch.stack(y_true).detach().cpu())
            avg_acc = balanced_accuracy_score(torch.stack(y_true).detach().cpu(), torch.stack(y_pred).detach().cpu())

        return mean_loss_epoch, avg_acc, auroc
