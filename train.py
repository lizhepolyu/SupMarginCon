"""
This module is used to train the speaker recognition model and evaluate its performance.
"""

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from SupMarginCon import SupMarginCon
from spkencoder import ECAPA_TDNN
import os
import time
import numpy as np
import random, glob
from scipy import signal
import torchmetrics

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        """
        Initialize the ECAPAModel with the given parameters.

        Args:
            lr (float): Initial learning rate.
            lr_decay (float): Learning rate decay factor.
            C (int): Number of channels in the model.
            n_class (int): Number of speaker classes.
            m (float): Margin parameter for AAMSoftmax loss.
            s (float): Scale parameter for AAMSoftmax loss.
            test_step (int): Step size for learning rate scheduler.
            **kwargs: Additional keyword arguments.
        """
        super(ECAPAModel, self).__init__()

        # Initialize the speaker encoder model and wrap it with DataParallel for multi-GPU training
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        self.speaker_encoder = nn.DataParallel(self.speaker_encoder)

        # Define the speaker loss and supervised contrastive loss
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.SupConLoss = SupMarginCon().cuda()
        self.classifier = nn.Linear(192, 1211).cuda()

        # Set up the optimizer and learning rate scheduler
        self.optim = torch.optim.Adam(
            list(self.parameters()) + list(self.classifier.parameters()),
            lr=lr,
            weight_decay=2e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=test_step,
            gamma=lr_decay
        )

        # Print the total number of parameters in the model
        print(time.strftime("%m-%d %H:%M:%S") + " Model parameter number = %.2f" % (
                sum(param.numel() for param in self.speaker_encoder.module.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        """
        Train the network for one epoch.

        Args:
            epoch (int): Current epoch number.
            loader (DataLoader): DataLoader for training data.

        Returns:
            tuple: Average loss, current learning rate, and training accuracy.
        """
        self.train()  # Set the model to training mode
        self.scheduler.step(epoch - 1)  # Update the learning rate scheduler
        total_loss = 0
        total_correct = 0
        total_samples = 0
        lr = self.optim.param_groups[0]['lr']  # Get the current learning rate

        for num, (dataaug, datares, labels) in enumerate(loader, start=1):
            self.optim.zero_grad()  # Clear the gradients
            dataaug = dataaug.cuda(non_blocking=True)
            datares = datares.cuda(non_blocking=True)
            labels = labels.cuda()

            # Concatenate augmented and original data
            audio = torch.cat([dataaug, datares], dim=0)
            bsz = labels.size(0)  # Batch size

            # Pass the audio through the speaker encoder to get features
            features = self.speaker_encoder(audio, aug=True)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)  # Split features into two halves

            # Stack features for contrastive loss
            features = torch.stack([f1, f2], dim=1)

            # Compute the supervised contrastive loss
            nloss = self.SupConLoss(features, labels)

            nloss.backward()  # Backpropagate the loss
            self.optim.step()  # Update the model parameters

            total_loss += nloss.item()  # Accumulate loss
            total_samples += labels.size(0)  # Accumulate sample count

            # ========== Calculate accuracy ==========
            # Use the classifier to map features to class predictions
            outputs = self.classifier(f1)
            _, preds = torch.max(outputs, 1)  # Get the predicted classes
            correct = torch.sum(preds == labels)  # Count correct predictions
            total_correct += correct.item()  # Accumulate correct predictions
            # ========================================

            total_steps = len(loader)  # Total number of steps in the epoch
            if num % 10 == 0 or num == total_steps:
                # Calculate accuracy and other metrics for logging
                accuracy = len(labels) * total_correct / total_samples
                current_time = time.strftime("%m-%d %H:%M:%S")
                training_progress = 100.0 * num / total_steps
                average_loss = total_loss / num
                sys.stderr.write(f"{current_time} [{epoch:2d}] Lr: {lr:.6f}, Training: {training_progress:.2f}%, "
                                 f"Loss: {average_loss:.5f}, Acc: {accuracy:.2f}%\r")
                if (num % 10 == 0 or num == total_steps) and num % 50 == 0:
                    sys.stderr.flush()

        # Return average loss, current learning rate, and accuracy
        return total_loss / num, lr, accuracy

    def eval_network(self, eval_list_path, eval_data_path):
        """
        Evaluate the network's performance on a given evaluation list.

        Args:
            eval_list_path (str): Path to the evaluation list file.
            eval_data_path (str): Directory path where evaluation data is stored.

        Returns:
            tuple: Equal Error Rate (EER) and minimum Detection Cost Function (minDCF).
        """
        self.eval()  # Set the model to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Read the evaluation list and collect unique file names
        with open(eval_list_path, 'r') as f:
            lines = f.readlines()

        pairs = []
        unique_files = set()
        for line in lines:
            label, file1, file2 = line.strip().split()
            pairs.append((int(label), file1, file2))
            unique_files.update([file1, file2])

        # Pre-compute embeddings for all unique files
        embeddings = {}
        max_audio_length = 300 * 160 + 240  # Define maximum audio length

        for file_name in tqdm.tqdm(sorted(unique_files), desc="Processing audio files"):
            audio_path = os.path.join(eval_data_path, file_name)
            audio, sample_rate = soundfile.read(audio_path)
            audio = np.asarray(audio, dtype=np.float32)

            # Process full-length audio
            data_full = torch.from_numpy(audio).unsqueeze(0).to(device)

            # Process segmented audio fragments for robustness
            if len(audio) <= max_audio_length:
                shortage = max_audio_length - len(audio)
                audio = np.pad(audio, (0, shortage), 'wrap')  # Pad audio if too short
            start_frames = np.linspace(0, len(audio) - max_audio_length, num=5)
            feats = [audio[int(s):int(s) + max_audio_length] for s in start_frames]
            data_split = torch.from_numpy(np.stack(feats)).to(device)

            # Compute embeddings for full and split audio
            with torch.no_grad():
                embedding_full = self.speaker_encoder(x=data_full, aug=False)
                embedding_full = F.normalize(embedding_full, p=2, dim=-1)
                embedding_split = self.speaker_encoder(x=data_split, aug=False)
                embedding_split = F.normalize(embedding_split, p=2, dim=-1)
            embeddings[file_name] = (embedding_full, embedding_split)  # Store embeddings

        # Compute similarity scores and labels for all pairs
        scores = []
        labels = []

        for label, file1, file2 in pairs:
            emb1_full, emb1_split = embeddings[file1]
            emb2_full, emb2_split = embeddings[file2]

            # Compute similarity scores between embeddings
            score_full = torch.mean(torch.matmul(emb1_full, emb2_full.T))
            score_split = torch.mean(torch.matmul(emb1_split, emb2_split.T))
            score = (score_full + score_split) / 2.0  # Average the scores
            scores.append(score.item())
            labels.append(label)

        # Compute Equal Error Rate (EER) and minimum Detection Cost Function (minDCF)
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

        return EER, minDCF

    def save_parameters(self, path):
        """
        Save model parameters to the specified path.

        Args:
            path (str): File path to save the model parameters.
        """
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        """
        Load model parameters from the specified path.

        Args:
            path (str): File path of the model parameters.
        """
        state_dict = torch.load(path)
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name.startswith('module.') and name[7:] in own_state:
                name = name[7:]  # Remove 'module.' prefix if present
            if name in own_state and own_state[name].size() == param.size():
                own_state[name].copy_(param)  # Copy parameter if size matches
            else:
                print(f"Parameter {name} not loaded.")  # Warn if parameter is not loaded
