import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, train_loader, scheduler, test_loader=None, epochs=25):
    model.train()
    losses, accuracies = [], []

    for epoch in range(epochs):
        true_labels, predictions = [], []
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        scheduler.step(epoch_loss)

    return model
