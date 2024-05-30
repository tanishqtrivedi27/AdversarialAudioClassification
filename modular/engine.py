import torch

def train_model(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0
    for _, (batch_input, batch_target) in dataloader:
        batch_input, batch_target = batch_input.to(device), batch_target.to(device)

        predictions = model(batch_input)
        loss = loss_fn(predictions, batch_target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    return train_loss

def test_model(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.inference_mode():
        for _, (batch_input, batch_target) in dataloader:
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)

            predictions = model(batch_input)
            loss = loss_fn(predictions, batch_target)
            test_loss += loss.item()

            test_pred_labels = predictions.argmax(dim=1)
            accuracy += ((test_pred_labels == batch_target).sum().item()/len(test_pred_labels))

    return test_loss, accuracy  