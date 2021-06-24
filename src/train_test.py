import torch


def train(model, optimizer, epoch, device, dataloader, loss_func):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        cur_loss = loss_func(output, target)

        cur_loss.backward()

        optimizer.step()

        print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(dataloader.dataset),
            100. * batch_idx / len(dataloader), cur_loss.item()))


def test(model, epoch, device, dataloader, loss_func, which_set):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            test_loss += loss_func(output, target).sum().item()
            prediction = output.argmax(dim=1, keepdim=True)

            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nEvaluating Epoch {} on {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, which_set, test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

    return correct / len(dataloader.dataset)
