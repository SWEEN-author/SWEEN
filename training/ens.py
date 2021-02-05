import torch
import torch.nn.functional as F

def ens_train(sigma, gauss_num, num_classes, model, trainloader, optimizer, device):

  cl_total = 0.0
  input_total = 0

  for _, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    input_size = len(inputs)
    input_total += input_size

    new_shape = [input_size * gauss_num]
    new_shape.extend(inputs[0].shape)
    inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
    noise = torch.randn_like(inputs, device=device) * sigma
    noisy_inputs = inputs + noise

    outputs = model(noisy_inputs)
    outputs = outputs.reshape((input_size, gauss_num, num_classes))

    # Classification loss
    outputs_softmax = outputs
    outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
    classification_loss = 0
    for i in range(gauss_num):
        classification_loss += F.nll_loss(
        outputs_logsoftmax[:,i,:], targets, reduction='sum')
    classification_loss /= gauss_num
    cl_total += classification_loss.item()


    # Final objective function
    loss = classification_loss
    loss /= input_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  cl_total /= input_total
  print('Classification Loss: {}'.format(cl_total))
