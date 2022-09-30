import torch


def tf2pytorch(tensor):
    return torch.tensor(tensor.numpy())


def convert_torch_tensor_to_numpy(torch_tensor, device='cpu'):
    if device == 'cpu':
        return torch_tensor.detach().numpy()
    else:
        return torch_tensor.cpu().detach().numpy()


def boolean_mask(inputs, inputs_key, mask):
    # print("Shape of " + mask_key +str(mask.shape))
    if inputs_key != 'prediction':
        input_states = inputs[inputs_key]
    else:
        input_states = inputs
    input_states_shape = input_states.shape
    # print("Shape of  " + inputs_key +str(input_states_shape))
    if inputs_key not in ('gt_future_is_valid', 'prediction'):
        input_states = input_states.reshape(input_states_shape[0] * input_states_shape[1], input_states_shape[2],
                                            input_states_shape[3])
        mask = mask.flatten().unsqueeze(-1).unsqueeze(-1).expand_as(input_states)
        output = input_states[mask]
        output = output.reshape(int(output.shape[0] / (input_states.shape[1] * input_states.shape[2])),
                                input_states.shape[1], input_states.shape[2])

    else:
        input_states = input_states.reshape(input_states_shape[0] * input_states_shape[1], input_states_shape[2])
        mask = mask.flatten().unsqueeze(-1).expand_as(input_states)
        # print(input_states.shape)
        # print(mask.shape)
        output = input_states[mask]
        # print(output.shape)
        output = output.reshape(int(output.shape[0] / input_states.shape[1]), input_states.shape[1])

    return output

