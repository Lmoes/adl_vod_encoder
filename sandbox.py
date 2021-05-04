import torch

some_tensor = torch.randn(15)  # one tensor length 15
list_of_tensors = some_tensor.split(5)  # list of 3 tensors of lenght 5
some_layer = torch.nn.Linear(5, 1)

# Now loop over the tensors and apply the layer on each and collect the outputs
output_list = []
for tensor in list_of_tensors:
    output = some_layer(tensor)
    output_list.append(output)