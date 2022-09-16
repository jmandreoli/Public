# The purpose of this snippet is to show that the "key" bias in pytorch multi-head attention is redundant
# Consequences:
# * memory space for that bias is wasted
# * torch spends time computing operations involving that bias and propagating null gradients back
# Probably not significant, but still could be avoided

import torch
MHA = torch.nn.MultiheadAttention(embed_dim=12,num_heads=4,batch_first=True)
# variant with different key and value dimensions:
#MHA = torch.nn.MultiheadAttention(embed_dim=12,num_heads=4,batch_first=True,kdim=17,vdim=19)
B = 13; M = 10; N = 8 # batch size, input size, output size
output_shape = (B,N,MHA.embed_dim) # expected shape of the result
biases = dict(zip(('query','key','value'),torch.chunk(MHA.in_proj_bias.data,3)))
def test(bias_name='key'): # either 'key', 'query' or 'value'
  query_input = torch.rand(B,N,MHA.embed_dim) # query
  key_input = torch.rand(B,M,MHA.kdim) # key
  value_input =  torch.rand(B,M,MHA.vdim) # value
  with torch.no_grad():
    # Retrieve the bias
    bias = biases[bias_name]
    # First set the selected bias to some value.
    bias[...] = 42*torch.rand(bias.shape)
    # and collect the result with the given input
    y1,_ = MHA(query_input,key_input,value_input)
    # Now set the same bias to another value.
    bias[...] = 51*torch.rand(bias.shape)
    # and collect the result with the same input
    y2,_ = MHA(query_input,key_input,value_input)
    assert y1.shape == output_shape and y2.shape == output_shape # sanity check
    print(f'Output shape: {output_shape}')
  # observe that with bias_name='key', y1 and y2 and essentially identical; not with the other two
  print(f'Delta[{bias_name}] = {torch.max(abs(y2-y1)).item():.2g}')

