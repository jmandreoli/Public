r"""
The purpose of this snippet is to show that the pytorch implementation of [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) is overparametrized in its management of biases.

The bias for the query, key and value are held in attribute `in_proj_bias` (present by default, can be disabled by passing `bias=False` in the invocation).

The snippet shows that modifying the key bias does not change the result. Changing the other two biases does change the result.

This can be shown from the formula (although it is not explicitly given in the doc, one has to guess it) using the fact that softmax is invariant to an additive constant.

The practical consequence is that it spends some time at each iteration both
* forward, involving a parameter which does not change the result and
* backward, propagating null gradients to update it.

Probably negligible in the tsunami of propagations which take place, still could be removed.
"""
import torch
def randomize(*l): # randomize the content of each tensor in l
  for x in l: x[...] = 42*torch.rand(1)*torch.rand(x.shape)
def cmp(u,v): return torch.amax(abs(v-u)).item() # compares two tensors

a = torch.nn.MultiheadAttention(embed_dim=12,num_heads=4,batch_first=True)
#a = torch.nn.MultiheadAttention(embed_dim=12,num_heads=4,batch_first=True,kdim=17,vdim=19) # variant with key and value dimensions
randomize(a.in_proj_bias.data,a.out_proj.bias.data) # replaces default initialisation (zeros) by random
B = 13; M = 10; N = 8
yʹ = torch.rand(B,N,a.embed_dim) # query input
xʹ = torch.rand(B,M,a.kdim) # key input
x =  torch.rand(B,M,a.vdim) # value input
outs = (B,N,a.embed_dim) # shape of the output
print(f'Output shape: {outs}')
with torch.no_grad():
  for bias_name,bias in zip(('query','key','value'),torch.chunk(a.in_proj_bias.data,3)): # retrieve the three biases (query,key,value)
    # First randomize the selected bias and compute the output.
    randomize(bias); y1,_ = a(yʹ,xʹ,x)
    # Now randomize the same bias again and compute the output.
    randomize(bias); y2,_ = a(yʹ,xʹ,x)
    assert y1.shape == outs and y2.shape == outs # sanity check
    # compare the two outputs
    print(f'Delta[{bias_name}] = {cmp(y1,y2):.2g}')
