```
class Gamma(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, param):
        ctx.save_for_backward(input, param)

        return param*input

    def backward(ctx, out_grad): # 返回两个梯度，分别是 input 和 param(gamma的)
        input, param = ctx.saved_tensors
        z = out_grad*input
        ret = torch.sum(z,(0,1))
        print(f"backward, input: {input}, out_grad: {out_grad}")
        return param*out_grad, ret


    self.gamma_1 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True)
    self.gamma_2 = nn.Parameter(init_values * torch.ones((out_features)), requires_grad=True) 

    if Block.use_customized_gamma:
        x = x + self.drop_path(self.gamma(p, self.gamma_1))
    else:
        x = x + self.drop_path(self.gamma_1 * p)
```
