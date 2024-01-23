import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class InfoNCE(nn.Module):
    def __init__(self, logit_scale):
        super(InfoNCE, self).__init__()
        self.logit_scale = logit_scale

    def forward(self, h1, h2):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            h1 = AllGatherFunction.apply(h1)
            h2 = AllGatherFunction.apply(h2)
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm', [h1, h2]) * temperature.to(device)
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=-1)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = F.cross_entropy(logits, labels)
        return loss
        
class ClsHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    """
    def __init__(self, in_channels, num_classes=1054, with_avg_pool=False):
        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(in_channels, num_classes, bias=False) # class centroid
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, x,y):
        if self.with_avg_pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        output = self.fc_cls(x)
        logits = self.sigmoid(output)
        loss = self.loss_fn(logits,y)
        return loss

# https://github.com/openai/CLIP/issues/111#issuecomment-931955836
class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype
        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        grad_output = grad_output.contiguous() # contiguous error
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)

class AllGatherFunction_ReduceOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.contiguous() # contiguous error
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]