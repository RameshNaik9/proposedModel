from torch import nn
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class SlotAttention(nn.Module):
    def __init__(self, num_classes, dim, class_means, vis=False, vis_id=0, loss_status=1, power=1):
        super(SlotAttention, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.class_means = nn.Parameter(class_means, requires_grad=False)  # Predefined class means
        self.scale = dim ** -0.5
        self.vis = vis
        self.vis_id = vis_id
        self.loss_status = loss_status
        self.power = power

        # Transform layers
        self.to_q = nn.Sequential(nn.Linear(dim, dim),)
        self.to_k = nn.Sequential(nn.Linear(dim, dim),)

    def forward(self, inputs, inputs_x):
        b, n, d = inputs.shape
        slots = self.class_means.unsqueeze(0).expand(b, -1, -1)  # Use predefined class means as slots

        # Compute keys and queries
        k = self.to_k(inputs)
        q = self.to_q(slots)

        # Calculate dot-product attention
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = F.softmax(dots, dim=-1)  # Normalize the attention scores

        # Apply attention to get updates
        updates = torch.einsum('bjd,bij->bid', inputs_x, attn)
        updates = updates / inputs_x.size(2)  # Normalize updates

        # if self.vis:
        #     if self.slots_per_class > 1:
        #         new_slots_vis = torch.zeros((slots_vis.size(0), self.num_classes, slots_vis.size(-1)))
        #         for slot_class in range(self.num_classes):
        #             new_slots_vis[:, slot_class] = torch.sum(torch.cat([slots_vis[:, self.slots_per_class*slot_class: self.slots_per_class*(slot_class+1)]], dim=1), dim=1, keepdim=False)
        #         slots_vis = new_slots_vis.to(updates.device)

        #     slots_vis = slots_vis[self.vis_id]
        #     slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max()-slots_vis.min()) * 255.).reshape(slots_vis.shape[:1]+(int(slots_vis.size(1)**0.5), int(slots_vis.size(1)**0.5)))
        #     slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
        #     for id, image in enumerate(slots_vis):
        #         image = Image.fromarray(image, mode='L')
        #         image.save(f'sloter/vis/slot_{id:d}.png')
        #     # print(self.loss_status*torch.sum(attn.clone(), dim=2, keepdim=False))
        #     # print(self.loss_status*torch.sum(updates.clone(), dim=2, keepdim=False))

        # if self.slots_per_class > 1:
        #     new_updates = torch.zeros((updates.size(0), self.num_classes, updates.size(-1)))
        #     for slot_class in range(self.num_classes):
        #         new_updates[:, slot_class] = torch.sum(updates[:, self.slots_per_class*slot_class: self.slots_per_class*(slot_class+1)], dim=1, keepdim=False)
        #     updates = new_updates.to(updates.device)

        attn_relu = torch.relu(attn)
        slot_loss = torch.sum(attn_relu) / attn.size(0) / attn.size(1) / attn.size(2)
        return self.loss_status * torch.sum(updates, dim=2), torch.pow(slot_loss, self.power)
