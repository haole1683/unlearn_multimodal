import torch
import torch.nn.functional as F

def gen_perturbation(generator ,text_embedding, images, args):
    batch_size = images.shape[0]
    sec_emb = text_embedding
    device = args.device
    noise = torch.randn(batch_size, 100).to(device)
    gen_image, _ = generator(noise, sec_emb)
    delta_im = gen_image
    norm_type = args.norm_type
    epsilon = args.epsilon
    if norm_type == "l2":
        temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        delta_im = delta_im * epsilon / temp
    elif norm_type == "linf":
        delta_im = torch.clamp(delta_im, -epsilon / 255., epsilon / 255)  # torch.Size([16, 3, 256, 256])

    delta_im = delta_im.to(images.device)
    delta_im = F.interpolate(delta_im, (images.shape[-2], images.shape[-1]))
    
    return delta_im
