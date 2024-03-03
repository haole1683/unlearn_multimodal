import torch
import torch.nn.functional as F

def gen_perturbation(generator ,text_embedding, noise_shape, evaluate=False,args=None):
    batch_size = noise_shape[0]
    sec_emb = text_embedding
    if evaluate:
        generator.eval()

    if not hasattr(args, 'norm_type'):
        norm_type = "l2"
    else:
        norm_type = args.norm_type
    
    if not hasattr(args, 'epsilon'):
        epsilon = 16
    else:
        epsilon = args.epsilon
    
    if not hasattr(args, 'device'):
        device = 'cuda:0'
    else:
        device = args.device
        
    noise = torch.randn(batch_size, 100).to(device)
    if evaluate:
        with torch.no_grad():
            gen_image, _ = generator(noise, sec_emb)
    else:
        gen_image, _ = generator(noise, sec_emb)
    delta_im = gen_image
    if norm_type == "l2":
        temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        delta_im = delta_im * epsilon / temp
    elif norm_type == "linf":
        delta_im = torch.clamp(delta_im, -epsilon / 255., epsilon / 255)  # torch.Size([16, 3, 256, 256])

    delta_im = delta_im.to(device)
    delta_im = F.interpolate(delta_im, (noise_shape[-2], noise_shape[-1]))
    
    return delta_im

def save_noise(noise, path):
    noise = noise.detach().cpu()
    torch.save(noise, path)

    

    
    
    
    