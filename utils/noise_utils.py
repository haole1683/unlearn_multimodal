import torch
import torch.nn.functional as F
import torch.nn as nn

def gen_perturbation(generator ,text_embedding, noise_shape,z_latent=None, evaluate=False,args=None):
    batch_size = text_embedding.shape[0]
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
    
    if hasattr(args, 'device'):
        device = args.device
    else:
        device = text_embedding.device
    
    if z_latent is None:
        z_latent = torch.randn(batch_size, 100).to(device)
        
    if evaluate:
        with torch.no_grad():
            output, _ = generator(z_latent, sec_emb)
    else:
        output, _ = generator(z_latent, sec_emb)
        
    limit_method = 'traditional'
    if limit_method == 'traditional':
        noise = limit_noise(output, norm_type="linf", epsilon=16, device=device, noise_shape=noise_shape)
    elif limit_method == 'sunye':
        noise = limit_noise_with_activation(output, epsilon=16, device=device, noise_shape=noise_shape)
    else:
        return output
    return noise

def save_noise(noise, path):
    noise = noise.detach().cpu()
    torch.save(noise, path)

def limit_noise(noise, norm_type="l2", epsilon=16, device="cuda:0", noise_shape=[3,32,32]):
    delta_im = noise
    
    if norm_type == "l2":
        temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        delta_im = delta_im * epsilon / temp
    elif norm_type == "linf":
        delta_im = torch.clamp(delta_im, -epsilon / 255., epsilon / 255)  # torch.Size([16, 3, 256, 256])

    delta_im = delta_im.to(device)
    delta_im = F.interpolate(delta_im, (noise_shape[-2], noise_shape[-1]))
    
    return delta_im

def limit_noise_with_activation(noise, epsilon=16, device="cuda:0", noise_shape=[3,32,32]):
    """limit noise with activation function

    Args:
        noise (Tensor): [batch,channel,width,height]
        epsilon (int, optional): epsilon. Defaults to 16.
        device (str, optional): device. Defaults to "cuda:0".
        noise_shape (list, optional): output shape. Defaults to [3,32,32].

    Returns:
        _type_: _description_
    """
    delta_im = noise
    
    activation_fn = nn.Tanh()
    
    delta_im = activation_fn(delta_im)
    delta_im = delta_im * epsilon / 255. 
    
    # resize to target noise shape
    
    delta_im = F.interpolate(delta_im, (noise_shape[-2], noise_shape[-1]))
    
    return delta_im
    
    
    
    