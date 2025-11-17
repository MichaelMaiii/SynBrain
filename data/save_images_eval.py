import numpy as np
import torch
from torchvision import transforms

sub = 1  
mode = "test"  
dir = "/home/bingxing2/ailab/group/ai4neuro/BrainVL/data/processed_data"
data = np.load(f'{dir}/subj0{sub}/nsd_{mode}_stim_sub{sub}.npy').astype(np.uint8)

transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((256, 256)),  
    transforms.ToTensor()  
])

all_images = []
for img in data:
    # (425,425,3）
    img_tensor = transform(img)  #(3,256,256)
    all_images.append(img_tensor)

all_images_tensor = torch.stack(all_images, dim=0)

eval_path = "/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn/evals"
save_path = f"{eval_path}/all_images.pt"
torch.save(all_images_tensor, save_path)

print(all_images_tensor.shape)  #(N, 3, 256, 256)