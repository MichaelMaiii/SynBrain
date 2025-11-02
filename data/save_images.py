import numpy as np
from PIL import Image
import os

# Load npy format Stimuli
dir = "/home/bingxing2/ailab/group/ai4neuro/BrainVL/data/processed_data"
sub = 1
mode = 'test'

data = np.load(f'{dir}/subj0{sub}/nsd_{mode}_stim_sub{sub}.npy')

print("Data shape:", data.shape)  #(N, 425, 425, 3)

## test image are shared across subjects, you can also save to 'xxx/evals/test_img' for final evaluation
# output_dir = f'{eval_dir}/eval/test_img'
output_dir = f'{dir}/subj0{sub}/{mode}_img'
os.makedirs(output_dir, exist_ok=True)

for i in range(data.shape[0]):
    img = Image.fromarray(data[i].astype(np.uint8))  
    img.save(os.path.join(output_dir, f"{i}.png"))

print("All images are saving to:", output_dir)