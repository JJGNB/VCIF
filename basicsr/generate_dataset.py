import os

import numpy as np
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.misc import set_random_seed
from torchvision import transforms
import torchvision
from PIL import Image
set_random_seed(2024)
input_lq_folder_0="./datasets/train/CPDM/lq_4/lq_0"
input_lq_folder_45="./datasets/train/CPDM/lq_4/lq_45"
input_lq_folder_90="./datasets/train/CPDM/lq_4/lq_90"
input_lq_folder_135="./datasets/train/CPDM/lq_4/lq_135"
input_gt_folder_0="./datasets/train/CPDM/gt_4/gt_0"
input_gt_folder_45="./datasets/train/CPDM/gt_4/gt_45"
input_gt_folder_90="./datasets/train/CPDM/gt_4/gt_90"
input_gt_folder_135="./datasets/train/CPDM/gt_4/gt_135"
output_lq_folder_0="./datasets/train/CPDM/lq_crop/lq_0"
output_lq_folder_45="./datasets/train/CPDM/lq_crop/lq_45"
output_lq_folder_90="./datasets/train/CPDM/lq_crop/lq_90"
output_lq_folder_135="./datasets/train/CPDM/lq_crop/lq_135"
output_gt_folder_0="./datasets/train/CPDM/gt_crop/gt_0"
output_gt_folder_45="./datasets/train/CPDM/gt_crop/gt_45"
output_gt_folder_90="./datasets/train/CPDM/gt_crop/gt_90"
output_gt_folder_135="./datasets/train/CPDM/gt_crop/gt_135"
tfs=transforms.ToTensor()
filename_dir=os.listdir(input_lq_folder_0)
if not os.path.exists(output_lq_folder_0):
    os.makedirs(output_lq_folder_0)
    os.makedirs(output_lq_folder_45)
    os.makedirs(output_lq_folder_90)
    os.makedirs(output_lq_folder_135)
    os.makedirs(output_gt_folder_0)
    os.makedirs(output_gt_folder_45)
    os.makedirs(output_gt_folder_90)
    os.makedirs(output_gt_folder_135)
totalloop=2
for loop in range(totalloop):
    step=0
    for file_name in filename_dir:
        step+=1
        base_name=file_name.replace("_0.png","")
        img_gt_0=np.array(Image.open(os.path.join(input_gt_folder_0,base_name+"_0.png")))
        img_gt_45=np.array(Image.open(os.path.join(input_gt_folder_45,base_name+"_45.png")))
        img_gt_90=np.array(Image.open(os.path.join(input_gt_folder_90,base_name+"_90.png")))
        img_gt_135=np.array(Image.open(os.path.join(input_gt_folder_135,base_name+"_135.png")))
        img_lq_0=np.array(Image.open(os.path.join(input_lq_folder_0,base_name+"_0.png")))
        img_lq_45=np.array(Image.open(os.path.join(input_lq_folder_45,base_name+"_45.png")))
        img_lq_90=np.array(Image.open(os.path.join(input_lq_folder_90,base_name+"_90.png")))
        img_lq_135=np.array(Image.open(os.path.join(input_lq_folder_135,base_name+"_135.png")))
        [img_gt_0,img_gt_45,img_gt_90,img_gt_135],[img_lq_0,img_lq_45,img_lq_90,img_lq_135] = paired_random_crop([img_gt_0,img_gt_45,img_gt_90,img_gt_135],[img_lq_0,img_lq_45,img_lq_90,img_lq_135],384)
                # flip, rotation
        img_gt_0,img_gt_45,img_gt_90,img_gt_135,img_lq_0,img_lq_45,img_lq_90,img_lq_135 = augment([img_gt_0,img_gt_45,img_gt_90,img_gt_135,img_lq_0,img_lq_45,img_lq_90,img_lq_135], True, True)
        img_gt_0=tfs(img_gt_0)
        img_gt_45=tfs(img_gt_45)
        img_gt_90=tfs(img_gt_90)
        img_gt_135=tfs(img_gt_135)
        img_lq_0=tfs(img_lq_0)
        img_lq_45=tfs(img_lq_45)
        img_lq_90=tfs(img_lq_90)
        img_lq_135=tfs(img_lq_135)
        torchvision.utils.save_image(img_gt_0,os.path.join(output_gt_folder_0,base_name+"_loop"+str(loop+1)+"_0.png"))
        torchvision.utils.save_image(img_gt_45,os.path.join(output_gt_folder_45,base_name+"_loop"+str(loop+1)+"_45.png"))
        torchvision.utils.save_image(img_gt_90,os.path.join(output_gt_folder_90,base_name+"_loop"+str(loop+1)+"_90.png"))
        torchvision.utils.save_image(img_gt_135,os.path.join(output_gt_folder_135,base_name+"_loop"+str(loop+1)+"_135.png"))
        torchvision.utils.save_image(img_lq_0,os.path.join(output_lq_folder_0,base_name+"_loop"+str(loop+1)+"_0.png"))
        torchvision.utils.save_image(img_lq_45,os.path.join(output_lq_folder_45,base_name+"_loop"+str(loop+1)+"_45.png"))
        torchvision.utils.save_image(img_lq_90,os.path.join(output_lq_folder_90,base_name+"_loop"+str(loop+1)+"_90.png"))
        torchvision.utils.save_image(img_lq_135,os.path.join(output_lq_folder_135,base_name+"_loop"+str(loop+1)+"_135.png"))
        print("loop:"+str(loop+1)+"  "+"step:"+str(step)+"done!")
