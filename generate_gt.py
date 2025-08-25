import os
import cv2
import torchvision
from torchvision import transforms
import numpy as np
import torch
from basicsr.archs.backbone_arch import backbone
from protocol import run_pipeline, AAB
weight_save_path="./checkpoints/backbone"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# you need to change the path of folders to meet the usage of "torchvision.datasets.ImageFolder"
test_I_path="./datasets/train/IVF/ir_patches/ir_patches"
test_V_path="./datasets/train/IVF/vi_patches/vi_patches"
save_path= 'datasets/train/IVF/en_test/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tfs = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
tfs_vi=transforms.Compose([
        transforms.ToTensor()
    ])
Test_Image_Number=len(os.listdir(test_I_path+'/ir'))
Data_VIS=torchvision.datasets.ImageFolder(test_V_path,transform=tfs_vi)
Data_IR=torchvision.datasets.ImageFolder(test_I_path,transform=tfs)
dataloader_VIS = torch.utils.data.DataLoader(Data_VIS, 1,shuffle=False)
dataloader_IR=torch.utils.data.DataLoader(Data_IR, 1,shuffle=False)
net=backbone()
net.load_state_dict(torch.load(os.path.join(weight_save_path, "backbone_for_vi.pkl"))['weight'])
net.eval()
net=net.cuda()
data_iter_VIS = iter(dataloader_VIS)
data_iter_IR = iter(dataloader_IR)
name_list=sorted(os.listdir(test_V_path+'/vi'))
for i in range(Test_Image_Number):
    data_VIS, _ = next(data_iter_VIS)
    data_IR, _ = next(data_iter_IR)
    data_VIS = data_VIS.cuda()
    data_IR = data_IR.cuda()
    with torch.no_grad():
        ab=AAB(data_VIS.unsqueeze(dim=0)).item()
        abb = ab*100
        abb = abb // 1
        if abb>8:
            en_img=net(data_VIS)
            [r, g, b] = torch.chunk(en_img.squeeze(dim=0), 3, dim=0)
            r = r.squeeze(dim=0)
            g = g.squeeze(dim=0)
            b = b.squeeze(dim=0)
            r = r.unsqueeze(dim=-1)
            g = g.unsqueeze(dim=-1)
            b = b.unsqueeze(dim=-1)
            img_oral = torch.concat([r, g, b], dim=-1).detach().cpu().numpy()
            img_post = run_pipeline(img_oral, 0, 1, 1, 0, 0, 0, 0)
            [r, g, b] = torch.chunk(torch.tensor(img_post), 3, dim=-1)
            img_post = torch.concat([b, g, r], dim=-1).numpy()
            img_post *= 255
            img_post = np.clip(img_post, 0, 255)
            img_post = img_post.astype("uint8")
            cv2.imwrite(save_path +name_list[i], img_post)
        else:
            [r, g, b] = torch.chunk(data_VIS.squeeze(dim=0), 3, dim=0)
            r = r.squeeze(dim=0)
            g = g.squeeze(dim=0)
            b = b.squeeze(dim=0)
            r = r.unsqueeze(dim=-1)
            g = g.unsqueeze(dim=-1)
            b = b.unsqueeze(dim=-1)
            img_oral = torch.concat([r, g, b], dim=-1).detach().cpu().numpy()
            img_post = run_pipeline(img_oral, 1, 1, 1, 1, 1, 1, 0)
            [r, g, b] = torch.chunk(torch.tensor(img_post), 3, dim=-1)
            img_post = torch.concat([b, g, r], dim=-1).numpy()
            img_post *= 255
            img_post = np.clip(img_post, 0, 255)
            img_post = img_post.astype("uint8")
            cv2.imwrite(save_path +name_list[i], img_post)
        print(name_list[i].split("/")[-1])
        print("save {0}".format(str(i)))