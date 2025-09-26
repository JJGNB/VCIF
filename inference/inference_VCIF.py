import os
import cv2
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import torch
from PIL import Image,ImageOps
from skimage.io import imsave
from basicsr.archs.VCIF_arch import VCIF
from basicsr.archs.backbone_arch import backbone
torch.cuda.set_device(0)
seed=42
torch.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
weight_save_path="./checkpoints"
test_I_path="datasets/eval/IVF/infrared"
test_V_path="datasets/eval/IVF/visible"
save_path="./results/LLVIP_results"
if not os.path.exists(save_path):
    os.makedirs(save_path)
Test_Image_Number=len(os.listdir(test_I_path))
# print(Test_Image_Number)
i=0
net1=backbone(source="vi").to("cuda")
net2=backbone(source="ir").to("cuda")
jnet=VCIF(net1,net2,fusion_type="MCSE").to("cuda")
net1.load_state_dict(
        torch.load(os.path.join(weight_save_path, "backbone/backbone_for_vi.pkl"),
                   map_location="cuda:0")['weight'])
net1.eval()
net2.load_state_dict(
        torch.load(os.path.join(weight_save_path, "backbone/backbone_for_ir.pkl"))[
            'weight'])
net2.eval()
jnet.load_state_dict(torch.load(os.path.join(weight_save_path, "net_g_latest.pth"))['params'])
jnet.eval()
for file_name in os.listdir(test_I_path):
    i += 1
    Test_IR = Image.open(os.path.join(test_I_path, file_name))
    Test_Vis = Image.open(os.path.join(test_V_path, file_name))
    Test_Vis_numpy = cv2.imread(os.path.join(test_V_path, file_name))
    Test_IR_numpy = cv2.imread(os.path.join(test_I_path, file_name))
    Test_Vis_numpy = cv2.cvtColor(Test_Vis_numpy, cv2.COLOR_BGR2RGB)
    Test_IR_numpy = cv2.cvtColor(Test_IR_numpy, cv2.COLOR_BGR2RGB)
    Test_Vis_numpy = cv2.resize(Test_Vis_numpy, (512, 512))
    Test_IR_numpy = cv2.resize(Test_IR_numpy, (512, 512))
    Test_IR = transforms.Grayscale(1)(Test_IR)
    Test_IR = transforms.Resize([512,512])(Test_IR)
    img_test1 = transforms.ToTensor()(Test_IR)
    test_HE = transforms.Compose([
         ImageOps.equalize,
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    Test_HE=test_HE(Test_Vis)
    img_test2 = transforms.ToTensor()(Test_Vis)
    img_test2 = transforms.Resize([512, 512])(img_test2)
    img_test1=img_test1.unsqueeze(0)
    img_test2 = img_test2.unsqueeze(0)
    img_he=Test_HE.unsqueeze(0)
    img_test1 = img_test1.cuda()
    img_test2 = img_test2.cuda()
    target_layer=jnet.se_sub[0]
    input_tensor=torch.cat((img_test1, img_test2), dim=1)
    with torch.no_grad():
        fusion_img = jnet(torch.concat((img_test1, img_test2), dim=1))
        torchvision.utils.save_image(fusion_img, os.path.join(save_path, file_name))
        print("finish {0}".format(i))
