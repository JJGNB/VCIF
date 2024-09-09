# VCIF: Visually Compelling Infrared and Visible Image Fusion under Darkness

## All figures can be found in the "Fig" folder
## Training and testing (TO BE DONE...)
## 1. Create Environment
See details in requirements.txt.
## 2. Prepare Your Dataset
LLIE tasks: [LOL](https://github.com/weichen582/RetinexNet) 

IVF tasks: [MSRS](https://github.com/Linfeng-Tang/PIAFusion) , [LLVIP](https://github.com/bupt-ai-cz/LLVIP) , [M3FD](https://github.com/JinyuanLiu-CV/TarDAL)

For LLIE training:
```bash
    dataset/
        train/
            LLIE/
                low_real_patches_512/ #low images
                normal_real_patches_512/ #ground truth
```
For IVF training:
```bash
    dataset/
        train/
            IVF/
                ir_patches_512/ #ir images
                vi_patches_512/ #vi images
                en_patches_512/ #pseudo ground truth
```
For LLIE testing:
```bash
    dataset/
        eval15/
            high/ #ground truth
            low/ #low images
```
For IVF testing:
```bash
    dataset/
        LLVIP_50/ # or MSRS_50 or M3FD_30
            infrared/ #vislbe images
            visible/ #vislbe images
```
You can also put the images you want to test in the above folders.
## 3. Pretrain Weights
You can find the pretrain weights at
```bash
    checkpoints/
        backbone_for_ir.pkl/ # net R
            backbone_for_vi.pkl/ # net L
            joint_net.pkl/ # fusion network
```
## 4. Testing:
Get Fused results:
```shell
python en_ir_joint_test.py
```
If you want to see the pseudo ground truth for training:
```shell
python backbone_test.py
```
## 5. Training:
stage I for net L and R:

L:(1) put your dataset(LOL or SICE) in the folders shown in "2. Prepare Your Dataset"

(2) run
```shell
python crop_patches_for_LLIE.py
```

  (3) revise 'train_type="enhancement"', 'checkpo' and run
```shell
python backbone_train.py
```
  (4) run
```shell
python generate_gt.py
```

(5) run
```shell
python crop_patches_for_IVF.py
```
(6) revise 'train_type="enhancement_post"','checkpo' and run
```shell
python backbone_train.py
```
R: (1) put youre dataset(LLVIP, MSRS or M3FD) in the folders shown in "2. Prepare Your Dataset"

(2) revise 'train_type="enhancement"' and run (to be done)






