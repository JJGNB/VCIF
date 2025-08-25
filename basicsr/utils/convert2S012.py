import torch
def I2S012(img_list,save_img=False):
    if isinstance(img_list,list):
        img_0,img_1,img_2,img_3=img_list
    else:
        img_0=img_list[:,0:3,:,:]
        img_1=img_list[:,3:6,:,:]
        img_2=img_list[:,6:9,:,:]
        img_3=img_list[:,9:12,:,:]
    S0=0.5*(img_0+img_1+img_2+img_3)
    S1=img_0-img_2
    S2=img_1-img_3
    DOLP=torch.sqrt(S1**2+S2**2)/S0
    if save_img:
        S0=torch.clip(S0,0,1)
        DOLP=torch.clip(DOLP,0,1)
        return S0,DOLP
    return S0,S1,S2