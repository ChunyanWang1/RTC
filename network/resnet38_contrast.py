import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

import network.resnet38d
from tool import pyutils

#from network.aspp import ASPP
from network.pamr import PAMR

# from network.gci import GCI
# from network.align import AlignModule
PAMR_KERNEL = [1, 2, 4, 8, 12, 24] #1, 2, 4, 8, 12, 24
PAMR_ITER=10


def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x

def focal_loss(x, p = 1, c = 0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)

class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout7 = nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(4096+256,4096, 1, bias=False) #4096
        # self.fc_proj1 = nn.Conv2d(4096, 128, 1, bias=False) #128
        # self.fc_proj2 = nn.Conv2d(4096, 128, 1, bias=False) #128
        # self.fc_proj3 = nn.Conv2d(4096, 128, 1, bias=False) #128

        
        self.proj1 = nn.Conv2d(4096, 256, 1, bias=False) #128
        self.proj2 = nn.Conv2d(256+48, 256, 1, bias=False) #128
        

        self.f8_3 = nn.Conv2d(512, 64, 1, bias=False)#512
        self.f8_4 = nn.Conv2d(1024, 128, 1, bias=False)#1024
        self.f8_5 = nn.Conv2d(4096, 128, 1, bias=False)#1024
        self.f9_1 = nn.Conv2d(256, 128, 1, bias=False)#192
        self.f9_2 = nn.Conv2d(64+3, 64, 1, bias=False)
        #self.f9 = nn.Conv2d(192+3, 192, 1, bias=False)


        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, 256, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

        self.fc8_seg_conv3 = nn.Conv2d(4096, 256, (3, 3), stride=1, padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv3.weight)

        self.fc8_seg_conv4 = nn.Conv2d(512, 256, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv4.weight)

        self.fc8_seg_conv5 = nn.Conv2d(256, 48, (3, 3),stride=1, padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv5.weight)

        # # # self.shallow_conv = nn.Conv2d(256, 128, 1, bias=False) #128



        
        self.classifier1 = nn.Conv2d(256, 21, 1, bias=False)
        self.classifier2 = torch.nn.Conv2d(256, 21, 1, bias=False)
        


        #torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.proj1.weight)
        torch.nn.init.xavier_uniform_(self.proj2.weight)
        torch.nn.init.xavier_uniform_(self.classifier1.weight)
        torch.nn.init.xavier_uniform_(self.classifier2.weight)
        
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9_2.weight)
        torch.nn.init.xavier_uniform_(self.f9_1.weight, gain=4)


        #self.from_scratch_layers = [self.proj1,self.proj2, self.classifier2,self.classifier1,self.fc8_seg_conv1,self.fc8_seg_conv2,self.fc8_seg_conv3,self.fc8_seg_conv4]#self.f8_3, self.f8_4, self.f9, ,self.fc_proj2,self.classifier2, self.fc_fea, self.f1,self.f2,self.f3
        self.from_scratch_layers = [self.fc8,self.f8_3, self.f8_4, self.f8_5,self.f9_1,self.f9_2, self.proj1,self.classifier1,self.proj2,self.classifier2,self.fc8_seg_conv1,self.fc8_seg_conv2,self.fc8_seg_conv3,self.fc8_seg_conv4,self.fc8_seg_conv5]#,self.fc8_seg_conv5  self.fc8,
        self._aff = PAMR(PAMR_ITER, PAMR_KERNEL)
        #self.from_scratch_layers += self.shallow_mask.from_scratch_layers
        # for m in self.aspp.modules():
        #     if isinstance(m, (nn.Conv2d,nn.BatchNorm2d)):
        #         self.from_scratch_layers.append(m)
        

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
    
    def get_class_embedding(self,img):
        index=torch.arange(21)[None,:].repeat(img.shape[0],1).to(img.device)
        class_embedding=self.relu(self.class_embed(index))
        return class_embedding

    def denorm(self, image):
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, MEAN, STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0, 1, 2), MEAN, STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image

    def run_pamr(self, im, mask):
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)
        masks_dec = self._aff(im, mask)
        return masks_dec

    def forward(self, x,require_seg=False):
        # y_raw=self.denorm(x)
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        x_seg=d['conv6'].clone()
        fea = self.dropout7(d['conv6'])
        n1,d1,h1,w1=x_seg.size()
        #shallow_fea=d['conv3'].clone()
        #print(shallow_fea.size())
        #x_seg=self.aspp(x_seg)

        # shallow_fea=F.interpolate(d['conv3'], size=(h1,w1), mode='bilinear', align_corners=True)
        # shallow_att=torch.mean(shallow_fea,dim=1).unsqueeze(1)
        # #fea=self.fc8(torch.cat([fea*shallow_att,shallow_fea],dim=1))
        # fea=fea+shallow_att*fea

        f_proj1=self.proj1(fea)
        f_proj1=F.relu(f_proj1, inplace=True)
        cam1=self.classifier1(f_proj1)
        

        cam_rv_down=self.FCM(cam1,d,x)
        cam_rv = F.interpolate(cam_rv_down, (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam1, (H, W), mode='bilinear', align_corners=True)
        if require_seg==True:
            fea1=self.fc8_seg_conv1(x_seg)
            fea1=self.fc8_seg_conv2(fea1)
            fea2=self.fc8_seg_conv3(x_seg)
            
            deep_feature = self.fc8_seg_conv4(torch.cat([fea1,fea2],dim=1))
            deep_feature=F.interpolate(deep_feature, size=d['conv3'].size()[2:], mode='bilinear', align_corners=True)
            #att=torch.mean(d['conv3'],dim=1).unsqueeze(1)
            #deep_feature=deep_feature+deep_feature*att
            shallow_feature=self.fc8_seg_conv5(d['conv3'])
            new_feature=torch.cat([deep_feature,shallow_feature],dim=1)
            
            # att=torch.mean(d['conv3'],dim=1).unsqueeze(1)
            # new_feature=new_feature+new_feature*att


            

            
            f_proj2 = self.proj2(new_feature)
            f_proj2=F.relu(f_proj2, inplace=True)
            seg_mask=self.classifier2(f_proj2)

            return cam,cam_rv,f_proj1,cam_rv_down,seg_mask
            #return cam,f_proj1,seg_mask
        else:
            return cam,cam_rv,f_proj1,cam_rv_down
            #return cam,f_proj1


        

       
    def FCM(self,cam,d,x):
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())#*labels.unsqueeze(-1).unsqueeze(-1)
            #cam_d=cam.detach()
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)+1e-5
            # max norm
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1-torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
    

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        f8_5 = F.relu(self.f8_5(d['conv6'].detach()), inplace=True)
        # f8_3 = F.relu(self.f8_3(d['conv5'].detach()), inplace=True)
        # f8_4 = F.relu(self.f8_4(d['conv6'].detach()), inplace=True)
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        #f = torch.cat([x_s,f8_3, f8_4], dim=1) #f8_4
        #f=self.f9(f)
        #cam_rv_down = self.PCM(cam_d_norm,f_proj)
        f1=self.f9_1(torch.cat([f8_4,f8_5],dim=1))
        f2=self.f9_2(torch.cat([x_s,f8_3],dim=1))
        # #n, c, h, w = f.size()

        cam_rv1_down = self.PCM(cam_d_norm,f1)
        cam_rv2_down=self.PCM(cam_d_norm,f2)
        cam_rv_down=(cam_rv1_down+cam_rv2_down)/2
        return cam_rv_down

    def PCM(self, cam, f):

        n,c,h,w = f.size()
        #f_mix=f.clone()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        #f = self.f9(f)
        f = f.view(n, -1, h*w)
        # # norm
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        #aff=F.softmax(torch.matmul(f.transpose(1, 2), f), dim=1)
        aff = aff/(torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv
    # def PCM(self, cam, f1,f2):

    #     n,c,h,w = f1.size()
    #     cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
    #     f1 = self.f9_1(f1)
    #     f2 = self.f9_2(f2)
    #     f1 = f1.view(n, -1, h*w)
    #     f2 = f2.view(n, -1, h*w)
    #     # norm
    #     f1 = f1 / (torch.norm(f1, dim=1, keepdim=True) + 1e-5)
    #     f2 = f2 / (torch.norm(f2, dim=1, keepdim=True) + 1e-5)

    #     aff1 = F.relu(torch.matmul(f1.transpose(1, 2), f1), inplace=True)
    #     aff1 = aff1/(torch.sum(aff1, dim=1, keepdim=True) + 1e-5)
    #     cam_rv1 = torch.matmul(cam, aff1)#.view(n, -1, h, w)
    #     cam_rv1=self.normal(cam_rv1)

    #     aff2 = F.relu(torch.matmul(f2.transpose(1, 2), f2), inplace=True)
    #     aff2 = aff2/(torch.sum(aff2, dim=1, keepdim=True) + 1e-5)

    #     cam_rv=torch.matmul(cam_rv1, aff2).view(n, -1, h, w)
    #     return cam_rv
    

    def normal(self,cam):
        n,c,_=cam.size()
        cam_d = F.relu(cam)
        cam_d_max = torch.max(cam_d, dim=-1)[0].view(n, c, 1)+1e-5
        # max norm
        cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
        cam_d_norm[:, 0, :] = 1-torch.max(cam_d_norm[:, 1:, :], dim=1)[0]
        cam_max = torch.max(cam_d_norm[:,1:,:], dim=1, keepdim=True)[0]
        cam_d_norm[:,1:,:][cam_d_norm[:,1:,:] < cam_max] = 0
        return cam_d_norm

    def _rescale_and_clean(self, masks, image, labels):
        """Rescale to fit the image size and remove any masks
        of labels that are not present"""
        masks = F.interpolate(masks, size=image.size()[-2:], mode='bilinear', align_corners=True)
        masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
        return masks

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm) or isinstance(m, nn.BatchNorm2d)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



