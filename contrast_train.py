import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from network.myTool0 import compute_seg_label,compute_seg_loss

#from utils.losses import DenseEnergyLoss, get_energy_loss
#from network.tree_loss import TreeEnergyLoss
from network.VARM import VARM
from sklearn.metrics import average_precision_score
from network.losses import refine_cams_with_bkg_v2,refine_cams_with_bkg_v1

# from chainercv.datasets import VOCSemanticSegmentationDataset
# from chainercv.evaluations import calc_semantic_segmentation_confusion
# from tool import metric
# from thop import profile
# from thop import clever_format 

CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def compute_class_relation(f_proj,prototypes):
    scores=F.softmax(torch.matmul(f_proj, prototypes.transpose(0, 1)),dim=-1)
    metrix_scores=torch.matmul(scores.transpose(0,1),scores)
    metrix_scores-=torch.diag_embed(torch.diag(metrix_scores))
    loss=0.1* torch.mean(torch.sum(metrix_scores,dim=-1))
    return loss

def normal(cam_rv_down,label,bg_threshold):
    cam_rv_down = F.relu(cam_rv_down)#(b,21,hw) .detach()
    # # ~(0,1)
    n1, c1, h1, w1 = cam_rv_down.shape
    max1 = torch.max(cam_rv_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
    min1 = torch.min(cam_rv_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
    cam_rv_down[cam_rv_down < min1 + 1e-5] = 0.
    norm_cam = (cam_rv_down - min1 - 1e-5) / (max1 - min1 + 1e-5)
    cam_rv_down = norm_cam
    cam_rv_down[:, 0, :, :] = bg_threshold
    scores = F.softmax(cam_rv_down * label, dim=1)
    pseudo_label = scores.argmax(dim=1, keepdim=True)
    return cam_rv_down,pseudo_label

def  pseudo_gtmask(mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8): #0.2
    """Convert continuous mask into binary mask"""
    bs,c,h,w = mask.size()
    mask = mask.view(bs,c,-1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max[:, 0] *= 0.7 #0.7
    mask_max[:, 1:] *= cutoff_top
    #mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
    pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs,c,h,w)

def balanced_mask_loss_ce(mask,mask_gt, pseudo_gt, gt_labels,ignore_index=255):#
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # # indices of the max classes
    # mask_gt = torch.argmax(pseudo_gt, 1)

    # # for each pixel there should be at least one 1
    # # otherwise, ignore
    # ignore_mask = pseudo_gt.sum(1) < 1.
    # mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs,c,h,w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs,c,-1).sum(-1)
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
    class_weight = (pseudo_gt * class_weight[:,:,None,None]).sum(1).view(bs, -1)

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss=loss.view(bs,-1)
    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1 # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)
    loss = batch_weight * (class_weight *loss).mean(-1)#+0.5*class_weight*loss.mean(-1)
    #loss = (class_weight *loss).mean(-1)
    return loss#.mean()


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance,
    # but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)






def validate(model, data_loader):

    # gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir="./VOC2012")
    # labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]

    val_loss_meter = pyutils.AverageMeter('val_loss','val_loss2')
    print('validating ... ', flush=True, end='')

    # class ground truth
    targets_all = []

    # class predictions
    preds_all = []

    model.eval()

    with torch.no_grad():
        preds = []
        for iter, pack in enumerate(data_loader):       
            img = pack[1].cuda()
            label_idx = pack[2]#.cuda(non_blocking=True)
            
            bg_score = torch.ones((1, 1))
            label = torch.cat((bg_score, label_idx), dim=1)
            label=label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)


            cam, cam_rv, f_proj, cam1_rv_down,seg_mask = model(img,require_seg=True)#cam2_rv_down
            
            seg=F.interpolate(seg_mask, size=img.size()[2:], mode='bilinear', align_corners=True)
            cls_score= F.adaptive_avg_pool2d(cam, (1, 1))
            cls_score2= F.adaptive_avg_pool2d(seg, (1, 1))

            cls_sigmoid=torch.sigmoid(cls_score[:,1:].squeeze(-1).squeeze(-1))
            preds_all.append(cls_sigmoid.cpu().numpy())
            targets_all.append(label_idx.numpy())


            loss = F.multilabel_soft_margin_loss(cls_score[:,1:,:,:], label[:,1:,:,:])
            loss2 =F.multilabel_soft_margin_loss(cls_score2[:,1:,:,:], label[:,1:,:,:]) 

            val_loss_meter.add({'val_loss': loss.item(),'val_loss2': loss2.item()}) #
        
    # confusion = calc_semantic_segmentation_confusion(preds, labels)
    # gtj = confusion.sum(axis=1)
    # resj = confusion.sum(axis=0)
    # gtjresj = np.diag(confusion)
    # denominator = gtj + resj - gtjresj
    # iou = gtjresj / denominator
    # print({'iou': iou, 'miou': np.nanmean(iou)})

    model.train()
    print('val_cls_loss:', val_loss_meter.pop('val_loss'))
    print('val_cls_loss2:', val_loss_meter.pop('val_loss2'))

    targets_stacked = np.vstack(targets_all)
    preds_stacked = np.vstack(preds_all)
    aps = average_precision_score(targets_stacked, preds_stacked, average=None)

    ni=0
    for className in CLASSES:
        print("AP_{}: {:4.3f}".format(className, aps[ni]))
        ni=ni+1

    meanAP = np.mean(aps)
    print('mAP: {:4.3f}'.format(meanAP))

    return #np.nanmean(iou)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_contrast", type=str)#resnet38_contrast_revise1_1  resnet38_contrast
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/train.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_cl", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", default='models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--bg_threshold", default=0.20, type=float)
    # parser.add_argument("--saved_dir", default='VOC2012', type=str)

    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    set_seed(1)

    model = getattr(importlib.import_module(args.network), 'Net')()

    # # metric.cal_flops_params(model)
    # # exit()
    # dummy_input=torch.rand(1, 3, 448,448)#.cuda(0)
    # flops,params=profile(model,inputs=(dummy_input,True))
    # flops,params=clever_format([flops,params],'%.3f')
    # print("params:",params)
    # print("flops:", flops)
    # exit()

    tblogger = SummaryWriter(args.tblog_dir)

    train_dataset = voc12.data.VOC12ClsDataset1(args.train_list, voc12_root=args.voc12_root,crop_size=args.crop_size,aug=True,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(448, 768),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                                          saturation=0.3, hue=0.1),
                                                   np.asarray,
                                                   model.normalize,
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))
    
    infer_dataset = voc12.data.VOC12ClsDataset1(args.val_list, voc12_root=args.voc12_root,aug=False,
                                               transform=transforms.Compose(
                                                      [np.asarray,
                                                       model.normalize,
                                                       imutils.HWC_to_CHW]))


    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    infer_data_loader = DataLoader(infer_dataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d

        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    

    varm = VARM(num_iter=10, dilations=[1, 2, 4, 8, 12, 24])
    varm=torch.nn.DataParallel(varm).cuda()

    # tree_loss=TreeEnergyLoss()
    # tree_loss=torch.nn.DataParallel(tree_loss).cuda()

    avg_meter = pyutils.AverageMeter('loss',
                                     'loss_cls',
                                     'loss_er',
                                     'loss_ecr',
                                     'loss_struct',
                                     'loss_nce',
                                     'loss_seg',
                                     'loss_class_relation',
                                     'loss_intra_nce',
                                     'loss_cross_nce1',
                                     'loss_cross_nce2',
                                     'loss_tree'
                                     )

    timer = pyutils.Timer("Session started: ")

   
    MIOU=0.
    for ep in range(args.max_epoches):
        
        for iter, pack in enumerate(train_data_loader):
            img_name=pack[0]
            img1 = pack[1]
            img2 = F.interpolate(img1,
                                 size=(128, 128),
                                 mode='bilinear',
                                 align_corners=True)
            N, C, H, W = img1.size()
            label_idx = pack[2]
            img_box=pack[3]
            

            bg_score = torch.ones((N, 1))
            label = torch.cat((bg_score, label_idx), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
            
            cam1, cam_rv1, f_proj1, cam_rv1_down,seg_mask = model(img1,require_seg=True)#,cam2_rv1_down
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])



            cam1 = F.interpolate(visualization.max_norm(cam1),
                                 size=(128, 128),
                                 mode='bilinear',
                                 align_corners=True) * label
            
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),
                                    size=(128, 128),
                                    mode='bilinear',
                                    align_corners=True) * label

            cam2, cam_rv2, f_proj2, cam_rv2_down = model(img2)#,seg_mask2 
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
            cam2 = visualization.max_norm(cam2) * label
            cam_rv2 = visualization.max_norm(cam_rv2) * label

            #cam2_origin=cam2.clone()
            loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])

            ns, cs, hs, ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))

            cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
            cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]

            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2

            img11=model.module.denorm(img1)
            
           

            if ep>=7:
                
                cam_rv = F.interpolate(visualization.max_norm(cam_rv1_down)*label,#visualization.max_norm(cam_rv1_down)*label
                                    size=(448, 448),
                                    mode='bilinear',
                                    align_corners=True) #* label
                
                # seg_label = refine_cams_with_bkg_v2(varm, img11.cuda(non_blocking=True), cams=cam_rv[:,1:], cls_labels=label_idx.cuda(non_blocking=True),
                #                                       img_box=img_box)

                seg_label = refine_cams_with_bkg_v1(varm, img11.cuda(non_blocking=True), cams=cam_rv[:,1:], cls_labels=label_idx.cuda(non_blocking=True),
                                                      img_box=img_box)

                
                
                seg_label=seg_label.type(torch.long)
                pseudo_gt=torch.zeros_like(cam_rv).type_as(seg_label).permute(1,0,2,3)
                for i in range(21):
                    pseudo_gt[i][seg_label==i]=1

                pseudo_gt=pseudo_gt.permute(1,0,2,3).detach()
                
                loss_seg = balanced_mask_loss_ce(seg_mask, seg_label.detach(),pseudo_gt, label_idx.cuda())
                loss_seg=loss_seg.mean()
            else:
                loss_seg=torch.tensor(0.)
                #loss_tree=torch.tensor(0.)






            ################################################################################
            ###################### Contrastive Learning ####################################
            ################################################################################

            f_proj1 = F.interpolate(f_proj1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
            cam_rv1_down = F.interpolate(cam_rv1_down, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
            
                
            


            with torch.no_grad():
                # # source
                nf,cf,hf,wf=f_proj1.size()
                fea1 = f_proj1.detach()
                fea2 = f_proj2.detach()
                c_fea1 = fea1.shape[1]

                cam_rv1_down,pseudo_label1=normal(cam_rv1_down,label,args.bg_threshold)
                cam_rv2_down,pseudo_label2=normal(cam_rv2_down,label,args.bg_threshold)
                
                
                n_sc1, c_sc1, h_sc1, w_sc1 = cam_rv1_down.shape#scores1.shape
                top_values1, top_indices1 = torch.topk(cam_rv1_down.transpose(0,1).reshape(c_sc1,-1),
                                                        k=h_sc1 * w_sc1 // 8, dim=-1) #[21,topk]
                
                top_values2, top_indices2 = torch.topk(cam_rv2_down.transpose(0,1).reshape(c_sc1,-1),
                                                    k=h_sc1 * w_sc1 // 8, dim=-1) #[21,topk]
                
                
                
                fea1 = fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)
                fea2 = fea2.permute(0, 2, 3, 1).reshape(-1, c_fea1)
                fea2_norm = F.normalize(fea2, dim=-1)
                fea1_norm = F.normalize(fea1,dim=-1)
   
                prototypes1 = torch.zeros(c_sc1,c_fea1).cuda()  # [21, 256]
                prototypes2 = torch.zeros(c_sc1,c_fea1).cuda()  # [21, 256]

                
                new_cam_rv1_down=torch.zeros_like(cam_rv1_down)
                new_cam_rv2_down=torch.zeros_like(cam_rv2_down)
 


                labels_num=torch.sum(label.squeeze(-1).squeeze(-1),dim=0)
                for i in range(c_sc1):
                    top_fea1 = fea2[top_indices1[i]] #[topk,128]
                    prototypes1[i] = torch.sum(top_values1[i].unsqueeze(-1) * top_fea1, dim=0) / torch.sum(top_values1[i])


                    top_fea2 = fea1[top_indices2[i]]
                    prototypes2[i] = torch.sum(top_values2[i].unsqueeze(-1) * top_fea2, dim=0) / torch.sum(top_values2[i])

                    
                    if labels_num[i]==0:
                        new_cam_rv1_down[:,i,:,:]=cam_rv1_down[:,i,:,:]
                        new_cam_rv2_down[:,i,:,:]=cam_rv2_down[:,i,:,:]

                    else:
                        top_fea_norm=F.normalize(top_fea1,dim=-1)
                        fea_att=F.softmax(torch.matmul(fea2_norm,top_fea_norm.transpose(0,1)),dim=1)
                        #fea_att=F.relu(torch.matmul(fea2_norm,top_fea_norm.transpose(0,1)),inplace=True)
                        new_cam1=torch.matmul(fea_att,top_values1[i].unsqueeze(-1))
                        new_cam_rv1_down[:,i,:,:]=cam_rv1_down[:,i,:,:]+new_cam1.view(n_sc1,h_sc1,w_sc1)


                        top_fea2_norm=F.normalize(top_fea2,dim=-1)
                        fea_att=F.softmax(torch.matmul(fea1_norm,top_fea2_norm.transpose(0,1)),dim=1)
                        #fea_att=F.relu(torch.matmul(fea1_norm,top_fea2_norm.transpose(0,1)),inplace=True)
                        new_cam2=torch.matmul(fea_att,top_values2[i].unsqueeze(-1))
                        new_cam_rv2_down[:,i,:,:]=cam_rv2_down[:,i,:,:]+new_cam2.view(n_sc1,h_sc1,w_sc1)


                prototypes1 = F.normalize(prototypes1, dim=-1)
                prototypes2 = F.normalize(prototypes2, dim=-1)



            
            new_cam_rv1= F.interpolate(visualization.max_norm(new_cam_rv1_down),
                                 size=cam1.size()[2:],
                                 mode='bilinear',
                                 align_corners=True) * label
            
            tensor_struct1=torch.abs(max_onehot(new_cam_rv1) - max_onehot(cam1))
            loss_struct1 = torch.mean(torch.topk(tensor_struct1[:, 1:, :, :].view(ns, -1), k=(int)(20 * hs * ws * 0.8), dim=-1)[0])

            
            new_cam_rv2=F.interpolate(visualization.max_norm(new_cam_rv2_down),
                                 size=cam2.size()[2:],
                                 mode='bilinear',
                                 align_corners=True) * label
            
            tensor_struct2=torch.abs(max_onehot(new_cam_rv2) - max_onehot(cam2))
            loss_struct2 = torch.mean(torch.topk(tensor_struct2[:, 1:, :, :].view(ns, -1), k=(int)(20 * hs * ws * 0.8), dim=-1)[0])

            loss_struct=loss_struct1+loss_struct2

            # for source
            n_f, c_f, h_f, w_f = f_proj1.shape
            f_proj1 = f_proj1.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
            f_proj1 = F.normalize(f_proj1, dim=-1)


            pseudo_label1 = pseudo_label1.reshape(-1)
            pseudo_label2 = pseudo_label2.reshape(-1)
            
            positives11=prototypes1[pseudo_label1] 
            positives12=prototypes1[pseudo_label2] 
            negitives1 = prototypes1


            

            # for target
            n_f, c_f, h_f, w_f = f_proj2.shape
            f_proj2 = f_proj2.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
            f_proj2 = F.normalize(f_proj2, dim=-1)

            

            positives21=prototypes2[pseudo_label2]
            positives22=prototypes2[pseudo_label1]
            negitives2 = prototypes2

            


            loss_class_relation=torch.tensor(0.0).cuda()
            labels_gt=torch.sum(label.squeeze(-1).squeeze(-1),dim=0)
            labels_gt[labels_gt!=0]=1
            exists = torch.nonzero(labels_gt).tolist()
            count=0
            for ids_i in exists:
                pt_i=prototypes1[ids_i]
                for ids_j in exists:
                    pt_j=prototypes2[ids_j]
                    if ids_i==ids_j:
                        continue
                    cos_sim=torch.sum(pt_i*pt_j)#torch.cosine_similarity(pt_i,pt_j)
                    loss_class_relation+=-torch.log(1-cos_sim)
                    count+=1
            
            loss_class_relation=0.5*(loss_class_relation/count)

            
            # # 1.  contrastive learning
            # 1.1 intra-view
            A1 = torch.exp(torch.sum(f_proj1 * positives11, dim=-1) / 0.1) 
            A2 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
            loss_intra_nce_1 = torch.mean(-1 * torch.log(A1 / A2))

           
            A3 = torch.exp(torch.sum(f_proj2 * positives21, dim=-1) / 0.1)
            A4 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
            loss_intra_nce_2 = torch.mean(-1 * torch.log(A3 / A4))


            loss_intra_nce = 0.1 * (loss_intra_nce_1 + loss_intra_nce_2) / 2

            # 1.2 cross-pseudo-label
            A1_view1 = torch.exp(torch.sum(f_proj1 * positives12, dim=-1) / 0.1)
            A2_view1 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
            loss_cross_nce1_1 = torch.mean(-1 * torch.log(A1_view1 / A2_view1))


            A3_view1 = torch.exp(torch.sum(f_proj2 * positives22, dim=-1) / 0.1) 
            A4_view1 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
            loss_cross_nce1_2 = torch.mean(-1 * torch.log(A3_view1 / A4_view1))

            loss_cross_nce1 = 0.1 * (loss_cross_nce1_1 + loss_cross_nce1_2) / 2


            # # 1.3 cross-pt cross-pseudo-label
            # A1_view2 = torch.exp(torch.sum(f_proj1 * positives21, dim=-1) / 0.1)
            # A2_view2 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives2.transpose(0, 1)) / 0.1), dim=-1)
            # loss_cross_nce2_1 = torch.mean(-1 * torch.log(A1_view2 / A2_view2))


            # A3_view2 = torch.exp(torch.sum(f_proj2 * positives11, dim=-1) / 0.1) 
            # A4_view2 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives1.transpose(0, 1)) / 0.1), dim=-1)
            # loss_cross_nce2_2 = torch.mean(-1 * torch.log(A3_view2 / A4_view2))

            # loss_cross_nce2 = 0.1 * (loss_cross_nce2_1 + loss_cross_nce2_2) / 2


            # 1.4 cross-pt
            A1_view2 = torch.exp(torch.sum(f_proj1 * positives22, dim=-1) / 0.1)
            A2_view2 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives2.transpose(0, 1)) / 0.1), dim=-1)
            loss_cross_nce2_1 = torch.mean(-1 * torch.log(A1_view2 / A2_view2))


            A3_view2 = torch.exp(torch.sum(f_proj2 * positives12, dim=-1) / 0.1) 
            A4_view2 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives1.transpose(0, 1)) / 0.1), dim=-1)
            loss_cross_nce2_2 = torch.mean(-1 * torch.log(A3_view2 / A4_view2))

            loss_cross_nce2 = 0.1 * (loss_cross_nce2_1 + loss_cross_nce2_2) / 2


            


            # 3. total nce loss
            loss_nce = loss_cross_nce2  + loss_intra_nce + loss_cross_nce1 #+ loss_cross_nce2 

            # 4. total loss
            loss = loss_cls + loss_er + loss_ecr + loss_nce +loss_struct + loss_seg #+loss_struct  #+loss_tree# +loss_class_relation  #



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item(),
                           'loss_struct': loss_struct.item(),
                           'loss_seg': loss_seg.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_intra_nce': loss_intra_nce.item(),
                           'loss_class_relation': loss_class_relation.item(),
                           'loss_cross_nce1': loss_cross_nce1.item(),
                           'loss_cross_nce2': loss_cross_nce2.item()
                           })

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d | ' % (optimizer.global_step - 1, max_step),
                      'loss: %.4f | loss_cls: %.4f | loss_er: %.4f | loss_ecr: %.4f | loss_struct: %.4f | loss_seg: %.4f |  loss_class_relation: %.4f |'
                      'loss_nce: %.4f | loss_intra_nce: %.4f | loss_cross_nce1: %.4f |  loss_cross_nce2: %.4f' #| loss_intra_nce: %.4f   loss_tree: %.4f '
                      % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_struct','loss_seg','loss_class_relation','loss_nce', 'loss_intra_nce',# 
                                      'loss_cross_nce1','loss_cross_nce2'),
                      'imps:%.1f | ' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s | ' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                loss_dict = {'loss': loss.item(),
                             'loss_cls': loss_cls.item(),
                             'loss_er': loss_er.item(),
                             'loss_ecr': loss_ecr.item(),
                             'loss_struct': loss_struct.item(),
                             'loss_seg': loss_seg.item(),
                             'loss_nce': loss_nce.item(),
                             'loss_class_relation': loss_class_relation.item(),
                             'loss_intra_nce': loss_intra_nce.item(),
                             'loss_inter_nce1': loss_cross_nce1.item(),
                             'loss_inter_nce2': loss_cross_nce2.item()
                             }

                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)

        else:
            print('')
            timer.reset_stage()
        
        if ep>6:# and ep%2==0:
                #miou = validate(model, infer_data_loader)
                validate(model, infer_data_loader)
                #torch.save({'net':model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'iter_' + str(optimizer.global_step) + '.pth'))
                #if ep>12:
                torch.save(model.module.state_dict(), os.path.join("save_pth_revise1_1",args.session_name + '_model_'+str(ep)+'.pth'))
                # if miou > bestiou:
                #     bestiou = miou
                #     #torch.save({'net':model.module.state_dict()}, os.path.join("save_pth3",args.session_name, 'ckpt', 'best.pth'))
                #     torch.save(model.module.state_dict(), os.path.join("save_pth3",args.session_name + '_ckpt_best.pth'))    

        # if ep>6 and ep%2==0:
        #     torch.save(model.module.state_dict(), os.path.join("save_pth3_2",args.session_name + '_model_'+str(ep)+'.pth'))
    
    print(args.session_name)
    # print(args.session_name)

    # torch.save(model.module.state_dict(), args.session_name + '.pth')
