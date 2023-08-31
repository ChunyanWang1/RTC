
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
#import torchvision.transforms.functional as F
from tool import imutils
from torchvision import transforms


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    #if "test" in dataset_path:
    #     img_name_list=[img_gt_name[-15:-4] for img_gt_name in img_gt_name_list]
    # else:
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    #img_name_list = img_gt_name_list
    return img_name_list



class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,crop_size=None, aug=False, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        
        self.aug=aug

        self.randomresize=imutils.RandomResizeLong(448, 768)
        self.flip=transforms.RandomHorizontalFlip()
        self.colorj=transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1)
        #np.asarray,
        self.normalize=Normalize()
        self.randomcrop=imutils.RandomCrop(crop_size)

        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        
        
        if self.aug:
            img=self.randomresize(img)
            img=self.flip(img)
            img=self.colorj(img)
            #img=np.asarray(img)
            img=self.normalize(img)
            img,img_box=self.randomcrop(img)
            img=imutils.HWC_to_CHW(img)
            img=torch.from_numpy(img)
            return name, img,img_box
        
        else:
            if self.transform:
                img = self.transform(img)
            
            return name, img

        #return name, img


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,crop_size=None,aug=False, transform=None):
        super().__init__(img_name_list_path, voc12_root,crop_size,aug, transform)
        self.aug=aug
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

        


        #self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    # def _pad(self, image):
    #     w, h = image.size

    #     pad_mask = PIL.Image.new("L", image.size)
    #     pad_height = self.pad_size[0] - h
    #     pad_width = self.pad_size[1] - w

    #     assert pad_height >= 0 and pad_width >= 0

    #     pad_l = max(0, pad_width // 2)
    #     pad_r = max(0, pad_width - pad_l)
    #     pad_t = max(0, pad_height // 2)
    #     pad_b = max(0, pad_height - pad_t)

    #     image = F.pad(image, (pad_l, pad_t, pad_r, pad_b), fill=0, padding_mode="constant")
    #     pad_mask = F.pad(pad_mask, (pad_l, pad_t, pad_r, pad_b), fill=1, padding_mode="constant")

    #     return image, pad_mask, [pad_t, pad_l]

    def __getitem__(self, idx):
        if self.aug:
            name, img, img_box = super().__getitem__(idx)
        else:
            name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label



class VOC12ClsDataset1(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,crop_size=None,aug=False, transform=None):
        super().__init__(img_name_list_path, voc12_root,crop_size,aug, transform)
        self.aug=aug
        self.label_list = load_image_label_list_from_npy(self.img_name_list)


    def __getitem__(self, idx):

        label = torch.from_numpy(self.label_list[idx])
        if self.aug:
            name, img, img_box = super().__getitem__(idx)
            return name, img, label,img_box     
        else:
            name, img = super().__getitem__(idx)
            return name, img, label       

class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        #self.pad_size = 1024
        self.inter_transform = inter_transform
        #self.batch_size = len(self.scales)*2


    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())


        return name, msf_img_list, label


class VOC12ClsDatasetMSF_test(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        #self.pad_size = 1024
        self.inter_transform = inter_transform
        #self.batch_size = len(self.scales)*2


    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())


        return name, msf_img_list

      

class VOC12ClsDatasetMS(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        return name, ms_img_list, label


class MultiscaleLoader(VOC12ClsDataset):
    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.pad_size = [1024,1024]
        self.inter_transform = inter_transform
        self.batch_size = len(self.scales)*2
        self.use_flips =True

    # def __init__(self, img_list, cfg, transform):
    #     super().__init__(img_list, DATA_ROOT)

    #     self.scales = cfg.SCALES
    #     self.pad_size = cfg.PAD_SIZE
    #     self.use_flips = cfg.FLIP
    #     self.transform = transform

    #     self.batch_size = len(self.scales)
    #     if self.use_flips:
    #         self.batch_size *= 2

    #     print("Inference batch size: ", self.batch_size)
    #     assert self.batch_size == cfg.BATCH_SIZE

    def __getitem__(self, idx):
        im_idx = idx // self.batch_size
        sub_idx = idx % self.batch_size

        scale = self.scales[sub_idx // (2 if self.use_flips else 1)]
        flip = self.use_flips and sub_idx % 2

        name, img, label = super().__getitem__(im_idx)
        #name, img = super().__getitem__(im_idx)

        target_size = (int(round(img.size[0]*scale)),
                       int(round(img.size[1]*scale)))

        s_img = img.resize(target_size, resample=PIL.Image.CUBIC)


        if flip:
            s_img = F.hflip(s_img)

        w, h = s_img.size
        im_msc, ignore, pads_tl = self._pad(s_img)
        pad_t, pad_l = pads_tl

        im_msc = self.inter_transform(im_msc)
        img = F.to_tensor(self.inter_transform(img))

        pads = torch.Tensor([pad_t, pad_l, h, w])

        ignore = np.array(ignore).astype(im_msc.dtype)[..., np.newaxis]
        im_msc = F.to_tensor(im_msc * (1 - ignore))

        return name, img, im_msc, pads, label





class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)

class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        # label_la = np.load(label_la_path, allow_pickle=True).item()
        # label_ha = np.load(label_ha_path, allow_pickle=True).item()
        # label = np.array(list(label_la.values()) + list(label_ha.values()))
        # TODO: 如果保存的是dict就用上面三行，如果是array就用下面三行
        label_la = np.load(label_la_path, allow_pickle=True)
        label_ha = np.load(label_ha_path, allow_pickle=True)
        label = np.array(list(label_la) + list(label_ha))

        label = np.transpose(label, (1, 2, 0))
        #print(label.shape)
        #print(img.shape)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                #print(img_label.shape)
                img_label = joint_transform(img_label)
                
                
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255  # la预测的背景设置为255
        label[label_ha == 0] = 0    # ha预测的背景设置为背景, 由于ha的背景阈值更低，所以背景pixel数量更少 相当于置信度高的背景
        label[no_score_region] = 255  # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label

class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        label = self.extract_aff_lab_func(label)

        return img, label
