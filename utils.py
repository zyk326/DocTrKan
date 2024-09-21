import os
import numpy as np
import math
import cv2
from config import Configs

cfg = Configs().parse() 
SPLITSIZE  = cfg.split_size

def imvisualize(imdeg, imgt, impred, ind, epoch='0',setting=''):
    """
    Visualize the predicted images along with the degraded and clean gt ones

    Args:
        imdeg (tensor): degraded image
        imgt (tensor): gt clean image
        impred (tensor): prediced cleaned image
        ind (str): index of images (name)
        epoch (str): current epoch
        setting (str): experiment name
    """
    # unnormalize data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    imdeg = imdeg.numpy()
    imgt = imgt.numpy()
    impred = impred.numpy()
    imdeg = np.transpose(imdeg, (1, 2, 0))
    imgt = np.transpose(imgt, (1, 2, 0))
    impred = np.transpose(impred, (1, 2, 0))
    for ch in range(3):
        imdeg[:,:,ch] = (imdeg[:,:,ch] *std[ch]) + mean[ch]
        imgt[:,:,ch] = (imgt[:,:,ch] *std[ch]) + mean[ch]
        impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]

    # avoid taking values of pixels outside [0, 1]
    impred[np.where(impred>1)] = 1
    impred[np.where(impred<0)] = 0

    # create vis folder
    if not os.path.exists('vis'+setting+'/epoch'+epoch):
        os.makedirs('vis'+setting+'/epoch'+epoch)
    
    # binarize the predicted image taking 0.5 as threshold
    impred = (impred>0.5)*1

    # save images
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_deg.png',imdeg*255)
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_gt.png',imgt*255)
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_pred.png',impred*255)
    
def psnr(img1, img2):
    """
    Count PSNR of two images

    Args:
        img1 (np.array): first image
        img2 (np.array): second image
    Returns:
        p (int): the PSNR value 
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    p = (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
    
    return p

def count_f_measure(predictions, targets, threshold=0.5):
    """
    计算 F-Measure

    Args:
        predictions (torch.Tensor): 模型预测图像
        targets (torch.Tensor): 真实图像
        threshold (float): 二值化阈值

    Returns:
        float: F-Measure 值
    """
    predictions = (predictions > threshold).astype(float)
    targets = (targets > threshold).astype(float)

    TP = (predictions * targets).sum()
    FP = (predictions * (1 - targets)).sum()
    FN = ((1 - predictions) * targets).sum()

    precision = TP / (TP + FP + 1e-6)  # 加小常数避免除零
    recall = TP / (TP + FN + 1e-6)

    if precision + recall == 0:
        return 0.0  # 避免除零

    f_measure = 2 * (precision * recall) / (precision + recall)
    return f_measure

def pseudo_f_measure(predictions, targets):
    """
    计算伪 F-Measure

    Args:
        predictions (torch.Tensor): 模型预测图像
        targets (torch.Tensor): 真实图像

    Returns:
        float: 伪 F-Measure 值
    """
    # 假设与 F-Measure 相似，只是使用不同的计算方法或权重
    # 这里用 F-Measure 示范，你可以根据需要修改

    return count_f_measure(predictions, targets, threshold=0.6)  # 伪F-Measure使用不同的阈值

def count_drd(predictions, targets):
    """
    计算距离倒数失真（DRD）

    Args:
        predictions (np.ndarray): 模型预测图像
        targets (np.ndarray): 真实图像

    Returns:
        float: 距离倒数失真值
    """
    # 确保 predictions 和 targets 是 NumPy 数组
    predictions_flat = predictions.reshape(predictions.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    # 计算 L2 范数
    norm_diff = np.linalg.norm(predictions_flat - targets_flat, ord=2, axis=1)

    mean_drd = norm_diff.mean()  # 求每对图像的平均距离


    return (mean_drd)  # 加小常数避免除零


def reconstruct(idx, h, w, epoch, setting, flipped=False):
    """
    reconstruct DIBCO (or other) full images from the binarized patches

    Args:
        idx (str): name of the image
        h (int): height of original image to be constructed from patches
        w (int): width of original image to be constructed from patches
        epoch (int): current epoch
        setting (str): experiment name
        flipped (bool): if the images are flipped, reconstruct and flip
    Returns:
        rec_image (np.array): the reconstruted image 

    """
    # initialize image
    rec_image = np.zeros(((h//SPLITSIZE + 1)*SPLITSIZE,(w//SPLITSIZE + 1)*SPLITSIZE,3))
    
    # fill the image 
    for i in range (0,h,SPLITSIZE):
        for j in range(0,w,SPLITSIZE):
            p = cv2.imread('vis'+setting+'/epoch'+str(epoch)+'/'+idx+'_'+str(i)+'_'+str(j)+'_pred.png')
            if flipped:
                p = cv2.rotate(p, cv2.ROTATE_180)
            rec_image[i:i+SPLITSIZE,j:j+SPLITSIZE,:] = p
    
    # trim the image from padding
    rec_image =  rec_image[:h,:w,:]
    
    return rec_image



def count_psnr(epoch, data_path, valid_data='2018',setting='',flipped = False , thresh = 0.5):
    """
    reconstruct images and count the PSNR for the full validation dataset

    Args:
        epoch (int): current epoch
        data_path (str): path of the data folder
        valid_data (str): which validation dataset
        setting (str): experiment name
        flipped (bool): whether the images are flipped
        thresh (int): binarization threshold after cleaning
    Returns: 
        avg_psnr (float): the PSNR result of the full dataset image pairs
    """
    total_psnr = 0
    total_FM = 0
    total_Fps = 0
    total_DRD = 0

    qo = 0
    
    gt_folder = data_path + 'DIBCOSETS/' + valid_data + '/gt_imgs' 
    gt_imgs = os.listdir(gt_folder)
    flip_status = 'flipped' if flipped else 'normal'
    
    if not os.path.exists('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status):
        os.makedirs('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status)

    for im in gt_imgs:
        gt_image = cv2.imread(gt_folder+'/'+im)
        max_p =  np.max(gt_image) # max_p is 1 or 255
        gt_image = gt_image / max_p
        pred_image = reconstruct(im.split('.')[0],gt_image.shape[0],gt_image.shape[1],epoch,setting, flipped = flipped)/ max_p
        pred_image = (pred_image>thresh)*1

        # 增加了评价指标
        total_psnr+=psnr(pred_image,gt_image)
        total_FM += count_f_measure(pred_image, gt_image)
        total_Fps += pseudo_f_measure(pred_image, gt_image)
        total_DRD += count_drd(pred_image, gt_image)

        qo+=1

        # save reconstructed cleaned image with the gt one.
        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status+'/'+im,gt_image*255)
        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status+'/'+im.split('.')[0]+'_pred.png',pred_image*255)

    avg_psnr = total_psnr/qo
    avg_fm = total_FM / qo
    avg_fps = total_Fps / qo
    avg_drd = total_DRD / qo

    return [avg_psnr, avg_fm, avg_fps, avg_drd]