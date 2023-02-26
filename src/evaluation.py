"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

In this script, a number of metrics useful to evaluate the performance of a model are defined.
"""

# import necessary modules
import torch


def get_accuracy(SR, GT, threshold=0.5):
    # """
    # Compute the accuracy.
    # INPUT:
    # SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    # GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    # threshold  = threshold on the probability
    #
    #  OUTPUT:
    #  Accuracy = (true positive + true negative) / (true positive + true negative + false positive + false negative)
    # """
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))
    corr = (SR == GT).float().sum(axis=[2, 3])
    image_size = SR.size(2) * SR.size(3)
    batch_acc = corr / image_size
    return batch_acc.squeeze(1)


def get_sensitivity(SR, GT, threshold=0.5, smooth=1e-6):
    # """
    # Compute the sensitivity (recall).
    # INPUT:
    # SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    # GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    # threshold  = threshold on the probability
    #
    # OUTPUT:
    # Sensitivity (Recall) = (true positive) / (true positive + false negative)
    # """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1).float() + (GT == 1).float()) == 2).float()
    FN = (((SR == 0).float() + (GT == 1).float()) == 2).float()
    SE_batch = TP.sum(axis=[2, 3]) / ((TP + FN).sum(axis=[2, 3]) + smooth)
    return SE_batch.squeeze(1)


def get_specificity(SR, GT, threshold=0.5, smooth=1e-6):
    """
    Compute the specificity.
    INPUT:
    SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    threshold  = threshold on the probability

    OUTPUT:
    Specificity = (true negative) / (true negative + false positive)
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0).float() + (GT == 0).float()) == 2).float()
    FP = (((SR == 1).float() + (GT == 0).float()) == 2).float()
    SP_batch = TN.sum(axis=[2, 3]) / ((TN + FP).sum(axis=[2, 3]) + smooth)
    return SP_batch.squeeze(1)


def get_precision(SR, GT, threshold=0.5, smooth=1e-6):
    """
    Compute the precision.
    INPUT:
    SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    threshold  = threshold on the probability

    OUTPUT:
    Precision = (true positive) / (true positive + false positive)
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1).float() + (GT == 1).float()) == 2).float()
    FP = (((SR == 1).float() + (GT == 0).float()) == 2).float()
    PC_batch = TP.sum(axis=[2, 3]) / ((TP + FP).sum(axis=[2, 3]) + smooth)
    return PC_batch.squeeze(1)


def get_JS(SR, GT, threshold=0.5, smooth=1e-6):
    # """
    # Compute the Jaccard similarity.
    # INPUT:
    # SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    # GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    # threshold  = threshold on the probability
    #
    # OUTPUT:
    # Jaccard similarity = (intersection of GT and SR) / (union of GT and SR)
    # """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    intersection = (SR & GT).float().sum((2, 3))
    union = (SR | GT).float().sum((2, 3))
    JS_batch = intersection / (union + smooth)
    return JS_batch.squeeze(1)


def get_DC(SR, GT, threshold=0.5, smooth=1e-6):
    # """
    # Compute the Dice Coefficient.
    # INPUT:
    # SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    # GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    # threshold  = threshold on the probability
    #
    # OUTPUT:
    # Dice Coefficient = (2 * intersection of GT and SR) / (GT + SR)
    # """

    SR = SR > threshold
    GT = GT == torch.max(GT)
    inter = (SR & GT).float().sum((2, 3))
    DC_batch = 2 * inter / (SR.sum(axis=[2, 3]) + GT.sum(axis=[2, 3]) + smooth)  # DC : Dice Coefficient
    return DC_batch.squeeze(1)
