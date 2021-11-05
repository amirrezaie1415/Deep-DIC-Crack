"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

In this script, a number of metrics useful to evaluate the performance of a model are defined.
"""

# import necessary modules
import torch

def get_accuracy(SR, GT, threshold=0.5):
    """
    Compute the accuracy.
    INPUT:
    SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    threshold  = threshold on the probability

     OUTPUT:
     Accuracy = (true positive + true negative) / (true positive + true negative + false positive + false negative)
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    """
    Compute the sensitivity (recall).
    INPUT:
    SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    threshold  = threshold on the probability

    OUTPUT:
    Sensitivity (Recall) = (true positive) / (true positive + false negative)
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1) + (GT == 1)) == 2
    FN = ((SR == 0) + (GT == 1)) == 2
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
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
    TN = ((SR == 0) + (GT == 0)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
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
    TP = ((SR == 1) + (GT == 1)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
    return PC



def get_JS(SR, GT, threshold=0.5):
    """
    Compute the Jaccard similarity.
    INPUT:
    SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    threshold  = threshold on the probability

    OUTPUT:
    Jaccard similarity = (intersection of GT and SR) / (union of GT and SR)
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    Inter = torch.sum((SR + GT) == 2)
    union = torch.sum((SR + GT) >= 1)
    JS = float(Inter) / (float(union) + 1e-6)  # JS = Jaccard similarity.
    return JS


def get_DC(SR, GT, threshold=0.5):
    """
    Compute the Dice Coefficient.
    INPUT:
    SR         = segmentation results (prediction) of the shape (# batches, # channels, height of image, width of image)
    GT         = ground truth of the shape (# batches, # channels, height of image, width of image)
    threshold  = threshold on the probability

    OUTPUT:
    Dice Coefficient = (2 * intersection of GT and SR) / (GT + SR)
    """

    SR = SR > threshold
    GT = GT == torch.max(GT)
    inter = torch.sum((SR + GT) == 2)
    DC = float(2 * inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)  # DC : Dice Coefficient
    return DC
