"""
## NOTICE ##
THIS CODE IS INSPIRED BY THE IMPLEMENTATIONS PROVIDED IN:
[FOR DICE LOSS]: https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
"""


class DiceLoss:
    def __call__(self, SR_flat, GT_flat):
        # SR_flat = the flatten probabilities (prediction)
        # GT_flat = the flatten ground truth
        smooth =  1
        intersection = (SR_flat * GT_flat).sum(-1)
        return 1 - ((2.0 * intersection + smooth) / (SR_flat.sum(-1) + GT_flat.sum(-1) + smooth)).mean()

