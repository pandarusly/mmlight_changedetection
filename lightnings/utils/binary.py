import numpy as np
# ------------------      metrics
from torch import Tensor
# from torchmetrics import MetricCollection
# from torchmetrics.classification import (  # type: ignore[attr-defined]
#     MulticlassAccuracy, MulticlassJaccardIndex)

# {
#                 "mprecision": MulticlassPrecision(
#                     num_classes=num_classes,
#                     ignore_index=ignore_index,
#                     average="micro"
#                 ),
#                 "mrecall": MulticlassRecall(
#                     num_classes=num_classes,
#                     ignore_index=ignore_index,
#                     average="micro"
#                 ),
#                 "mF1_1": MulticlassFBetaScore(
#                     num_classes=num_classes,
#                     ignore_index=ignore_index,
#                     beta=1.0,
#                     average="micro"
#                 ),
# }
# MulticlassFBetaScore,
# MulticlassPrecision,
# MulticlassRecall,
# BinaryAccuracy,
# BinaryJaccardIndex,
# BinaryFBetaScore,
# BinaryRecall,
# BinaryPrecision

def build_metric(num_classes=2,ignore_index=255,metrics='mIoU'):
    assert num_classes > 1 ,"num_classes should > 2, but found num_classes = {}".format(num_classes)
 
    # basic_dict = {
    #     "aAcc": MulticlassAccuracy(
    #         num_classes=num_classes,
    #         ignore_index=ignore_index,
    #         average="macro",
    #     ),
    #     "mAcc": MulticlassAccuracy(
    #         num_classes=num_classes,
    #         ignore_index=ignore_index,
    #         average="micro", 
    #     )
    # }

    # update_dict = {
    #     'mIoU': MulticlassJaccardIndex(
    #         num_classes=num_classes,
    #         ignore_index=ignore_index,
    #         average='none'
    #     )
    # }
    # basic_dict.update(
    #     update_dict
    # )
    # train_metrics = MetricCollection(
    #     basic_dict
    # )
    # val_metrics = train_metrics.clone()
    # test_metrics = train_metrics.clone()

    train_metrics = ConfuseMatrixMeter(num_classes)
    val_metrics = ConfuseMatrixMeter(num_classes)
    test_metrics = ConfuseMatrixMeter(num_classes)
    return train_metrics, val_metrics, test_metrics

 

# ------------------      metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def compute(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def reset(self):
        self.initialized = False


# ------------------    metrics
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class,metric=["mIoU","mFscore"]):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        self.metric = metric

    def __call__(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        if isinstance(pr, Tensor):
            pr = pr.detach().cpu().int().numpy()
            gt = gt.detach().cpu().numpy()
        
        val = get_confuse_matrix(
            num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def compute(self):
        scores_dict = cm2score(self.sum,self.metric)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / \
        (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix,metric):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / \
        (recall + precision + np.finfo(np.float32).eps)
    # mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    # mean_iu = np.nanmean(iu)

    # freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    # --- 全部
    # cls_iou = dict(zip(['iou_' + str(i) for i in range(n_class)], iu))
    #
    # cls_precision = dict(zip(['precision_' + str(i) for i in range(n_class)], precision))
    # cls_recall = dict(zip(['recall_' + str(i) for i in range(n_class)], recall))
    # cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))
    #
    # score_dict = {'acc': acc, 'miou': mean_iu, 'mf1': mean_F1}
    # score_dict.update(cls_iou)
    # score_dict.update(cls_F1)
    # score_dict.update(cls_precision)
    # score_dict.update(cls_recall)

    # --- 1


    score_dict = {"acc": acc}
    # cls_iou = dict(iou_1=iu[1])
    # score_dict.update(cls_iou)'
    if  "mIoU" in metric:
        cls_IoU = dict(mIoU=iu)
        score_dict.update(cls_IoU)
  
    if 'mFscore' in  metric :
 
        cls_F1 = dict(mFscore=F1)
        score_dict.update(cls_F1)
    # cls_precision = dict(precision_1=precision[1])
    # cls_recall = dict(recall_1=recall[1])
    # score_dict.update(cls_precision)
    # score_dict.update(cls_recall)

    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""

    def __fast_hist(label_gt, label_pred):
        mask = (label_gt >= 0) & (label_gt < num_classes)

        hist = np.bincount(
            num_classes * label_gt[mask].astype(int) + label_pred[mask],
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict["miou"]
