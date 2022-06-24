"""
Segmentation metrics calculation script
for CATARACTS Segmentation Challenge 2020
"""
import numpy as np


class ConfusionMatrix:
    """
    Class that calculates the confusion matrix.
    It keeps track of computed confusion matrix
    until it has been reseted.
    The ignore label should always be >= num_classes
    :param num_classes: [int] Number of classes
    :param: confusion_matrix: 2D ndarray of confusion_matrix
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_confusion_matrix(self):
        """
        Returns confusion matrix
        :return: confusion_matrix: 2D ndarray of confusion_matrix
        """
        return self.confusion_matrix

    def update_confusion_matrix(self, gt_mask, pre_mask):
        """
        Calculates the confusion matrix for a given ground truth
        and predicted segmentation mask and updates it
        :param gt_mask: 2D ndarray of ground truth segmentation mask
        :param pre_mask: 2D ndarray of predicted segmentation mask
        :return: confusion_matrix: 2D ndarray of confusion_matrix
        """
        assert gt_mask.shape == pre_mask.shape, f" {gt_mask.shape} == {pre_mask.shape}"

        gt_mask_mask = (gt_mask >= 0) & (gt_mask < self.num_classes)
        pre_mask_mask = (pre_mask >= 0) & (pre_mask < self.num_classes)
        mask_final = np.logical_and(gt_mask_mask,pre_mask_mask)
        label = self.num_classes * gt_mask[mask_final].astype("int") + pre_mask[mask_final].astype("int")
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
        return self.confusion_matrix


def pixel_accuracy(confusion_matrix):
    """
    Calculates mean intersection over union given
    the confusion matrix of ground truth and predicted
    segmentation masks
    :param confusion_matrix: 2D ndarray of confusion_matrix
    :return: acc: [float] pixel accuracy
    """
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return acc


def pixel_accuracy_class(confusion_matrix):
    """
    Calculates pixel accuracy per class given
    the confusion matrix of ground truth and predicted
    segmentation masks
    :param confusion_matrix: 2D ndarray of confusion_matrix
    :return: acc: [float] mean pixel accuracy per class
    """
    acc_c = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    acc = np.nanmean(acc_c)
    return acc,acc_c


def mean_intersection_over_union(confusion_matrix):
    """
    Calculates mean intersection over union given
    the confusion matrix of ground truth and predicted
    segmentation masks
    :param confusion_matrix: 2D ndarray of confusion_matrix
    :return: miou: [float] mean intersection over union
    """
    miou_c = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    )
    miou = np.nanmean(miou_c)
    return miou,miou_c


def per_class_intersection_over_union(confusion_matrix):
    """
    Calculates mean intersection over union given
    the confusion matrix of ground truth and predicted
    segmentation masks
    :param confusion_matrix: 2D ndarray of confusion_matrix
    :return: miou: [float] mean intersection over union
    """
    iou = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    )
    return iou


def segmentation_metrics(gt_masks, pred_masks, num_classes):
    """
    Calculates segmentation metrics (pixel accuracy, pixel accuracy per class,
    and mean intersection over union) for a list of ground truth and predicted
    segmentation masks for a given number of classes
    :param gt_masks: [list] 2D ndarray of ground truth segmentation masks
    :param pred_masks: [list] 2D ndarray of predicted segmentation masks
    :param num_classes: [int] Number of classes
    :return: pa, pac, miou [float, float, float]: metrics
    """
    assert len(gt_masks) == len(pred_masks)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        confusion_matrix.update_confusion_matrix(gt_mask, pred_mask)

    cm = confusion_matrix.get_confusion_matrix()
    pa = pixel_accuracy(cm)
    pac, pac_c = pixel_accuracy_class(cm)
    miou, miou_c = mean_intersection_over_union(cm)
    return pa, pac, pac_c, miou, miou_c


def iou_per_class_metrics(gt_masks, pred_masks, num_classes):
    """
    Calculates segmentation metrics (pixel accuracy, pixel accuracy per class,
    and mean intersection over union) for a list of ground truth and predicted
    segmentation masks for a given number of classes
    :param gt_masks: [list] 2D ndarray of ground truth segmentation masks
    :param pred_masks: [list] 2D ndarray of predicted segmentation masks
    :param num_classes: [int] Number of classes
    :return: pa, pac, miou [float, float, float]: metrics
    """
    assert len(gt_masks) == len(pred_masks)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        confusion_matrix.update_confusion_matrix(gt_mask, pred_mask)

    cm = confusion_matrix.get_confusion_matrix()
    miou = per_class_intersection_over_union(cm)
    return miou


def segmentation_metrics_task1(gt_masks, pred_masks):
    """
    Calculates segmentation metrics (pixel accuracy, pixel accuracy per class,
    and mean intersection over union) for a list of ground truth and predicted
    segmentation masks for Task 1
    :param gt_masks: [list] 2D ndarray of ground truth segmentation masks
    :param pred_masks: [list] 2D ndarray of predicted segmentation masks
    :return: pa, pac and miou: [float, float, float] metrics
    """
    pa, pac, miou = segmentation_metrics(gt_masks=gt_masks,
                                         pred_masks=pred_masks,
                                         num_classes=8)
    return pa, pac, miou


def segmentation_metrics_task2(gt_masks, pred_masks):
    """
    Calculates segmentation metrics (pixel accuracy, pixel accuracy per class,
    and mean intersection over union) for a list of ground truth and predicted
    segmentation masks for Task 2
    :param gt_masks: [list] 2D ndarray of ground truth segmentation masks
    :param pred_masks: [list] 2D ndarray of predicted segmentation masks
    :return: pa, pac and miou: [float, float, float] metrics
    """
    pa, pac, miou = segmentation_metrics(gt_masks=gt_masks,
                                         pred_masks=pred_masks,
                                         num_classes=17)
    return pa, pac, miou


def segmentation_metrics_task3(gt_masks, pred_masks):
    """
    Calculates segmentation metrics (pixel accuracy, pixel accuracy per class,
    and mean intersection over union) for a list of ground truth and predicted
    segmentation masks for Task 3
    :param gt_masks: [list] 2D ndarray of ground truth segmentation masks
    :param pred_masks: [list] 2D ndarray of predicted segmentation masks
    :return: pa, pac and miou: [float, float, float] metrics
    """
    pa, pac, miou = segmentation_metrics(gt_masks=gt_masks,
                                         pred_masks=pred_masks,
                                         num_classes=25)
    return pa, pac, miou


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, help="Metrics for task [1-3]", default=1)
    args = parser.parse_args()

    assert args.task in (1, 2, 3), f"{args.task} not a valid task"

    if args.task == 1:
        # Generating sample ground truth and predicted masks for Task 1 (8 classes)
        gt = []
        prediction = []
        for i in range(10):
            gt.append(np.random.randint(8, size=(224, 224), dtype=np.uint8))
            prediction.append(np.random.randint(8, size=(224, 224), dtype=np.uint8))
        # Example use for Task 1
        metrics = segmentation_metrics_task1(gt, prediction)

    if args.task == 2:
        # Generating sample ground truth and predicted masks for Task 2 (17 classes + ignore label)
        gt = []
        prediction = []
        for i in range(10):
            gt_i = np.random.randint(17, size=(224, 224), dtype=np.uint8)
            prediction_i = np.random.randint(17, size=(224, 224), dtype=np.uint8)
            gt_i[200:224, 0:50] = 255
            gt.append(gt_i)
            prediction.append(prediction_i)

        # Example use for Task 2
        metrics = segmentation_metrics_task2(gt, prediction)

    if args.task == 3:
        # Generating sample ground truth and predicted masks for Task 3 (25 classes + ignore label)
        gt = []
        prediction = []
        for i in range(10):
            gt_i = np.random.randint(25, size=(224, 224), dtype=np.uint8)
            prediction_i = np.random.randint(25, size=(224, 224), dtype=np.uint8)
            gt_i[200:224, 0:50] = 255
            gt.append(gt_i)
            prediction.append(prediction_i)

        # Example use for Task 3
        metrics = segmentation_metrics_task3(gt, prediction)
