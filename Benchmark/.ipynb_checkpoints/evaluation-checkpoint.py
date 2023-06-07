import math
import numpy as np
from sklearn.metrics import confusion_matrix

class SegmentationEvaluator:
    def __init__(self, n_classes=19):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        
    def update(self, targets, predictions, weight=1.):
        predictions = np.argmax(predictions, axis=1)
        self.confusion_matrix += weight*confusion_matrix(targets.flatten(), predictions.flatten(), labels=np.arange(self.n_classes))
        
    def compute_balanced_matrix(self):
        return self.confusion_matrix/np.expand_dims(self.confusion_matrix.sum(axis=1), 1).repeat(self.n_classes, 1)
        
    def compute_accuracy(self, balanced=False):
        
        mat = self.compute_balanced_matrix() if balanced else self.confusion_matrix
        
        return mat.diagonal().sum()/mat.sum()
    
    def compute_iou_c(self, cls, balanced=False):
        
        mat = self.compute_balanced_matrix() if balanced else self.confusion_matrix
        
        return mat[cls, cls] / (mat[cls, :].sum() + mat[:, cls].sum() - mat[cls, cls])
    
    def compute_miou(self, balanced=False):
        
        miou = 0.
        remove = 0.
        for i in range(self.n_classes):
            tmp_miou = self.compute_iou_c(i, balanced)
            if not math.isnan(tmp_miou):
                miou += tmp_miou
            else:
                remove += 1
        
        if self.n_classes == remove:
            return math.nan
        
        return miou/(self.n_classes-remove)

def write_segmentation_performance(path, evaluator_now, evaluator_before, evaluator_after, init = False):

    with open(path,"a") as f:
        if init:
            f.write("accuracy;miou;accuracy_bwt;miou_fwt;accuracy_fwt;miou_fwt")
            for i in range(evaluator_now.n_classes):
                f.write(f';iou_{i}')
        else:
            f.write(
                str(evaluator_now.compute_accuracy())    + ";" + 
                str(evaluator_now.compute_miou())        + ";" + 
                str(evaluator_before.compute_accuracy()) + ";" +
                str(evaluator_before.compute_miou())     + ";" +
                str(evaluator_after.compute_accuracy())  + ";" +
                str(evaluator_after.compute_miou())
            )
            
            for i in range(evaluator_now.n_classes):
                iou = evaluator_now.compute_iou_c(i)
                if math.isnan(iou):
                    f.write(";")
                else:
                    f.write(";" + str(iou))
        f.write("\n")
