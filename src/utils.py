import glob
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import auc
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize((n_px, n_px), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

def _transform_flow(n_px):
    return Compose(
        [
            #Resize(n_px, interpolation=BICUBIC),
            ToTensor(),
        ]
    )


preprocess = _transform(224)
preprocess_flow = _transform_flow(224)



def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def combine_heatmap_img(img, pred):
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    img = preprocess(img).permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    vis = show_cam_on_image(img, pred)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def cal_CIOU(infer, gtmap, thres=0.5):
    infer_map = np.zeros((224, 224))
    infer_map[infer >= thres] = 1
    ciou = np.sum(infer_map * gtmap) / (
        np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))
    )
    return (
        ciou,
        np.sum(infer_map * gtmap),
        (np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))),
    )


def clean_pred(pred):
    pred = normalize_img(pred)
    threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
    pred[pred < threshold] = 0
    return pred


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def process_heatmap(heatmap):
    heatmap_arr = heatmap.data.cpu().numpy()
    heatmap_now = cv2.resize(
        heatmap_arr[0, 0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR
    )
    heatmap_now = normalize_img(-heatmap_now)
    pred = 1 - heatmap_now
    return pred


def compute_metrics(preds, thres=0.5, at=0.3):
    metrics = {}
    ious = [cal_CIOU(pred, gt_map, thres=thres)[0] for _, pred, gt_map in preds]

    results = []
    for i in range(21):
        result = np.sum(np.array(ious) >= 0.05 * i)
        result = result / len(ious)
        results.append(result)
    x = [0.05 * i for i in range(21)]

    metrics["auc"] = auc(x, results)
    metrics["cIoU"] = np.sum(np.array(ious) >= at) / len(ious)
    return metrics



def parse_annot(path_to_annot):
    """
    parses a single annotation text file 
    returns: a list of bounding boxes i.e. 4-tuples (x, y, w, h) 
    """
    with open(path_to_annot) as f:
        annotations = f.readlines()
    annotations = [i.strip() for i in annotations]                     # remove newlines
    annotations = [i.split(" ") for i in annotations]                  # convert strings to lists 
    annotations = [i[1:] for i in annotations]                         # remove classes
    bboxes = [[float(i) for i in bbox] for bbox in annotations]        # cast to float
    return bboxes 


####################################################
#                      YOLO UTILS                  #
####################################################

inference_dir = "inference_no_filt"
  
def parse_single_yolo_annot(path_to_annot):
    """
    parses a single yolo annotation text file 
    returns: frame number, and a list of 4-tuples (x, y, w, h)
    """

    with open(path_to_annot) as f:
        annotations = f.readlines()
    annotations = [i.strip() for i in annotations]                     # remove newlines
    annotations = [i.split(" ") for i in annotations]                  # convert strings to lists 
    annotations = [i[1:] for i in annotations]                         # remove class labels
    bboxes = [[float(i) for i in bbox] for bbox in annotations]        # cast to float
    frame = int(path_to_annot.split(".")[-2].split("_")[-1])           # get frame number 
    return frame, bboxes 


def parse_yolo_annot(filename, label_dir):
    """
    parses all annotations for a given video file into a dictionary
    """ 
    annot_files = glob.glob(f"{label_dir}/{filename}_*")               
    annotations = dict()                                               
    for path_to_annot in annot_files:                                  
        frame, bboxes = parse_single_yolo_annot(path_to_annot)
        annotations[frame] = bboxes

    return annotations

def parse_urbansas_annot(filename, label_dir):
    """
    parses urbansas annotations
    To remove after fixing the centre problem
    """
    annot_files = glob.glob(f"{label_dir}/{filename}_*")               
    annotations = dict()                                               
    for path_to_annot in annot_files:                                  
        frame, bboxes = parse_single_yolo_annot(path_to_annot)
        for i in range(len(bboxes)):
            bboxes[i][0] += bboxes[i][2]/2
            bboxes[i][1] += bboxes[i][3]/2
            
        annotations[frame] = bboxes

    return annotations


def remove_non_moving(preds1, preds2, iou_thresh):
    """
    removes non-moving bounding boxes 
    bboxes with an iou higher than iou_thresh in subsequent frames would be removed
    arguments:
            1. preds - bounding boxes
            2. iou_thresh (float) - a threshold for iou between bounding boxes
                                    values in [0,1]
    returns:
            1. filtered_preds - filtered version of the preds
    """

    iou_matrix = pairwise_iou(preds1, preds2)
    max_iou = np.max(iou_matrix, axis=1)
    to_keep = [max_iou<iou_thresh][0]
    preds1 = [box for idx, box in enumerate(preds1) if to_keep[idx]]
    return preds1


def bbox_iou(box1, box2, eps = 1e-9):
    """calculates iou (intersection over union) for two boxes (x,y,w,h)
    """
    # transform from xywh to xyxy
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # area of intersection 
    inter = max(0, (min(b1_x2, b2_x2) - max(b1_x1, b2_x1))) * \
            max(0, (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)))

    # area of union
    w1, h1 = box1[2], box1[3] 
    w2, h2 = box2[2], box2[3] 
    union = w1*h1 + w2*h2 - inter + eps

    # intersection over union
    iou = inter/union

    return iou

def pairwise_iou(boxes1, boxes2):
    """calculates pairwise iou between two lists of bounding boxes
    returns a matrix of iou values with dimensions - len(boxes1) x len(boxes2)
    """
    l1, l2 = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((l1, l2))

    for i in range(l1):
        for j in range(l2):
            iou_matrix[i, j] = bbox_iou(boxes1[i], boxes2[j])
    
    return iou_matrix


def get_vision_topline_pred(img_name):
    try:
        frame, preds = parse_single_yolo_annot(f"../{inference_dir}/valid/{img_name}.txt")
    except:
        return None
    
    next_frame = frame+4
    next_img_name = "_".join(img_name.split("_")[:-1]) + "_" + str(next_frame)
    try:
        _, next_frame_preds = parse_single_yolo_annot(f"../{inference_dir}/valid/{next_img_name}.txt")
    except:
        return []
    filtered_preds = remove_non_moving(preds, next_frame_preds, iou_thresh=0.95)
    return filtered_preds

def get_vision_baseline_pred(img_name):
    try:
        frame, preds = parse_single_yolo_annot(f"../{inference_dir}/valid/{img_name}.txt")
    except:
        return None
    return preds


def generate_mask_from_pred(preds):
    mask = np.zeros([224, 224])
            
    for item in preds:
        xc, yc, w, h = item[0]*224, item[1]*224, item[2]*224, item[3]*224
        x1, y1 = int(xc-w/2), int(yc-h/2)
        x2, y2 = int(x1+w), int(y1+h)
        temp = np.zeros([224, 224])
        temp[y1:y2, x1:x2] = 1
        mask += temp
    
    mask[mask>=1] = 1

    return mask