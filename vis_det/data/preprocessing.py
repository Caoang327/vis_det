from ..data import Vis_DatasetMapper
from detectron2.structures import Instances


__all__ = ["data_preprocessing", "spliting_gt_instances"]


def query_image(image_name, dataset):
    """
    Get the image and ground-truth layouts based on the image name.

    Args:
    -- image_name: the name of the image.

    Returns:
    -- img: D2 format image, including gt_layouts and images
    """
    target_image_name = "datasets/coco/val2017/" + image_name + ".jpg"
    Find_Flag = False
    for d in dataset:
        if d['file_name'] == target_image_name:
            img = d
            Find_Flag = True
        else:
            continue

    if not Find_Flag:
        raise NotImplementedError("Not find this image.")

    return img


def data_preprocessing(cfg, input_dict, model):
    """
    Data preprocessing for images with annotations.

    Args:
    -- cfg: cfg for the dataset.
    -- input: a dict contains the img and annotations from the dataset.
    -- model: model

    Returns:
    -- images: ImageList object.
    -- gt_instances: the ground truth layout information.
    -- mapper_target: outputs of Vis_DatasetMapper
    """
    mapper = Vis_DatasetMapper(cfg, False, True)
    batched_input = mapper.__call__(input_dict)
    gt_instances = [x["instances"].to(model.device) for x in [batched_input]]
    images = model.preprocess_image([batched_input])
    return images, gt_instances, [batched_input]


def spliting_gt_instances(gt_instances):
    """
    spliting the groundtruth instances in the gt_instances.

    Args:
    -- gt_instances: the ground truth instances.

    Returns:
    -- splited_instances: A list, each element of which contains the gt for each instance.
    """

    boxes_dt = gt_instances[0].get_fields()
    boxes = boxes_dt['gt_boxes']
    targets = boxes_dt['gt_classes']
    if "gt_masks" in boxes_dt:
        masks = boxes_dt["gt_masks"]
    splited_instances = []
    for i in range(len(boxes)):
        temp_instances = Instances(gt_instances[0].image_size)
        if "gt_masks" in boxes_dt:
            temp_instances._fields = {'gt_boxes': boxes[i], 'gt_classes': targets[i:i+1], 'gt_masks': masks[i]}
        else:
            temp_instances._fields = {'gt_boxes': boxes[i], 'gt_classes': targets[i:i+1]}
        splited_instances.append(temp_instances)
    return splited_instances
