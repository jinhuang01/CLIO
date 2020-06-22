from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import argparse
import numpy as np

from vision.ssd.multi_headed_ssd import MultiHeadedSSD
from vision.ssd.predictor import Predictor
import os
from glob import glob
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--net', default='mb2-ssd-lite', type=str)
parser.add_argument('--trained_model', type=str)
parser.add_argument('--label_file', type=str)
parser.add_argument('--input', type=str)
# parser.add_argument('--output_file', type=str)
args = parser.parse_args()


class_names = [name.strip() for name in open(args.label_file).readlines()]

if args.net == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif args.net == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif args.net == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif args.net == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif args.net == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif args.net == 'multi-headed':
    from vision.ssd.config import mobilenetv1_ssd_config
    net = MultiHeadedSSD(len(class_names), width_mult=1.0, config=mobilenetv1_ssd_config, is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

net.load(args.trained_model)

if args.net == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif args.net == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif args.net == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif args.net == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif args.net == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
elif args.net == 'multi-headed':
    predictor = Predictor(
        net, 
        net.config.image_size, 
        net.config.image_mean,
        net.config.image_std, 
        nms_method=None,
        iou_threshold=net.config.iou_threshold,
        candidate_size=200,
        sigma=0.5
    )
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

if os.path.isdir(args.input):
    filenames = glob(args.input + '/*')
elif os.path.isfile(args.input):
    filenames = [args.input]
else:
    print('cannot read input')


start = time()
for filename in filenames:
    orig_image = cv2.imread(filename)
    # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image = np.stack([image] * 3, axis=-1)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)


    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    # (box[0] + 20, box[1] + 40),
                    (box[0]+20 ,box[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255, 0, 255),
                    1)  # line type


    name_tokens = filename.split('.')
    path = name_tokens[0] + '_boxed.' + name_tokens[1]
    cv2.imwrite(path, orig_image)
    #print(f"Found {len(probs)} objects. The output image is {path}")
end = time()
total_time = end - start
print(total_time, total_time / len(filenames))
