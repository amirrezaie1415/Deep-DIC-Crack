from networks import TernausNet16
import os
import numpy as np
import cv2 as cv
import torch
from tqdm import tqdm
import skimage.io
from utils import zero_pad
from utils import sliding_window
import glob
import pathlib
import torchvision.transforms as T
import warnings
import shutil
import argparse

warnings.filterwarnings("ignore")


def predict(config):
    # load model
    model_name = 'checkpoint_best.pth'
    threshold = 0.5
    model_path = os.path.join('../models', model_name)
    result_path = '../predictions'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    model = TernausNet16()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.train(False)
    model.eval()

    desired_size = 256
    transform = T.Compose([T.Resize((desired_size, desired_size)), T.ToTensor(),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    image_names = [i.split(os.sep)[-1] for i in glob.glob(os.path.join(config.images_path, '*'))]
    FILE_EXTENSIONS = ['.jpg', '.JPG',
                       '.jpeg', '.JPEG',
                       '.png', '.PNG',
                       '.tif', '.tiff', '.TIFF',
                       '.bmp', '.BMP']

    for ind, file in enumerate(image_names):
        tmp = pathlib.Path(file)
        if tmp.suffix not in FILE_EXTENSIONS:
            del image_names[ind]

    for image_name in tqdm(image_names):
        image_file = skimage.io.imread(os.path.join(config.images_path, image_name))
        image_file = (image_file / image_file.max()) * 255
        image_file = np.uint8(image_file)

        org_im_h = image_file.shape[0]
        org_im_w = image_file.shape[1]
        padded_image = zero_pad(image_file, desired_size)

        window_names = []
        windows = []  # as Tensor (ready for to use for deep learning)
        for (x, y, window) in sliding_window(padded_image, step_size=desired_size,
                                             window_size=(desired_size, desired_size)):
            window_names.append(image_name[:-4] + "_{:d}".format(x) + "_{:d}".format(y))
            window = T.ToPILImage()(window)  # as PIL
            window = window.convert('RGB')
            window = transform(window)
            windows.append(torch.reshape(window, [1, 3, desired_size, desired_size]))

        overlay_crack = zero_pad(np.zeros((org_im_h, org_im_w), dtype="uint8"),
                                 desired_size=desired_size)

        with torch.no_grad():
            for window, window_name in zip(windows, window_names):
                window = window.to(device)
                SR = model(window)
                SR_probs = torch.sigmoid(SR)
                SR_probs_arr = SR_probs.view(desired_size, desired_size)
                # SR_probs.detach().numpy().reshape(desired_size, desired_size)
                binary_result = SR_probs_arr > threshold
                binary_result = binary_result.to('cpu').detach().numpy()
                y = int(window_name.split('_')[-1])
                x = int(window_name.split('_')[-2])
                overlay_crack[y:y + desired_size, x:x + desired_size] = binary_result
        overlay_crack = overlay_crack[:org_im_h, :org_im_w] * 255

        image_name_save = image_name[:-4] + '_mask.png'
        skimage.io.imsave(os.path.join(result_path, image_name_save), overlay_crack)
        skimage.io.imsave(os.path.join(result_path, image_name), image_file)

        overlay_name_save = image_name[:-4] + '_overlay.jpg'
        prediction_rgb = np.zeros((overlay_crack.shape[0], overlay_crack.shape[1], 3), dtype='uint8')
        prediction_rgb[:, :, 0] = overlay_crack

        if np.ndim(image_file) == 2:
            image_file = np.stack((image_file,) * 3, axis=-1)
        overlayed_prediction = cv.addWeighted(image_file, 1.0, prediction_rgb, 1.0, 0)
        skimage.io.imsave(os.path.join(result_path, overlay_name_save), overlayed_prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='../dataset/test/', help='path to save the best model')
    config = parser.parse_args()
    predict(config)
