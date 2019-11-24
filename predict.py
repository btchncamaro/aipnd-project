import torch
import argparse
from train import get_pretrained_model
from PIL import Image
import numpy as np
import json


def main():
    command_args = setCommandArgs()
    model, model_name = load_model(command_args)
    topk_probs, topk_classes = predict(command_args, model)
    display_category_names(command_args, model, topk_probs, topk_classes)


def display_category_names(command_args, model, topk_probs, topk_classes):
    if command_args.category_names != None:
        with open(command_args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    elif model.cat_to_name != None:
        cat_to_name = model.cat_to_name

    if cat_to_name != None:
        print(f"\nTop {command_args.top_k} predictions for image {command_args.input}:")
        for prob, clazz in zip(topk_probs, topk_classes):
            if str(clazz) not in cat_to_name.keys():
                break
            print(f"    {cat_to_name[str(clazz)]} [{clazz}]: {prob*100:.4f}%")
        print('\n')
    else:
        print(f"\ntopk_probs: {topk_probs}")
        print(f"topk_classes: {topk_classes}")


def load_model(command_args):
    print(f"loading checkpoint: {command_args.checkpoint}")
    checkpoint_load = torch.load(command_args.checkpoint)
    model, model_name = get_pretrained_model(checkpoint_load['arch'])

    model.classifier = checkpoint_load['classifier']
    model.load_state_dict(checkpoint_load['state_dict'])
    model.cat_to_name = checkpoint_load['cat_to_name']

    return model, model_name


def predict(command_args, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    device = torch.device("cuda" if command_args.gpu else "cpu")
    model.to(device)
    model.eval()

    img = Image.open(command_args.input)
    processed_img = process_image(img)
    processed_img = processed_img.reshape(1, 3, 224, 224)
    processed_img_tensor = torch.from_numpy(processed_img)

    logps = model.forward(processed_img_tensor.to(device).float())

    # Calculate accuracy
    ps = torch.exp(logps)
    ps_topk = ps.topk(command_args.top_k)

    topk_probs = ps_topk[0][0].cpu().detach().numpy()
    topk_classes = ps_topk[1][0].cpu().numpy()

    return topk_probs, topk_classes


# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    if image.width > image.height:
        image = image.resize((int(image.width * 256 / image.height), 256), resample=Image.BILINEAR)
    else:
        image = image.resize((256, int(image.height * 256 / image.width)), resample=Image.BILINEAR)

    # crop out the center 224x224 portion of the image
    image = image.crop((int(image.width / 2) - 112,
                        int(image.height / 2) - 112,
                        int(image.width / 2) + 112,
                        int(image.height / 2) + 112))

    # The color channel needs to be first and retain the order of the other two dimensions.
    np_image = np.array(image)
    np_image = np_image.transpose((2, 0, 1))

    # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1.
    np_image = np_image / 255.0

    # You'll want to subtract the means from each color channel, then divide by the standard deviation.
    np_image[0] = (np_image[0] - 0.485) / 0.229
    np_image[1] = (np_image[1] - 0.456) / 0.224
    np_image[2] = (np_image[2] - 0.406) / 0.225

    return np_image


def setCommandArgs():
    parser = argparse.ArgumentParser(
        description='Options for predicting an image\'s class'
    )

    parser.add_argument('input', help='path to an image file.')
    parser.add_argument('checkpoint', help='path to a model checkpoint')
    parser.add_argument('--top_k', action="store", dest="top_k", required=False, type=int, default=5,
                        help='return top k most likely classes')
    parser.add_argument('--category_names', action="store", dest="category_names",
                        help='json file with mapping of class indexes to category names')
    parser.add_argument('--gpu', action="store_true", dest="gpu", default=False, help='use gpu')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

# python predict.py ./flowers/valid/24/image_06819.jpg  checkpoint_vgg16.pth --top_k 8 --gpu
# python predict.py ./flowers/valid/24/image_06819.jpg  checkpoint_vgg16.pth --top_k 8 --category_names cat_to_name.json --gpu