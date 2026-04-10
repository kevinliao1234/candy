import argparse
import torch
from tool.darknet2pytorch import Darknet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgfile', type=str, help='path to cfg file')
    parser.add_argument('weightfile', type=str, help='path to darknet weight file')
    parser.add_argument('--save_path', type=str, default='yolov4.pth', help='path to save .pth file')
    args = parser.parse_args()

    model = Darknet(args.cfgfile)
    model.load_weights(args.weightfile)
    print(f'Model loaded from {args.weightfile}')

    torch.save(model.state_dict(), args.save_path)
    print(f'Saved PyTorch model to {args.save_path}')
