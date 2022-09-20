import os
import sys
import torch

# Available after the above append
# it's in the model folder
from models.p2pnet import P2PNet
from models.backbone import Backbone_VGG

def main():
    model_name = "taipanlan_wights"

    print("Loading Model")
    # Create the model
    model_backbone = Backbone_VGG("vgg16_bn", True)
    model = P2PNet(model_backbone, 2, 2)

    # Load Weights
    checkpoint = torch.load(f"{model_name}/best_mae.pth", map_location=torch.device('cuda:0'))
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    model.eval()  # Put in inference mode

    # Create dummy input
    #dummy_input = torch.randn(1, 3, 640, 640).cuda()
    width =height= 640
    onnx_model_name = f"{model_name}_h{height}_w{width}.onnx"
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    dummy_input = torch.randn(1, 3, new_width, new_height).cuda()
    # dummy_input = (dummy_input0, dummy_input1)
    print(dummy_input.shape)
    outputs = model(dummy_input)
    print(outputs['pred_logits'].shape)
    print(outputs['pred_points'].shape)
    # Export as ONNX
    print(f"Exporting as ONNX: {onnx_model_name}")
    torch.onnx._export(
        model,
        dummy_input,
        onnx_model_name,  # Output name
        opset_version=11,  # ONNX Opset Version
        export_params=True,  # Store the trained parameters in the model file
        do_constant_folding=True,  # Execute constant folding for optimization
        input_names=['input'],  # the model's input names
        # output_names = ['pred_logits', 'pred_points'], # the model's output names (see forward in the architecture)
        output_names=['pred_logits', 'pred_points'],  # the model's output names (see forward in the architecture)
        dynamic_axes={
            # Input is an image [batch_size, channels, width, height]
            # all of it can be variable so we need to add it in dynamic_axes
            'input': {
                0: 'batch_size',
                1: 'channel',
                2: 'height',
                3: 'width'
            },
            'pred_logits': [0, 1, 2],
            'pred_points': [0, 1, 2],
        }
    )


if __name__ == "__main__":
    main()
