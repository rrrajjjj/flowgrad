from email.policy import strict
import torch

from ..utils import clean_pred, preprocess, preprocess_flow
from ..modules.gradcam import GradCAM
from ..modules.resnet import BasicBlock
from ..modules.resnet import resnet18
from ..modules.resnet import ResNetSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_URL = "https://github.com/hohsiangwu/rethinking-visual-sound-localization/releases/download/v0.1.0-alpha/rc_grad.pt"
checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_URL, map_location=device, progress=True
        )


class RCGrad:
    def __init__(self, modal="vision", checkpoint = checkpoint):
        super(RCGrad).__init__()

        image_encoder = resnet18(modal=modal, pretrained=False)
        audio_encoder = ResNetSpec(
            BasicBlock,
            [2, 2, 2, 2],
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )

        image_encoder.load_state_dict(
            {
                k.replace("image_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("image_encoder")
            }
        )
        audio_encoder.load_state_dict(
            {
                k.replace("audio_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("audio_encoder")
            }
        )

        target_layers = [image_encoder.layer4[-1]]
        self.audio_encoder = audio_encoder
        self.cam = GradCAM(
            model=image_encoder,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=None,
        )
        self.modal=modal

    def pred_audio(self, img, audio, flow = None):
        in_tensor = preprocess(img)
        if self.modal == "flow_IC":
            in_flow = preprocess_flow(flow)
            in_tensor = torch.cat((in_tensor, in_flow), dim=0)

        grayscale_cam = self.cam(
            input_tensor=in_tensor.unsqueeze(0).float(),
            targets=[self.audio_encoder(torch.from_numpy(audio).unsqueeze(0))],
        )
        pred_audio = grayscale_cam[0, :]
        pred_audio = clean_pred(pred_audio)
        return pred_audio


class FlowGradEN:
    def __init__(self, checkpoint = checkpoint):
        super(FlowGradEN).__init__()

        image_encoder = resnet18(modal="vision", pretrained=False)
        flow_encoder = resnet18(modal="flow_EN", pretrained = False)
        audio_encoder = ResNetSpec(
            BasicBlock,
            [2, 2, 2, 2],
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )


        image_encoder.load_state_dict(
            {
                k.replace("image_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("image_encoder")
            }, 
            strict=False
        )
        audio_encoder.load_state_dict(
            {
                k.replace("audio_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("audio_encoder")
            },
            strict=False
        )

        flow_encoder.load_state_dict(
            {
                k.replace("flow_encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("flow_encoder")
            },
            strict=False
        )

        self.gradcam_encoder_img = image_encoder 
        self.gradcam_encoder_flow = flow_encoder

        target_layers_img = [self.gradcam_encoder_img.layer4[-1]]
        target_layers_flow = [self.gradcam_encoder_flow.layer4[-1]]
        self.audio_encoder = audio_encoder
        self.cam_img = GradCAM(
            model=self.gradcam_encoder_img,
            target_layers=target_layers_img,
            use_cuda=False,
            reshape_transform=None,
        )
        self.cam_flow = GradCAM(
            model=self.gradcam_encoder_flow,
            target_layers=target_layers_flow,
            use_cuda=False,
            reshape_transform=None,
        )

    def pred_audio(self, img, audio, flow = None):
        img_tensor = preprocess(img)
        flow_tensor = preprocess_flow(flow)

        grayscale_cam_img = self.cam_img(
            input_tensor=img_tensor.unsqueeze(0).float(),
            targets=[self.audio_encoder(torch.from_numpy(audio).unsqueeze(0))],
        )
        grayscale_cam_flow = self.cam_flow(
            input_tensor=flow_tensor.unsqueeze(0).float(),
            targets=[self.audio_encoder(torch.from_numpy(audio).unsqueeze(0))],
        )

        pred_audio_img = grayscale_cam_img[0, :]
        pred_audio_flow = grayscale_cam_flow[0, :]
        pred_audio = clean_pred(pred_audio_img*pred_audio_flow)
        return pred_audio

