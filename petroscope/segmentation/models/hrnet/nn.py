"""
HRNet + OCR network implementation for semantic segmentation.

This module implements the original High-Resolution Network (HRNet) with
Object-Contextual Representations (OCR) exactly as described in:
- "Deep High-Resolution Representation Learning for Visual Recognition"
- "Object-Contextual Representations for Semantic Segmentation"

Based on the official implementation:
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    torch = None

from petroscope.utils.lazy_imports import torch, nn, F  # noqa

# Configuration constants
BN_MOMENTUM = 0.1
ALIGN_CORNERS = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for HRNet."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, momentum=BN_MOMENTUM
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SpatialGather_Module(nn.Module):
    """
    Aggregate the context features according to the initial
    predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size()
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block.
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                           maps (save memory cost)
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))

        self.f_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.f_object = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.f_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.f_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # Add bg context
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)

        if self.scale > 1:
            context = F.interpolate(
                input=context,
                size=(h, w),
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock2D, self).__init__(
            in_channels, key_channels, scale
        )


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """

    def __init__(
        self, in_channels, key_channels, out_channels, scale=1, dropout=0.1
    ):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels, key_channels, scale
        )
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(
                _in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            raise ValueError(error_msg)

    def _make_one_branch(
        self, branch_index, block, num_blocks, num_channels, stride=1
    ):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM,
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = (
            num_channels[branch_index] * block.expansion
        )
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_inchannels[i], momentum=BN_MOMENTUM
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM,
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM,
                                    ),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=ALIGN_CORNERS,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        super(HighResolutionNet, self).__init__()

        # Stem net
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        self.stage1_cfg = config["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # Stage2
        self.stage2_cfg = config["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        # Stage3
        self.stage3_cfg = config["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        # Stage4
        self.stage4_cfg = config["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

        last_inp_channels = int(sum(pre_stage_channels))
        ocr_mid_channels = config["OCR"]["MID_CHANNELS"]
        ocr_key_channels = config["OCR"]["KEY_CHANNELS"]

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(
                last_inp_channels,
                ocr_mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(ocr_mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(config["NUM_CLASSES"])

        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05,
        )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels,
            config["NUM_CLASSES"],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(
                last_inp_channels,
                last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_inp_channels,
                config["NUM_CLASSES"],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer
    ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self, layer_config, num_inchannels, multi_scale_output=True
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # Store input size for final upsampling
        input_size = (x.size(2), x.size(3))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Upsampling to highest resolution branch
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x1 = F.interpolate(
            y_list[1],
            size=(x0_h, x0_w),
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        x2 = F.interpolate(
            y_list[2],
            size=(x0_h, x0_w),
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        x3 = F.interpolate(
            y_list[3],
            size=(x0_h, x0_w),
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )

        feats = torch.cat([y_list[0], x1, x2, x3], 1)

        out_aux_seg = []

        # OCR
        out_aux = self.aux_head(feats)
        # Compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        # Upsample final outputs to input resolution
        out_aux = F.interpolate(
            out_aux,
            size=input_size,
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        out = F.interpolate(
            out,
            size=input_size,
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return out_aux_seg

    def init_weights(self):
        """Initialize model weights with normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_seg_model(config, **kwargs):
    """Create segmentation model from config."""
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model


# Configuration for different models
hrnet_config = {
    "hrnetv2_w18": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM",
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4],
            "NUM_CHANNELS": [18, 36],
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [18, 36, 72],
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [18, 36, 72, 144],
            "FUSE_METHOD": "SUM",
        },
        "OCR": {"MID_CHANNELS": 512, "KEY_CHANNELS": 256},
    },
    "hrnetv2_w32": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM",
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4],
            "NUM_CHANNELS": [32, 64],
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [32, 64, 128],
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [32, 64, 128, 256],
            "FUSE_METHOD": "SUM",
        },
        "OCR": {"MID_CHANNELS": 512, "KEY_CHANNELS": 256},
    },
    "hrnetv2_w48": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM",
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4],
            "NUM_CHANNELS": [48, 96],
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [48, 96, 192],
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [48, 96, 192, 384],
            "FUSE_METHOD": "SUM",
        },
        "OCR": {"MID_CHANNELS": 512, "KEY_CHANNELS": 256},
    },
}


class HighResolutionNetOCR(nn.Module):
    """HRNet + OCR implementation matching the original paper."""

    def __init__(
        self,
        num_classes,
        width=32,
        in_channels=3,
        ocr_mid_channels=512,
        ocr_key_channels=256,
        dropout=0.05,
    ):
        super(HighResolutionNetOCR, self).__init__()

        self.num_classes = num_classes
        backbone_key = f"hrnetv2_w{width}"

        if backbone_key not in hrnet_config:
            raise ValueError(f"Unsupported width: {width}")

        config = hrnet_config[backbone_key].copy()
        config["NUM_CLASSES"] = num_classes
        config["OCR"]["MID_CHANNELS"] = ocr_mid_channels
        config["OCR"]["KEY_CHANNELS"] = ocr_key_channels

        self.backbone = HighResolutionNet(config)
        # Initialize weights with random values
        self.backbone.init_weights()

    def forward(self, x):
        return self.backbone(x)


# Wrapper class for backward compatibility
class HRNetWithOCR(nn.Module):
    """
    HRNet neural network implementation.

    This is the pure PyTorch neural network implementation without training
    logic. For the full model with training capabilities, use the HRNet wrapper
    in model.py.
    """

    def __init__(
        self,
        n_classes,
        width=32,
        in_channels=3,
        ocr_mid_channels=512,
        dropout=0.1,
        use_aux_head=True,
        pretrained=False,
        **kwargs,
    ):
        super(HRNetWithOCR, self).__init__()

        # Extract width from backbone parameter if provided
        backbone = kwargs.get("backbone", f"hrnetv2_w{width}")

        # Store parameters for reference
        self.n_classes = n_classes
        self.width = width
        self.in_channels = in_channels
        self.ocr_mid_channels = ocr_mid_channels
        self.dropout = dropout
        self.use_aux_head = use_aux_head
        self.pretrained = pretrained
        self.backbone = backbone

        if "hrnetv2_w" in backbone:
            width = int(backbone.split("w")[-1])
            self.width = width

        # Map to OCR key channels based on mid channels
        ocr_key_channels = ocr_mid_channels // 2

        # Create the actual model
        self.model = HighResolutionNetOCR(
            num_classes=n_classes,
            width=width,
            in_channels=in_channels,
            ocr_mid_channels=ocr_mid_channels,
            ocr_key_channels=ocr_key_channels,
            dropout=dropout,
        )

    def forward(self, x):
        """Forward pass through the network."""
        outputs = self.model(x)

        if self.training and self.use_aux_head and len(outputs) == 2:
            # Return tuple for auxiliary loss computation (main, aux)
            return (outputs[1], outputs[0])
        else:
            # Return single tensor for inference
            return outputs[-1] if isinstance(outputs, list) else outputs
