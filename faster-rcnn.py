import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, ops
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 简单的 Backbone 网络
class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.stride = 4  # 1 * 2 * 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


# RPN（区域提议网络）
class RPN(nn.Module):
    def __init__(self, in_channels, mid_channels=256, num_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        cls_scores = self.cls_layer(x)
        bbox_preds = self.reg_layer(x)
        return cls_scores, bbox_preds


# 生成 Anchor Box
def generate_anchors(feature_map_size, stride=4, scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    anchors = []
    h, w = feature_map_size
    base_size = stride
    for i in range(h):
        for j in range(w):
            cx = j * stride + stride / 2
            cy = i * stride + stride / 2
            for scale in scales:
                for ratio in ratios:
                    w_box = base_size * scale * (ratio ** 0.5)
                    h_box = base_size * scale / (ratio ** 0.5)
                    anchors.append([cx - w_box / 2, cy - h_box / 2, cx + w_box / 2, cy + h_box / 2])
    return torch.tensor(anchors, dtype=torch.float32)


# Faster R-CNN 主模型
class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.backbone = SimpleBackbone()
        self.rpn = RPN(in_channels=256, num_anchors=9)
        self.roi_align = ops.RoIAlign(output_size=(7, 7), spatial_scale=1 / 4, sampling_ratio=-1)
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        self.stride = self.backbone.stride

    # 每个batch调用一次，batch_size=1
    def forward(self, images, targets=None):
        feature_map = self.backbone(images)
        batch_size, _, h, w = feature_map.shape
        # rpn_cls_scores的形状是(batch_size, 9 * 2, 224/4, 224/4)，
        # rpn_bbox_preds的形状是(batch_size, 9 * 4, 224/4, 224/4)
        rpn_cls_scores, rpn_bbox_preds = self.rpn(feature_map)
        anchors = generate_anchors((h, w), stride=self.stride).to(images.device)

        if self.training:
            assert targets is not None, "Targets must be provided during training"
            losses = {}
            for i in range(batch_size):
                rpn_loss_cls, rpn_loss_bbox = self.compute_rpn_loss(
                    rpn_cls_scores[i], rpn_bbox_preds[i], anchors, targets[i]
                )

                # rpn_cls_scores[i:i+1]的形状是(i, 9 * 2, 224/4, 224/4)
                # proposals的形状是(n, 4)，其中n是proposals的数量，4是[x1, y1, x2, y2]
                proposals = self.generate_proposals(rpn_cls_scores[i:i + 1], rpn_bbox_preds[i:i + 1], anchors)
                # torch.full((proposals.shape[0], 1), i, device=proposals.device)表示创建一个形状为(n,1)的tensor，每一项为i
                # 然后将新创建的tensor与proposals进行按列拼接，得到rois,其形状为：（n,5）
                # 其中n是proposals的数量，5是[batch_index, x1, y1, x2, y2]
                rois = torch.cat([torch.full((proposals.shape[0], 1), i, device=proposals.device), proposals], dim=1)
                roi_features = self.roi_align(feature_map, rois)
                # roi_features.size(0)表示获得roi_features的行数，也就是roi的数量
                # 下面这行代码的意思是，将roi_features的形状从(N, 256, 7, 7)变成(N, 256 * 7 * 7)
                x = roi_features.view(roi_features.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                cls_scores = self.cls_score(x)
                bbox_deltas = self.bbox_pred(x)
                det_loss_cls, det_loss_bbox = self.compute_detection_loss(cls_scores, bbox_deltas, targets[i],
                                                                          proposals)
                losses[f"rpn_loss_cls_{i}"] = rpn_loss_cls
                losses[f"rpn_loss_bbox_{i}"] = rpn_loss_bbox
                losses[f"det_loss_cls_{i}"] = det_loss_cls
                losses[f"det_loss_bbox_{i}"] = det_loss_bbox
            return losses
        else:
            predictions = []
            for i in range(batch_size):
                proposals = self.generate_proposals(rpn_cls_scores[i:i + 1], rpn_bbox_preds[i:i + 1], anchors)
                rois = torch.cat([torch.full((proposals.shape[0], 1), i, device=proposals.device), proposals], dim=1)
                roi_features = self.roi_align(feature_map, rois)
                x = roi_features.view(roi_features.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                cls_scores = self.cls_score(x)
                bbox_deltas = self.bbox_pred(x)
                boxes = self.apply_bbox_deltas(proposals, bbox_deltas, cls_scores.argmax(dim=1))
                scores = cls_scores.softmax(dim=1)
                max_scores, pred_labels = scores.max(dim=1)
                predictions.append({
                    "boxes": boxes,
                    "labels": pred_labels,
                    "scores": max_scores
                })
            return predictions

    # 这个函数作用计算RPN的loss，包括分类损失和回归损失。
    def compute_rpn_loss(self, cls_scores, bbox_preds, anchors, target):
        # cls_scores.view类似于numpy中的reshpae，其中第一个参数-1表示自己计算，第二个参数指明列数
        cls_scores = cls_scores.view(-1, 2)
        bbox_preds = bbox_preds.view(-1, 4)
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        ious = ops.box_iou(anchors, gt_boxes)
        max_ious, max_idx = ious.max(dim=1)
        labels = torch.zeros(anchors.shape[0], dtype=torch.int64, device=cls_scores.device) - 1
        labels[max_ious > 0.7] = 1
        labels[max_ious < 0.3] = 0

        valid_mask = labels >= 0
        # valid_mask表示删除掉anchors中与gt_boxes不匹配的anchors。
        cls_scores = cls_scores[valid_mask]  # 这行代码的含义是删除掉anchors中与gt_boxes不匹配的anchors。
        labels = labels[valid_mask]
        bbox_preds = bbox_preds[valid_mask]
        anchors = anchors[valid_mask]

        rpn_loss_cls = F.cross_entropy(cls_scores, labels, ignore_index=-1) if labels.numel() > 0 else torch.tensor(0.0,
                                                                                                                    device=cls_scores.device)

        pos_mask = labels == 1  # 将所有正样本找到，其结果为pas_mask[0]=true?
        if pos_mask.sum() > 0:
            pos_anchors = anchors[pos_mask]
            pos_preds = bbox_preds[pos_mask]
            pos_gt = gt_boxes[max_idx[pos_mask]]
            target_deltas = self.encode_boxes(pos_anchors, pos_gt)  # 得到真实位置与AnchorBox的偏移量。
            rpn_loss_bbox = F.smooth_l1_loss(pos_preds, target_deltas,
                                             reduction="sum") / pos_mask.sum()  # 计算预测的偏移量与真实偏移量的差异。
        else:
            rpn_loss_bbox = torch.tensor(0.0, device=cls_scores.device)

        return rpn_loss_cls, rpn_loss_bbox

    def compute_detection_loss(self, cls_scores, bbox_deltas, target, proposals):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # box_iou 函数计算 proposals 中的每个 box 与 gt_boxes 中的每个 box 之间的 IoU值
        # 返回的是(N,M)的张量，dim=1表示按行寻找最大值
        ious = ops.box_iou(proposals, gt_boxes)
        # 找出每个 proposal 的最大 IoU 及对应的 ground-truth box 索引
        max_ious, max_idx = ious.max(dim=1)

        # 创建一个与proposals相同维度的tensor，并将其初始化为0。
        labels = torch.full((proposals.shape[0],), 0, dtype=torch.int64, device=cls_scores.device)
        pos_mask = max_ious >= 0.5
        labels[pos_mask] = gt_labels[max_idx[pos_mask]]

        det_loss_cls = F.cross_entropy(cls_scores, labels)

        if pos_mask.sum() > 0:
            pos_proposals = proposals[pos_mask]
            pos_deltas = bbox_deltas[pos_mask]
            pos_gt_boxes = gt_boxes[max_idx[pos_mask]]
            target_deltas = self.encode_boxes(pos_proposals, pos_gt_boxes)
            det_loss_bbox = F.smooth_l1_loss(pos_deltas, target_deltas, reduction="sum") / pos_mask.sum()
        else:
            det_loss_bbox = torch.tensor(0.0, device=cls_scores.device)

        return det_loss_cls, det_loss_bbox

    def encode_boxes(self, proposals, gt_boxes):
        # proposals中的数据格式是[x1, y1, x2, y2]，gt_boxes中的数据格式是[x1, y1, x2, y2]。
        proposals_w = proposals[:, 2] - proposals[:, 0]
        proposals_h = proposals[:, 3] - proposals[:, 1]
        proposals_cx = proposals[:, 0] + proposals_w / 2
        proposals_cy = proposals[:, 1] + proposals_h / 2

        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_cx = gt_boxes[:, 0] + gt_w / 2
        gt_cy = gt_boxes[:, 1] + gt_h / 2

        dx = (gt_cx - proposals_cx) / proposals_w
        dy = (gt_cy - proposals_cy) / proposals_h

        # 这里使用log的原因是为了让dw,dh发生微小的变化，从而更容易通过损失函数的计算更新参数
        # 当gt_w 等于 proposals_w时，log(gt_w / proposals_w) = 0，所以dw = 0
        # 当gt_w 大于 proposals_w时，log(gt_w / proposals_w) 是一个很小的正值
        # 当gt_w 小于 proposals_w时，log(gt_w / proposals_w) 是一个很小的负值
        dw = torch.log(gt_w / proposals_w)
        dh = torch.log(gt_h / proposals_h)
        return torch.stack([dx, dy, dw, dh], dim=1)

    def apply_bbox_deltas(self, proposals, deltas, labels=None):
        # proposals中的数据格式是[x1, y1, x2, y2]，
        # deltas中的数据格式是[dx, dy, dw, dh]。
        proposals_w = proposals[:, 2] - proposals[:, 0]
        proposals_h = proposals[:, 3] - proposals[:, 1]
        proposals_cx = proposals[:, 0] + proposals_w / 2
        proposals_cy = proposals[:, 1] + proposals_h / 2

        if labels is not None:
            # labels是一个一维张量，其长度等于proposals的长度。
            # labels中的每个元素表示对应的proposal的类别标签。
            batch_size = proposals.shape[0]
            indices = torch.arange(batch_size, device=deltas.device) * self.cls_score.out_features + labels
            dx = deltas.view(-1, 4)[indices, 0]
            dy = deltas.view(-1, 4)[indices, 1]
            dw = deltas.view(-1, 4)[indices, 2]
            dh = deltas.view(-1, 4)[indices, 3]
        else:
            dx = deltas[:, 0]
            dy = deltas[:, 1]
            dw = deltas[:, 2]
            dh = deltas[:, 3]

        # dx 表示预测的候选框的偏移比例，比如0.1表示向右移动10%的宽度
        pred_cx = dx * proposals_w + proposals_cx
        pred_cy = dy * proposals_h + proposals_cy
        pred_w = torch.exp(dw) * proposals_w
        pred_h = torch.exp(dh) * proposals_h
        return torch.stack([pred_cx - pred_w / 2, pred_cy - pred_h / 2, pred_cx + pred_w / 2, pred_cy + pred_h / 2],
                           dim=1)

    def generate_proposals(self, cls_scores, bbox_preds, anchors):
        # cls_scores原始形状是(batch_size, 9 * 2, 224/4, 224/4)
        # cls_scores.view(-1, 2)相当于将一个tensor reshape 成二维tensor, 即转成n行2列
        # dim=-1表示按前面参数的最后一个维度计算softmax, 最后一个维度是2
        # [:, 1]表示取c前面参数ls_scores的第二列
        cls_probs = torch.softmax(cls_scores.view(-1, 2), dim=-1)[:, 1]
        bbox_preds = bbox_preds.view(-1, 4)
        proposals = self.apply_bbox_deltas(anchors, bbox_preds)
        scores = cls_probs

        keep = ops.nms(proposals, scores, iou_threshold=0.7)
        proposals = proposals[keep]
        scores = scores[keep]

        # proposals是候选区，共有4列，其中第0列表示cx, 1列表示cy, 2列表示w, 3列表示h
        # 为了不让预测的框超出图像范围，所以对proposals进行裁剪
        proposals[:, 0] = torch.clamp(proposals[:, 0], min=0)
        proposals[:, 1] = torch.clamp(proposals[:, 1], min=0)
        proposals[:, 2] = torch.clamp(proposals[:, 2], max=224)
        proposals[:, 3] = torch.clamp(proposals[:, 3], max=224)

        # numel函数的作用是返回张量中元素的总数
        if scores.numel() == 0:
            return torch.empty((0, 4), device=anchors.device)
        # scores = torch.tensor([0.2, 0.8, 0.1, 0.9, 0.5, 0.7, 0.3, 0.6, 0.4, 0.95, 0.85, 0.15])
        # 取值最大的10个值的索引值
        top_n = torch.topk(scores, min(10, scores.shape[0])).indices
        return proposals[top_n]


# 数据集
class CustomDataset(Dataset):
    def __init__(self):
        self.images = [Image.open("example.jpg").convert("RGB").resize((224, 224))]
        self.targets = [{"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
                         "labels": torch.tensor([1], dtype=torch.int64)}]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = transforms.ToTensor()(self.images[idx])
        return img, self.targets[idx]


# 训练函数
def train_model(model, data_loader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, targets in data_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")


# 推理函数
def inference(model, image_path, device):
    model.eval()
    img = transforms.ToTensor()(Image.open(image_path).convert("RGB").resize((224, 224))).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(img)[0]
    img = transforms.ToPILImage()(img.squeeze(0).cpu())
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score > 0.5:
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_max, linewidth=2, edgecolor="r",
                                     facecolor="none")
            ax.add_patch(rect)
            plt.text(x_min, y_min, f"Label: {label.item()}, Score: {score.item():.2f}", color="white", fontsize=12,
                     bbox=dict(facecolor="red", alpha=0.5))
    plt.axis("off")
    plt.show()


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FasterRCNN(num_classes=2).to(device)
    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_model(model, data_loader, optimizer, num_epochs=5, device=device)
    inference(model, "example.jpg", device)


if __name__ == "__main__":
    main()
