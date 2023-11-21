import os
import random
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import build_dataset
from datasets.utils import build_data_loader
import torchvision.transforms as transforms
import clip
from clip.utils import (cls_acc,
                        clip_classifier,
                        build_cache_model,
                        pre_load_features)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Meta-Adapter in yaml format')
    args = parser.parse_args()

    return args


class MetaAdapter(nn.Module):
    def __init__(self, dim=1024, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_proj = nn.Linear(dim, 1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, 1)

    def forward(self, query, key, value):
        B, K, C = key.shape
        res = query

        query = query.reshape(B, 1, C)
        key = torch.cat([query, key], dim=1)
        value = torch.cat([query, value], dim=1)
        query = self.q_proj(query).reshape(B, self.num_heads, C)
        key = self.k_proj(key)

        query = query.reshape(B, self.num_heads, 1, -1).permute(0, 2, 1, 3)
        key = key.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)
        value = value.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)

        attn_weight = (query @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float))).softmax(-1)
        attn = attn_weight @ value

        alpha = torch.nn.functional.sigmoid(self.alpha_proj(res).reshape(B, -1, 1, 1))
        attn = (alpha * attn).squeeze()

        attn = res + attn
        attn = F.normalize(attn, p=2, dim=-1)
        return attn


def run_meta_adapter(cfg, cache_keys, test_features, test_labels, clip_weights, clip_model,
                     train_loader_image):
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("**** Zero-shot CLIP's test accuracy on novel classes: {:.2f}. ****".format(acc))

    adapter = MetaAdapter(dim=cache_keys.shape[0]).to(clip_model.dtype).cuda()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_image))

    best_acc, best_epoch = 0.0, 0

    query = clip_weights.T
    key = cache_keys.T.reshape(query.shape[0], -1, query.shape[1])

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_image)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            weights = adapter(query, key, key)
            tip_logits = 100. * image_features @ weights.T

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # update cache_keys
            with torch.no_grad():
                for tar, feat in zip(target, image_features):
                    key[tar] = torch.cat([feat[None, :], key[tar][:key.shape[1] - 1]], dim=0)

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()

        query_test = clip_weights.T
        key_test = cache_keys.T.reshape(query_test.shape[0], -1, query_test.shape[1])
        weights = adapter(query_test, key_test, key_test)
        tip_logits = 100. * test_features @ weights.T
        acc = cls_acc(tip_logits, test_labels)

        if acc > best_acc:
            best_acc = acc
            torch.save(adapter.state_dict(), cfg['cache_dir'] + "/best_meta_" + str(cfg['shots']) + "shots.pt")
            torch.save(key, cfg['cache_dir'] + "/keys" + str(cfg['shots']) + "shots.pt")

    print("**** Meta-Adapter's best accuracy: {:.2f}. ****".format(best_acc))


def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    if cfg['dataset'] == 'imagenet':
        test_loader = torch.utils.data.DataLoader(dataset.val, batch_size=64, num_workers=8, shuffle=False)
        train_loader_cache = torch.utils.data.DataLoader(dataset.full, batch_size=64, num_workers=8, shuffle=False)
        train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=64, num_workers=8, shuffle=True)
    else:
        test_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                        shuffle=False)
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        train_loader_cache = build_data_loader(data_source=dataset.full, batch_size=64, tfm=train_tranform,
                                               is_train=True, shuffle=False)
        train_loader_F = build_data_loader(data_source=dataset.train, batch_size=64, tfm=train_tranform,
                                           is_train=True, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("Constructing cache model by few-shot visual features and labels.")
    cache_keys = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("Loading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    run_meta_adapter(cfg, cache_keys, test_features, test_labels, clip_weights, clip_model, train_loader_F)


if __name__ == '__main__':
    main()
