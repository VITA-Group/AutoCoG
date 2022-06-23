import torch
import torch.nn.functional as F


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


def train(cfg, model, optimizer, loss_fn, data, epoch, edge_weight, architecture=None, mask_loss=0):
    model.train()
    torch.cuda.empty_cache()
    if cfg.DATA.NAME.lower() != 'ogbn-arxiv':
        logits = model(data.x, data.edge_index, edge_weight)
        logits = F.log_softmax(logits, 1)
        loss = loss_fn(logits[data.train_mask],
                       data.y[data.train_mask])
        smooth_loss = -logits[data.train_mask].mean(dim=-1).mean()
        loss = 0.97 * loss + 0.03 * smooth_loss
    elif cfg.DATA.NAME.lower() == 'ogbn-arxiv':
        pred = model(data.x, data.edge_index, edge_weight=edge_weight)
        pred = F.log_softmax(pred[data.train_mask], 1)
        loss = loss_fn(pred, data.y.squeeze(1)[data.train_mask])
    else:
        raise NotImplementedError
    loss = loss + mask_loss
    optimizer.zero_grad()
    loss.backward()
    if cfg.MODEL.CLIP_NORM > 0:
        torch.nn.utils.clip_grad_norm(model.net_parameters(),
                                      cfg.MODEL.CLIP_NORM)
    optimizer.step()
    if architecture is not None and epoch >= cfg.TRAIN.BEGIN_EPOCH:
        data.edge_weight = edge_weight.detach().clone()
        architecture.step(data, model)
    return loss.item()


def validate(cfg, model, data, loss_fn):
    logits = model(data.x, data.edge_index, data.edge_weight, val=True)
    logits = F.log_softmax(logits, 1)
    if cfg.DATA.NAME.lower() != 'ogbn-arxiv':
        acc_valid = evaluate(logits, data.y, data.val_mask)
        loss_val = loss_fn(logits[data.val_mask], data.y[data.val_mask])
        smooth_loss = -logits[data.val_mask].mean(dim=-1).mean()
        loss_val = 0.97 * loss_val + 0.03 * smooth_loss
    elif cfg.DATA.NAME.lower() == 'ogbn-arxiv':
        loss_val = loss_fn(logits[data.val_mask], data.y.squeeze(1)[data.val_mask])
        y_pred = logits.argmax(dim=-1, keepdim=True)
        acc_valid = data.evaluator.eval({
            'y_true': data.y[data.val_mask],
            'y_pred': y_pred[data.val_mask]
        })['acc']
    else:
        raise NotImplementedError
    return acc_valid, loss_val


def test(cfg, model, data, loss_fn):
    if cfg.DATA.NAME.lower() != 'ogbn-arxiv':
        logits = model(data.x, data.edge_index, data.edge_weight)
        logits = F.log_softmax(logits, 1)
        acc_test = evaluate(logits, data.y, data.test_mask)
        loss_test = loss_fn(logits[data.test_mask], data.y[data.test_mask])
    elif cfg.DATA.NAME.lower() == 'ogbn-arxiv':
        logits = model(data.x, data.edge_index, data.edge_weight)
        logits = F.log_softmax(logits, 1)
        loss_test = loss_fn(logits[data.test_mask], data.y.squeeze(1)[data.test_mask])
        y_pred = logits.argmax(dim=-1, keepdim=True)
        acc_test = data.evaluator.eval({
            'y_true': data.y[data.test_mask],
            'y_pred': y_pred[data.test_mask]
        })['acc']
    else:
        raise NotImplementedError
    return acc_test, loss_test.item()


def validate_test(cfg, model, data, edge_weight, loss_fn):
    logits = model(data.x, data.edge_index, edge_weight, val=True)
    logits = F.log_softmax(logits, 1)
    if cfg.DATA.NAME.lower() != 'ogbn-arxiv':
        acc_valid = evaluate(logits, data.y, data.val_mask)
        acc_test = evaluate(logits, data.y, data.test_mask)
        loss_test = loss_fn(logits[data.test_mask], data.y[data.test_mask])
        loss_val = loss_fn(logits[data.val_mask], data.y[data.val_mask])
    elif cfg.DATA.NAME.lower() == 'ogbn-arxiv':
        loss_val = loss_fn(logits[data.val_mask], data.y.squeeze(1)[data.val_mask])
        loss_test = loss_fn(logits[data.test_mask], data.y.squeeze(1)[data.test_mask])
        y_pred = logits.argmax(dim=-1, keepdim=True)
        acc_valid = data.evaluator.eval({
            'y_true': data.y[data.val_mask],
            'y_pred': y_pred[data.val_mask]
        })['acc']
        acc_test = data.evaluator.eval({
            'y_true': data.y[data.test_mask],
            'y_pred': y_pred[data.test_mask]
        })['acc']
    else:
        raise NotImplementedError
    return acc_valid, acc_test, loss_val, loss_test
