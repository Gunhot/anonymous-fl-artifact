import os
import torch
from dataLoader.dataset import get_dataset
from nn_models.resnet import ResNet20, ResNet18, ResNet50
from nn_models.mobilenet import MobileNetV2
from torch.utils.data import DataLoader

def build_model(args):
    in_channels = 1 if args.dataset in ['mnist', 'femnist'] else 3
    if args.model == 'ResNet18':
        return ResNet18(num_classes=args.num_classes)
    elif args.model == 'ResNet50':
        return ResNet50(num_classes=args.num_classes)
    elif args.model == 'MobileNetV2':
        return MobileNetV2(num_classes=args.num_classes, in_channels=in_channels)
    elif args.model == 'ResNet20':
        return ResNet20(num_classes=args.num_classes)


def get_model_sd(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_model_sd(model, model_sd):
    model.load_state_dict(model_sd)
    return model

def gpu_train_worker(trainQ, train_ack_q, device, args):
    dataset, _ = get_dataset(args)
    train_model = build_model(args)

    while True:
        msg = trainQ.get()

        if msg == 'kill':
            break

        elif msg['type'] == 'train':
            processing_node = msg['node']
            set_model_sd(train_model, msg['weight'])
            round = msg['round']
            local_epoch = msg.get('local_epoch')
            finetune = msg['finetune']
            virtual_sd = msg['virtual_weight']

            model_weight, finetune_weight, virtual_weight = processing_node.train(
                device, msg['lr'], train_model, dataset, round,
                local_epoch=local_epoch, finetune=finetune, virtual_sd=virtual_sd
            )

            result = {
                'weight': model_weight,
                'finetune_weight': finetune_weight,
                'virtual_weight': virtual_weight,
                'id': processing_node.nodeID
            }

            train_ack_q.put(result)

            del model_weight
            del finetune_weight
            del virtual_weight
            del result

        del msg


def evaluate_classification(model, test_loader, args, device, roundIdx, type):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

        preds = preds.cpu()
        labels = labels.cpu()
        correct += (preds == labels).sum().item()
        total += len(labels)

    accuracy = correct / total if total > 0 else 0.0
    log_str = f"Round {roundIdx} | Accuracy: {accuracy:.4f}\n"
    print(f"[{type}] {log_str}", end='')

    with open(os.path.join(args.log_name, f"{type}.txt"), "a") as f:
        f.write(log_str)



def gpu_test_worker(testQ, ackQ, device, args, worker_type="server"):
    _, test_dataset = get_dataset(args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    local_model = build_model(args)
    local_model.to(device)

    while True:
        msg = testQ.get()

        if msg == 'kill':
            break
        elif isinstance(msg, str):
            pass
        elif isinstance(msg, dict) and 'weight' in msg:
            model_sd = msg['weight']
            roundIdx = msg['round']
            current_worker_type = msg.get('worker_type', worker_type)

            try:
                set_model_sd(local_model, model_sd)
                evaluate_classification(local_model, test_loader, args, device, roundIdx, current_worker_type)
            except Exception as e:
                print(e)
                pass
            finally:
                ackQ.put(True)
                del model_sd

        del msg
