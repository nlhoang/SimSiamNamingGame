"""
SimSiam Naming Game: 2-agent emergent communication.
Trains model.SSNG.SSNG and evaluates the emergent messages via linear-probe classification accuracy.
"""
import argparse
import datetime
import sys
from pathlib import Path
from tempfile import mkdtemp
import torch
from base.evaluation import train_classifier, evaluate_classifier, extract_features, safe_first
from base.utils import param_count, display_loss, Logger, str2bool
from config import get_dataloader_train, get_dataloader_eval, configurations, set_seeds
from model import SSNG


def args_define():
    parser = argparse.ArgumentParser(description='Train the 2-agent SSNG model.')
    parser.add_argument('--Note', type=str, default='KL w, not ce')
    parser.add_argument('--EmCom', type=str, default='EmCom', choices=['EmCom', 'NoCom'])
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--dataset', default='CIFAR',
                        choices=['MNIST', 'FashionMNIST', 'KMNIST', 'CIFAR', 'CIFAR100', 'ImageNet100'])
    parser.add_argument('--num-workers', type=int, default=5, help='12 or 18 or 24')
    parser.add_argument('--eval-epochs', type=int, default=50, help='number of epochs [default: 100]')
    parser.add_argument('--interval-saved', type=int, default=10, help='interval for saving models')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
    parser.add_argument('--debug', type=str2bool, default=False, help='debug vs running')
    args = parser.parse_args()
    return args


def initialize(args):
    args.num_of_agent = 2
    args.eval_model = 'SSNG'
    configurations(args)

    if args.debug:
        args.epochs = 2
        args.eval_epochs = 2

    runId = args.eval_model + '-' + args.dataset + '-' + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    sys.stdout = Logger('{}/run.log'.format(runPath))
    print('Expt:', runPath)
    print('RunID:', runId)
    return runPath


def evaluate(args, model, eval_dataloader_train, eval_dataloader_test, device,
             softLang=False, hardLang=True, eval_latent=False):
    (train_labels,
     train_feature1, train_latent1, train_sign1,
     train_feature2, train_latent2, train_sign2) = extract_features(
        model, eval_dataloader_train, device, softLang=softLang, hardLang=hardLang)
    print('Finish extract train features')

    (test_labels,
     test_feature1, test_latent1, test_sign1,
     test_feature2, test_latent2, test_sign2) = extract_features(
        model, eval_dataloader_test, device, softLang=softLang, hardLang=hardLang)
    print('Finish extract test features')

    train_features = [train_feature1, train_feature2]
    train_latents = [train_latent1, train_latent2]
    train_signs = [train_sign1, train_sign2]
    test_features = [test_feature1, test_feature2]
    test_latents = [test_latent1, test_latent2]
    test_signs = [test_sign1, test_sign2]

    for agent_idx in range(2):
        # --- sign classifier ---
        tr_sign = train_signs[agent_idx]
        te_sign = test_signs[agent_idx]

        classifier_sign = train_classifier(features=tr_sign, labels=train_labels, num_classes=args.num_classes,
                                           device=device, epochs=args.eval_epochs)
        acc_sign = evaluate_classifier(classifier=classifier_sign, features=te_sign,
                                       labels=test_labels, device=device, max_k=1)
        print(f"Accuracy of {chr(ord('A') + agent_idx)} (sign): {safe_first(acc_sign)}")

        if eval_latent:
            # --- feature classifier ---
            tr_feat = train_features[agent_idx]
            te_feat = test_features[agent_idx]

            classifier_feature = train_classifier(features=tr_feat, labels=train_labels, num_classes=args.num_classes,
                                                  device=device, epochs=args.eval_epochs)
            acc_feature = evaluate_classifier(classifier=classifier_feature, features=te_feat,
                                              labels=test_labels, device=device, max_k=1)
            print(f"Accuracy of {chr(ord('A') + agent_idx)} (feature): {safe_first(acc_feature)}")

            # --- latent classifier ---
            tr_lat = train_latents[agent_idx]
            te_lat = test_latents[agent_idx]

            classifier_latent = train_classifier(features=tr_lat, labels=train_labels, num_classes=args.num_classes,
                                                 device=device, epochs=args.eval_epochs)
            acc_latent = evaluate_classifier(classifier=classifier_latent, features=te_lat,
                                             labels=test_labels, device=device, max_k=1)
            print(f"Accuracy of {chr(ord('A') + agent_idx)} (latent): {safe_first(acc_latent)}")


def get_model(args):
    model = SSNG.SSNG(feature_dim=args.feature_dim, latent_dim=args.latent_dim,
                      word_length=args.word_length, dictionary_size=args.dictionary_size,
                      backbone1=args.backbone, backbone2=args.backbone2,
                      pretrained_backbone=args.pretrained_backbone, freeze_backbone=args.freeze_backbone)
    train = SSNG.train if args.EmCom == 'EmCom' else SSNG.train_NoCom  # SSNG vs. No-Communication ablation

    return model, train


if __name__ == "__main__":
    args = args_define()
    args.run_path = initialize(args) + '/'
    print(args)
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, train = get_model(args)
    train_dataloader, saved = get_dataloader_train(args)
    model.to(device)
    #print(model)
    print('Model Size: {}'.format(param_count(model)))

    # Training and Evaluating
    loss_history = train(model=model, dataloader=train_dataloader, learning_rate=args.learning_rate, device=device,
                         epochs=args.epochs, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                         save_interval=args.interval_saved, save_prefix=saved)

    if loss_history is not None:
        display_loss(loss_history, save_path=args.run_path+'loss.png')
    print('--------------------')

    eval_dataloader_train, eval_dataloader_test = get_dataloader_eval(args)
    evaluate(args, model, eval_dataloader_train, eval_dataloader_test, device, softLang=False, hardLang=True)
