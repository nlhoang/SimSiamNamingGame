import argparse
import datetime
import sys
from pathlib import Path
from tempfile import mkdtemp
import torch
from base.evaluation import train_classifier, evaluate_classifier, extract_features_generalized, safe_first
from base.utils import param_count, display_loss, Logger, str2bool
from config import get_dataloader_train, get_dataloader_eval, configurations, set_seeds
from model import GSSNG


def args_define():
    parser = argparse.ArgumentParser(description='Train the generalized N-agent SSNG model.')
    parser.add_argument('--Note', type=str, default='generalized N-agent SSNG')
    parser.add_argument('--EmCom', type=str, default='EmCom', choices=['EmCom', 'NoCom'])
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--num-of-agent', type=int, default=3, help='number of communicating agents (>= 2)')
    parser.add_argument('--dataset', default='CIFAR',
                        choices=['MNIST', 'FashionMNIST', 'KMNIST', 'CIFAR', 'CIFAR100', 'ImageNet100'])
    parser.add_argument('--num-workers', type=int, default=5, help='12 or 18 or 24')
    parser.add_argument('--eval-epochs', type=int, default=50, help='number of epochs [default: 100]')
    parser.add_argument('--interval-saved', type=int, default=1, help='interval for saving models')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
    parser.add_argument('--debug', type=str2bool, default=False, help='debug vs running')
    args = parser.parse_args()
    args.eval_model = f'SSNG-{args.num_of_agent}'
    return args


def initialize(args):
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


def get_backbones(args):
    """
    config.py only defines args.backbone, backbone2, ..., backbone5 (per dataset).
    Cycle through whichever of those are defined for the chosen dataset so any
    num_of_agent (including > 5) gets a full list of backbone names.
    """
    candidates = [args.backbone]
    for i in range(2, 6):
        candidates.append(getattr(args, f'backbone{i}', None))
    candidates = [b for b in candidates if b is not None]
    if not candidates:
        raise ValueError(f"No backbone configured for dataset '{args.dataset}'")
    return [candidates[i % len(candidates)] for i in range(args.num_of_agent)]


def evaluate(args, model, eval_dataloader_train, eval_dataloader_test, device,
             softLang=False, hardLang=True, eval_latent=False):
    train_labels, train_features, train_latents, train_signs = extract_features_generalized(
        model, eval_dataloader_train, device, softLang=softLang, hardLang=hardLang)
    print('Finish extract train features')

    test_labels, test_features, test_latents, test_signs = extract_features_generalized(
        model, eval_dataloader_test, device, softLang=softLang, hardLang=hardLang)
    print('Finish extract test features')

    # Loop over each agent (0..num_of_agent-1)
    for agent_idx in range(args.num_of_agent):
        # --- sign classifier ---
        tr_sign = train_signs[agent_idx]
        te_sign = test_signs[agent_idx]

        classifier_sign = train_classifier(features=tr_sign, labels=train_labels, num_classes=args.num_classes,
                                           device=device, epochs=args.eval_epochs)
        acc_sign = evaluate_classifier(classifier=classifier_sign, features=te_sign,
                                       labels=test_labels, device=device, max_k=1)
        print(f"Accuracy of agent {agent_idx} (sign): {safe_first(acc_sign)}")

        if eval_latent:
            # --- feature classifier ---
            tr_feat = train_features[agent_idx]
            te_feat = test_features[agent_idx]

            classifier_feature = train_classifier(features=tr_feat, labels=train_labels, num_classes=args.num_classes,
                                                  device=device, epochs=args.eval_epochs)
            acc_feature = evaluate_classifier(classifier=classifier_feature, features=te_feat,
                                              labels=test_labels, device=device, max_k=1)
            print(f"Accuracy of agent {agent_idx} (feature): {safe_first(acc_feature)}")

            # --- latent classifier ---
            tr_lat = train_latents[agent_idx]
            te_lat = test_latents[agent_idx]

            classifier_latent = train_classifier(features=tr_lat, labels=train_labels, num_classes=args.num_classes,
                                                 device=device, epochs=args.eval_epochs)
            acc_latent = evaluate_classifier(classifier=classifier_latent, features=te_lat,
                                             labels=test_labels, device=device, max_k=1)
            print(f"Accuracy of agent {agent_idx} (latent): {safe_first(acc_latent)}")


# Per-agent projector/langCoder depths used by the earlier fixed-N SSNG models
_PROJ_LAYERS = {2: [2, 2], 3: [2, 2, 3], 4: [2, 2, 3, 3], 5: [2, 2, 3, 3, 4]}
_LC_ENC_LAYERS = {2: [1, 1], 3: [1, 1, 2], 4: [1, 1, 2, 2], 5: [1, 1, 2, 2, 2]}
_LC_DEC_LAYERS = {2: [1, 1], 3: [1, 1, 2], 4: [1, 1, 2, 2], 5: [1, 2, 1, 2, 2]}


def get_model(args):
    backbones = get_backbones(args)
    n = args.num_of_agent
    model = GSSNG.MultiAgentSSNG(
        num_agents=n,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        word_length=args.word_length,
        dictionary_size=args.dictionary_size,
        backbones=backbones,
        proj_layers=_PROJ_LAYERS.get(n, 2),
        lc_enc_layers=_LC_ENC_LAYERS.get(n, 1),
        lc_dec_layers=_LC_DEC_LAYERS.get(n, 1),
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
    )
    train = GSSNG.train if args.EmCom == 'EmCom' else GSSNG.train_NoCom
    return model, train


def resume_training(model, optimizer, scheduler, scaler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch


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