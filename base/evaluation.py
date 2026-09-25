import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def safe_first(x, default="-"):
    """Return x[0] if x is indexable and non-empty, otherwise default."""
    if x is None:
        return default
    try:
        return x[0]
    except (IndexError, TypeError):
        return default


def extract_features(model, dataloader, device,
                     softLang=False, hardLang=True):
    model.eval()
    feature1, feature2 = [], []
    latent1, latent2 = [], []
    sign1, sign2 = [], []
    labels = []

    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels.append(targets.cpu())

            with torch.amp.autocast(device_type='cuda'):
                y1 = model.encoder1(imgs)
                z1 = model.projector1(y1)
                probs1, onehot1, message1 = model.langCoder1.encoder(z1)
                tokens1 = probs1 if softLang else onehot1 if hardLang else message1
                if tokens1.dim() == 3:
                    tokens1 = tokens1.view(tokens1.size(0), -1)

                y2 = model.encoder2(imgs)
                z2 = model.projector2(y2)
                probs2, onehot2, message2 = model.langCoder2.encoder(z2)
                tokens2 = probs2 if softLang else onehot2 if hardLang else message2
                if tokens2.dim() == 3:
                    tokens2 = tokens2.view(tokens2.size(0), -1)

            feature1.append(y1.contiguous().cpu())
            latent1.append(z1.contiguous().cpu())
            sign1.append(tokens1.contiguous().cpu())

            feature2.append(y2.contiguous().cpu())
            latent2.append(z2.contiguous().cpu())
            sign2.append(tokens2.contiguous().cpu())

    labels = torch.cat(labels).numpy()
    feature1 = torch.cat(feature1).numpy()
    latent1 = torch.cat(latent1).numpy()
    sign1 = torch.cat(sign1).numpy()

    feature2 = torch.cat(feature2).numpy()
    latent2 = torch.cat(latent2).numpy()
    sign2 = torch.cat(sign2).numpy()

    return (labels,
            feature1, latent1, sign1,
            feature2, latent2, sign2)


def extract_features_generalized(model, dataloader, device,
                                 softLang=False, hardLang=False):
    """
    Same as extract_features but for model.GSSNG.MultiAgentSSNG, which keeps its
    per-agent modules in nn.ModuleLists (model.encoders/projectors/langcoders)
    instead of numbered attributes (encoder1, encoder2, ...). Returns lists of length
    model.num_agents instead of a fixed 5-agent tuple, so it works for any N.
    """
    model.eval()
    num_agents = model.num_agents

    all_features = [[] for _ in range(num_agents)]
    all_latents  = [[] for _ in range(num_agents)]
    all_signs    = [[] for _ in range(num_agents)]
    labels = []

    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels.append(targets.cpu())

            with torch.amp.autocast(device_type='cuda'):
                for i in range(num_agents):
                    y = model.encoders[i](imgs)
                    z = model.projectors[i](y)
                    probs, onehot, message = model.langcoders[i].encoder(z)
                    tokens = probs if softLang else onehot if hardLang else message

                    if tokens.dim() == 3:
                        tokens = tokens.view(tokens.size(0), -1)

                    all_features[i].append(y.contiguous().cpu())
                    all_latents[i].append(z.contiguous().cpu())
                    all_signs[i].append(tokens.contiguous().cpu())

    labels = torch.cat(labels).numpy()
    features = [torch.cat(feats).numpy() for feats in all_features]
    latents = [torch.cat(lats).numpy() for lats in all_latents]
    signs = [torch.cat(sg).numpy() for sg in all_signs]

    return labels, features, latents, signs


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


def train_classifier(features, labels, num_classes, device, epochs=100):
    classifier = Classifier(features.shape[1], num_classes).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(labels).long())
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    classifier.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #print(f'====> Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')
    return classifier


def evaluate_classifier(classifier, features, labels, device, max_k=5, batch_size=128, print=False):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    classifier.eval()
    num_classes = None
    correct_cumsum = None  # will hold cumulative correct counts for k=1..K
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = classifier(inputs)  # [B, C]

        # set K per batch (in case max_k > C)
        if num_classes is None:
            num_classes = outputs.size(1)
            K = min(max_k, num_classes)
            correct_cumsum = torch.zeros(K, device=device, dtype=torch.long)

        # top-K indices
        _, topk_idx = torch.topk(outputs, k=K, dim=1)  # [B, K]

        # matches[b, j] = 1 if the true class is within top-(j+1) for sample b
        matches = topk_idx.eq(targets.unsqueeze(1))                # [B, K] bool
        cum_any = matches.cumsum(dim=1).clamp(max=1).to(torch.long)  # [B, K] 0/1

        # sum over batch → counts for each k
        correct_cumsum += cum_any.sum(dim=0)

        total += targets.size(0)

    # accuracies for k=1..K
    accuracies = (correct_cumsum.float() / total * 100.0).tolist()

    if print:
        for k, acc in enumerate(accuracies, start=1):
            print(f"Top-{k} Accuracy: {acc:.2f}%")

    return accuracies  # list of length K

