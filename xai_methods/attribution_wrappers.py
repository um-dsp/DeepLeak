import torch
import numpy as np
from captum.attr import Attribution
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from alibi.explainers import AnchorImage
from aix360.algorithms.protodash import ProtodashExplainer

def Anchors_attri(model, img_tensor, target=0):
    is_inputs_tuple = False
    if _is_tuple(img_tensor):
        is_inputs_tuple = True
        img_tensor = img_tensor[0]

    model.eval()
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    def predictor(x):
        if len(x.shape) == 5:
            x = x[0]
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float()
            if len(x_tensor.shape) == 3:
                x_tensor = x_tensor.unsqueeze(0)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            output = model(x_tensor.to(device))
            return (output.argmax(dim=1) == target).cpu().numpy().astype(int)

    explainer = AnchorImage(
        predictor,
        img_tensor.shape[1:],
        segmentation_fn='slic',
        segmentation_kwargs={'n_segments': 15, 'compactness': 20, 'sigma': 4}
    )

    img_numpy = img_tensor.cpu().numpy()
    explanation = explainer.explain(img_numpy, p_sample=0.5, batch_size=len(img_numpy), min_samples_start=len(img_numpy))
    mask = explanation.anchor

    if len(mask.shape) == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    mask = torch.tensor(mask, device=img_tensor.device, dtype=torch.float32)
    mask = _format_tensor_into_tuples(mask)
    return _format_output(is_inputs_tuple, mask)

class AnchorsAttribution(Attribution):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def attribute(self, inputs, target=0):
        return Anchors_attri(self.model, inputs, target=target)

class ProtoDashAttribution(Attribution):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def attribute(self, inputs, target=0, train_data=None, num_prototypes=90, num_samples=100):
        num_prototypes = len(inputs[0]) if _is_tuple(inputs) else len(inputs)
        num_samples = num_prototypes + 10
        return self._proto_attr(inputs, train_data, target, num_prototypes, num_samples)

    def _proto_attr(self, img_tensor, train_data, target=0, num_prototypes=90, num_samples=50):
        model = self.model
        is_inputs_tuple = False
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(train_data) > num_samples:
            indices = np.random.choice(len(train_data), num_samples, replace=False)
            train_data = train_data[indices]

        train_tensor = torch.tensor(train_data, device=device)

        if isinstance(img_tensor, tuple):
            is_inputs_tuple = True
            img_tensor = img_tensor[0]

        with torch.no_grad():
            train_embeddings = model(train_tensor)
            query_embedding = model(img_tensor.to(device))
            train_embeddings_np = train_embeddings.cpu().numpy()
            query_embedding_np = query_embedding.cpu().numpy()

        explainer = ProtodashExplainer()
        W, S, _ = explainer.explain(query_embedding_np, train_embeddings_np, m=num_prototypes,
                                    kernelType="Gaussian", sigma=2.0, optimizer='osqp')

        selected_prototypes = torch.tensor(train_data[S], device=device)
        weights = torch.tensor(W.reshape(-1, 1, 1, 1), device=device)
        attribution = weights * selected_prototypes
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        attribution = _format_tensor_into_tuples(attribution)
        return _format_output(is_inputs_tuple, attribution)
