from transformers import RobertaModel
import torch
from huggingface_hub import PyTorchModelHubMixin

def mean_pool(toks, mask):

    input_mask_expanded = mask.unsqueeze(-1).expand(toks.size()).to(toks.dtype)
        
    sum_embeddings = torch.sum(toks * input_mask_expanded, 1)

    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def mat_cos_sim(v1, v2):

    n1 = v1 / v1.norm(dim = 1)[:, None]
    n2 = v2 / v2.norm(dim = 1)[:, None]

    return torch.matmul(n1, n2.transpose(0,1))


class MatchingMod(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get('device')
        self.roberta = RobertaModel.from_pretrained(kwargs.get('model_config'))
        self.classifier = torch.nn.Linear(1024*3, 2)

    def forward(self, anchor_ids, anchor_mask, positive_ids, positive_mask):

        anchor_ids = anchor_ids.to(self.device)
        anchor_mask = anchor_mask.to(self.device)

        positive_ids = positive_ids.to(self.device)
        positive_mask = positive_mask.to(self.device)

        anchor = self.roberta(anchor_ids, attention_mask = anchor_mask).last_hidden_state
        positive = self.roberta(positive_ids, attention_mask = positive_mask).last_hidden_state

        anchor = mean_pool(anchor, anchor_mask)
        positive = mean_pool(positive, positive_mask)
        ap_diff = torch.abs(anchor - positive)

        emb = torch.cat([anchor, positive, ap_diff], dim = 1)
        scores = self.classifier(emb)

        return scores


