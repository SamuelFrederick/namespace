import torch
from .embed_name import EmbedName
import numpy as np


class FilterName:
    """
    Class to filter closest name matches using cosine similarity. 

    Attributes
    ----------
    embedding_model : str, default "SamFrederick/namespace500k"
        A fine-tuned embeddings model from HuggingFaceHub.
    return_sim : bool, default False
        Whether to return cosine similarities in tuple output.
    device : str, optional, default "cpu"
        Uses cpu for model predictions by default. GPU computations may be faster if available (e.g., 'mps' on Apple Silicon or 'cuda').
    tokenizer : str, optional, default "roberta-large"
        The tokenizer to use, compatible with embedding_model.
    k : int, optional, default None
        The number of nearest neighbors to filter to.

        Must supply either k or threshold.
    threshold : float, optional, default None
        Threshold for determining close matches.

        Must supply either k or threshold

    Examples
    -------
    Return top 3 nearest matches using Apple Silicon.

    >>> filterer = FilterName(embedding_model = 'SamFrederick/namespace1m', k = 3, device = 'mps')

    Return matches with cosine similarity greater than or equal to 0.5.

    >>> filterer = FilterName(embedding_model = 'SamFrederick/namespace1m', threshold = 0.5)

    """
    def __init__(
        self, 
        embedding_model: str = 'SamFrederick/namespace500k', 
        return_sim: bool = False, 
        tokenizer: str = 'roberta-large', 
        device: str = 'cpu',
        k: int = None,
        threshold: int = None
    ):
        self.emb_mod = EmbedName(
            embedding_model = embedding_model, 
            tokenizer = tokenizer,
            device = device
        )
        self.k = k
        self.thresh = threshold
        self.return_sim = return_sim
        
    def filter_names(
        self,
        name1: list[str], 
        name2: list[str]
    ) -> list[tuple]:
        """
        Method to return only close matches of names using cosine similarity.

        Parameters
        ----------
        name1 : list[str]
            A list of strings containing the left set of names.
        name2 : list[str]
            A list of strings containing the right set of names to match to name1.

        Returns 
        -------
        list[tuple]
            A list of tuples of the form [(name1, closename2),...].
        
        Examples
        -------
        Filter to 3 nearest neighbors.

        >>> name1 = ['Jonathan Smith', 'Emily Dickinson', 'Jane Austen', 'Charles Dickens']
        >>> name2 = ['Smith, John, Jr.', 'Emily "Emma" Dickinson', 'Austen, Jane', 'Dickens, Chuck']
        >>> filter = FilterName(k = 3)
        >>> filter.filter_names(name1, name2)

        Filter to possible matches with cosine similarity at or above 0.75.

        >>> name1 = ['Jonathan Smith', 'Emily Dickinson', 'Jane Austen', 'Charles Dickens']
        >>> name2 = ['Smith, John, Jr.', 'Emily "Emma" Dickinson', 'Austen, Jane', 'Dickens, Chuck']
        >>> filter = FilterName(threshold = 0.75)
        >>> filter.filter_names(name1, name2)

        """
        mat = self.emb_mod.cosine_similarity(name1, name2)
        if self.k:
            if self.k<=len(name2):
                out = torch.topk(mat, k = self.k, dim = 1)
            else:
                all_out = [(n1, n2) for n1 in name1 for n2 in name2]
                return all_out

            n2 = np.array(name2)
            if self.k == 1:
                n2 = n2.flatten().tolist()
                if self.return_sim:
                    all_out = [(name1[i], n2[out.indices[i]], float(out.values[i])) for i in range(len(name1))]
                else:
                    all_out = [(name1[i], n2[out.indices[i]]) for i in range(len(name1))]
            else:
                all_out = []
                for i, val in enumerate(name1):
                    if self.return_sim:
                        all_out.extend([(val, str(n2[out.indices[i]][v2]), float(out.values[i][v2])) for v2 in range(self.k)])
                    else:
                        all_out.extend([(val, str(v2)) for v2 in n2[out.indices[i]]])
        elif self.thresh:
            out = torch.argwhere(mat>=self.thresh)
            if self.return_sim:
                all_out = [(name1[x[0]], name2[x[1]], float(mat[x[0], x[1]])) for x in out]
            else:
                all_out = [(name1[x[0]], name2[x[1]]) for x in out]

        return all_out
