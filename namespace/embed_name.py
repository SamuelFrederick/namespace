from transformers import AutoTokenizer, RobertaModel
import torch
from .utilities import mean_pool, mat_cos_sim

class EmbedName:
    """
    Class to retrieve embeddings for names. 

    Attributes
    ----------
    embedding_model : str, default "SamFrederick/namespace500k"
        A fine-tuned embeddings model from HuggingFaceHub, should be a RoBERTa model.
    tokenizer : str, optional, default "roberta-large"
        The tokenizer to use, compatible with embedding_model.

    Examples
    --------
    >>> embedder = EmbedName(embedding_model = 'SamFrederick/namespace1m')

    """
    def __init__(self, embedding_model, tokenizer = 'roberta-large', **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = RobertaModel.from_pretrained(embedding_model)
        self.model.eval()

    @torch.no_grad()
    def embed_name(
        self, 
        name: list[str]
    ) -> torch.tensor:
        """
        Method to return embeddings for name(s).

        Parameters
        ----------
        name : list[str]
            A list of strings containing the names for which embeddings are desired.

        Returns 
        -------
        torch.tensor
            A torch.tensor containing mean-pooled name embeddings.
        
        Examples
        -------
        >>> name = 'John Smith'
        >>> embedder = EmbedName(embedding_model = 'SamFrederick/namespace1m')
        >>> embedder.embed_name(name)

        """
        tokens = self.tokenizer(
            name,
            add_special_tokens = True,
            truncation = True,
            max_length = 15,
            padding = 'max_length',
            return_tensors = 'pt')
        
        emb = self.model(tokens['input_ids'], tokens['attention_mask'])
        
        emb = mean_pool(emb.last_hidden_state, tokens['attention_mask'])

        return emb
    
    @torch.no_grad()
    def cosine_similarity(
        self, 
        name1: list[str], 
        name2: list[str]
    ) -> torch.tensor:
        """
        Method to return cosine similarities for name embeddings.

        Parameters
        ----------
        name1 : list[str]
            A list of strings containing the first set of names to compare.
        name2 : list[str]
            A list of string containing the second set of names to compare.

        Returns 
        -------
        torch.tensor
            A torch.tensor containing cosine similarities between names.
        
        Examples
        -------
        >>> embedder = EmbedName(embedding_model = 'SamFrederick/namespace1m')
        >>> embedder.cosine_similarity('John Smith', 'Smith, Jonathan')

        """
        n1 = self.embed_name(name1)
        n2 = self.embed_name(name2)
        if isinstance(name1, list) or isinstance(name2, list):
            return mat_cos_sim(n1, n2)
            
        else:
            cos_sim = torch.nn.CosineSimilarity()

            return cos_sim(n1, n2)