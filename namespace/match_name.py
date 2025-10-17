from transformers import AutoTokenizer
import re
import pandas as pd
import torch
import numpy as np
from .utilities import MatchingMod

class MatchName:
    """
    Class to match names in different formats with different components. 

    Attributes
    ----------
    classification_model : str, default "SamFrederick/namematch500k"
        A classification model to predict whether names match from HuggingFaceHub.
    device : str, optional, default "cpu"
        Uses cpu for model predictions by default. GPU computations may be faster if 
        available (e.g., 'mps' on Apple Silicon or 'cuda').
    tokenizer : str, optional, default "roberta-large"
        The tokenizer to use, compatible with classification_model.
    filter : FilterName, optional, default None
        An object of class FilterName to first filter datasets to top-k possible matches.

    Examples
    -------
    Basic name matching on Apple Silicon.

    >>> matcher = MatchName(classification_model = 'SamFrederick/namematch2m', device = 'mps')

    Filtering names prior to matching.

    >>> filterer = FilterName(embedding_model = 'SamFrederick/namespace1m', k = 3, device = 'mps')
    >>> matcher = MatchName(classification_model = 'SamFrederick/namematch2m', device = 'mps', filter = filterer)

    """
    def __init__(self, classification_model = 'SamFrederick/namematch500k', **kwargs):
        self.device = kwargs.get('device', 'cpu')
        self.filter_match = kwargs.get('filter', None)
        self.tokenizer = kwargs.get('tokenizer', 'roberta-large')
        self.classification_model = classification_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.model = MatchingMod.from_pretrained(
            classification_model,
            device = self.device
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_match(
        self, 
        name1: list[str], 
        name2: list[str]
    ) -> list[float]:
        """
        Method to predict whether names match one another.

        Parameters
        ----------
        name1 : list[str]
            A string or list of strings containing the left set of names.
        name2 : list[str]
            A string or list of strings containing the right set of names to match to name1.
        
        Returns 
        -------
        list
            A list of probabilities that the names match.

        Examples
        --------
        >>> name1 = 'Jon Smith'
        >>> name2 = 'Smith, Jonathan, Jr.
        >>> matcher = MatchName()
        >>> matcher.predict_match(name1, name2)

        """
        if isinstance(name1, list):
            name1 = [re.sub(r' +', ' ', x.lower()).strip() for x in name1]
        else:
            name1 = re.sub(r' +', ' ', name1.lower()).strip()
        
        if isinstance(name2, list):
            name2 = [re.sub(r' +', ' ', x.lower()).strip() for x in name2]
        else:
            name2 = re.sub(r' +', ' ', name2.lower()).strip()

        n1 = self.tokenizer(
            name1,
            add_special_tokens = True,
            truncation = True,
            max_length = 15,
            padding = 'max_length',
            return_tensors = 'pt')

        n2 = self.tokenizer(
            name2,
            add_special_tokens = True,
            truncation = True,
            max_length = 15,
            padding = 'max_length',
            return_tensors = 'pt')
        
        probs = torch.softmax(
            self.model(n1['input_ids'], n1['attention_mask'], n2['input_ids'], n2['attention_mask']), 
            dim = 1
        )

        if probs.shape[0] > 1:
            out = probs[:, 1].cpu().numpy().tolist()
        else:
            out = probs[0,1].item()

        return out

    def _batch_predict(
        self, 
        pairs: list[tuple], 
        batch_size: int = 100
    ) -> list[tuple]:
        n1 = [x[0] for x in pairs]
        n2 = [x[1] for x in pairs]

        out = []
        for x in range(0, len(pairs), batch_size):
            out.extend(self.predict_match(n1[x:(x+batch_size)], n2[x:(x+batch_size)]))
        
        return out

    def all_probs(
        self, 
        names1: list[str], 
        names2: list[str] = None, 
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Method to generate all pairs between one or two lists of names and predict
        whether the pairs are a match.

        Parameters
        ----------
        names1 : list[str]
            A list of strings containing the left set of names.
        names2 : list[str], optional
            A list of strings containing the right set of names to match to names1.

            If names2 is not provided, the function will classify matches between names1 and itself.
        batch_size: int, default 100
            An integer determining the number of name pairs in a prediction batch.

        Returns 
        -------
        DataFrame
            A DataFrame with three columns: name1, name2, and prob, where prob indicates the
             probability that name1 and name2 match.
        
        Examples
        -------
        >>> name1 = ['Jonathan Smith', 'Emily Dickinson', 'Jane Austen', 'Charles Dickens']
        >>> name2 = ['Smith, John, Jr.', 'Emily "Emma" Dickinson', 'Austen, Jane', 'Dickens, Chuck']
        >>> matcher = MatchName()
        >>> matcher.all_probs(name1, name2)

        """
        if not names2:
            names2 = names1
        
        if self.filter_match:
            pairs = self.filter_match.filter_names(names1, names2)
        else:
            pairs = [(n1, n2) for n1 in names1 for n2 in names2]

        probs = self._batch_predict(pairs, batch_size = batch_size)

        out = []
        for i in range(len(pairs)):
            out.append({'name1': pairs[i][0], 'name2': pairs[i][1], 'prob': probs[i]})

        return pd.DataFrame(out)

    def merge_name(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        how: str, 
        left_name_col: str, 
        right_name_col: str, 
        left_exact: list[str] = None, 
        right_exact: list[str] = None, 
        merge_threshold: float = 0.5, 
        return_marginal: bool = False,
        marginal: list[float] = [0.1, 0.9], 
        batch_size: int = 100, 
        crosswalk: bool = False
    ) -> pd.DataFrame:
        """
        Method to fuzzy merge two datasets containing names in different formats. 

        Parameters
        ----------
        df1 : DataFrame
            The left DataFrame to merge.
        df2 : DataFrame
            The right DataFrame to merge.
        how : str, ['inner', 'outer', 'left', 'right']
            Type of merge to perform.
        left_name_col : str
            Name of the name column from df1 on which to fuzzy merge.
        right_name_col : str
            Name of the name column from df2 on which to fuzzy merge.
        left_exact : list[str], optional
            Name(s) of columns from df1 on which to exact match, if desired.
        right_exact : list[str], optional
            Name(s) of columns from df2 on which to exact match, if desired.
        
            If left_exact is provided but right_exact is not, it is assumed that the
            left_exact columns are the same as right_exact.
        merge_threshold : float, default 0.5
            Return merges with name match probabilities at or above this threshold.
        return_marginal : bool, default False
            Return name matches with "marginal" probabilities for further checking.
        marginal : list[float], default [0.1, 0.9]
            A list containing the lower and upper limits of what to consider marginal matches.
        batch_size : int, default 100
            An integer determining the number of name pairs in a prediction batch.
        crosswalk : bool, default False
            A boolean indicating whether or not to return a crosswalk (subsets to only merge columns)

        Returns 
        -------
        DataFrame
            A DataFrame containing the merged DataFrames, including a prob column, 
            indicating the probability that the name columns match.

            If return_marginal is True, this will return a second DataFrame with 3 columns
             (name1, name2, prob), including only the "marginal" matches.
        
        Examples
        -------
        >>> name1 = pd.DataFrame({'name1': ['Jonathan Smith', 'Emily Dickinson', 'Jane Austen', 'Charles Dickens']})
        >>> name2 = pd.DataFrame({'name2': ['Smith, John, Jr.', 'Emily "Emma" Dickinson', 'Austen, Jane', 'Dickens, Chuck']})
        >>> matcher = MatchName()
        >>> matcher.merge_name(name1, name2, how = 'left', left_name_col = 'name1', right_name_col='name2')
        """
        if left_name_col == right_name_col:
            df2 = df2.rename(columns = {right_name_col: right_name_col + '_right'})
            right_name_col += '_right'

        if not left_exact and not right_exact:
            left = df1.copy()
            right = df2.copy()
            p = self.all_probs(
                left.loc[:, left_name_col].unique().tolist(), 
                right.loc[:, right_name_col].unique().tolist(), 
                batch_size = batch_size
            )
            p = p.rename(columns = {'name1': left_name_col, 'name2': right_name_col})
            if return_marginal:
                marginal_out = p[(p.prob>=marginal[0]) & (p.prob<=marginal[1])]
            p = p[p.prob>=merge_threshold]
            left = left.merge(p, on = left_name_col, how = 'left')
            right = right.merge(p, on = right_name_col, how = 'left')
            out = left.merge(right, on = [left_name_col, right_name_col, 'prob'], how = how)
            if return_marginal: 
                return out, marginal_out
            return out
        elif not left_exact:
            left_exact = right_exact
        elif not right_exact:
            right_exact = left_exact
            
        if how == 'left': 
            frame = df1.copy().loc[:, left_exact].drop_duplicates().reset_index(drop = True)
        elif how == 'right': 
            frame = df2.copy().loc[:, right_exact].drop_duplicates().reset_index(drop = True)
        elif how == 'inner': 
            frame = df1.copy().loc[:, left_exact].drop_duplicates()
            frame = frame.merge(df2.loc[:, right_exact].drop_duplicates(), how = 'inner', left_on = left_exact, right_on = right_exact).drop_duplicates().reset_index(drop = True)
        elif how == 'outer': 
            frame = df1.copy().loc[:, left_exact].drop_duplicates()
            frame = frame.merge(df2.loc[:, right_exact].drop_duplicates(), how = 'outer', left_on = left_exact, right_on = right_exact).drop_duplicates().reset_index(drop = True)

        if crosswalk:
            df1 = df1.loc[:, [left_name_col] + left_exact]
            df2 = df2.loc[:, [right_name_col] + right_exact]

        out = []
        marginal_out = []
        for i in range(len(frame)):
            if how !='right': 
                left = frame.iloc[[i]].merge(df1, how = 'left', left_on = left_exact, right_on = left_exact, indicator = True)
            else:
                left = frame.iloc[[i]].merge(df1, how = 'left', left_on = right_exact, right_on = left_exact, indicator = True)
            left = left[left._merge=='both'].drop('_merge', inplace = False, axis = 1)
            if how!='left':
                right = frame.iloc[[i]].merge(df2, how = 'left', left_on = right_exact, right_on = right_exact, indicator = True)
            else:
                right = frame.iloc[[i]].merge(df2, how = 'left', left_on = left_exact, right_on = right_exact, indicator = True)
            right = right[right._merge=='both'].drop('_merge', inplace = False, axis = 1)
            if len(right)==0 and how in ['left', 'outer']:
                out.append(left)
            elif len(left) == 0 and how in ['right', 'outer']: 
                out.append(right)
            elif int(np.min([len(right), len(left)])) == 0:
                continue
            else:
                if int(np.max([len(left), len(right)])) ==1:
                    n1 = left.loc[:, left_name_col].unique().tolist()[0]
                    n2 = right.loc[:, right_name_col].unique().tolist()[0]
                    p = pd.DataFrame({
                        'name1': n1, 
                        'name2': n2
                        }, index = [0])
                    p['prob'] = self.predict_match(n1, n2)
                else:
                    p = self.all_probs(
                        left.loc[:, left_name_col].unique().tolist(), 
                        right.loc[:, right_name_col].unique().tolist(), 
                        batch_size = batch_size
                    )
                p = p.rename(columns = { 'name1':left_name_col, 'name2': right_name_col})
                
                if return_marginal:
                    marginal_out.append(p[(p.prob>=marginal[0]) & (p.prob<=marginal[1])])
                p = p[p.prob>=merge_threshold]
                left = left.merge(p, on = left_name_col, how = 'left')
                right = right.merge(p, on = right_name_col, how = 'left')


                lm = rm = [left_name_col, right_name_col, 'prob'] 
                if how == 'left':
                    rm += left_exact
                elif how == 'right': 
                    lm += right_exact
                else:
                    lm += left_exact + right_exact
                    rm += left_exact + right_exact


                o = left.merge(
                    right, 
                    left_on = lm, 
                    right_on = rm, 
                    how = how
                )
                out.append(o)

        out = pd.concat(out).reset_index(drop = True)
        if return_marginal:
            marginal_out = pd.concat(marginal_out).reset_index(drop = True)
            return out, marginal_out

        return out

    def dedupe_name(
        self, 
        df: pd.DataFrame, 
        name_col: str, 
        exact: list[str] = None, 
        merge_threshold: float = 0.5, 
        return_marginal: bool = False,
        marginal: list[float] = [0.1, 0.9], 
        batch_size: int = 100
    )-> pd.DataFrame:
        """
        Method to deduplicate the names in a dataset. 

        This uses the length of the different name options as a heuristic and selects the longest
        name version. 

        Parameters
        ----------
        df : DataFrame
            The DataFrame to deduplicate.
        name_col : str
            Name of the name column from df to deduplicate.
        exact : list[str], optional
            Name(s) of columns from df on which to exact match, if desired.
        merge_threshold : float, default 0.5
            Return merges with name match probabilities at or above this threshold.
        return_marginal : bool, default False
            Return name matches with "marginal" probabilities for further checking.
        marginal : list[float], default [0.1, 0.9]
            A list containing the lower and upper limits of what to consider marginal matches.
        batch_size : int, default 100
            An integer determining the number of name pairs in a prediction batch.

        Returns 
        -------
        DataFrame
            A DataFrame containing the deduplicated data, with the deduplicated key in the `full_name` column.

            If return_marginal is True, this will return a second DataFrame with 3 columns
             (name1, name2, prob), including only the "marginal" matches.
        
        Examples
        --------
        >>> name1 = pd.DataFrame({'name1': [
                'Jonathan Smith', 'Emily Dickinson',
                'Emily "Emma" Dickinson',  'Jane Austen', 
                'Charles Dickens','Dickens, Chuck']})
        >>> matcher = MatchName()
        >>> matcher.dedupe_name(name1, name_col = 'name1')
        """

        if not exact:
            p = self.all_probs(
                df.loc[:, name_col].unique().tolist(), 
                df.loc[:, name_col].unique().tolist(), 
                batch_size = batch_size
            )
            if return_marginal:
                marginal_out = p[(p.prob>=marginal[0]) & (p.prob<=marginal[1])]
            p = p[p.prob>=merge_threshold]

            tmp = {}
            for i in range(len(p)):
                if p.name1.iloc[i] in tmp:
                    tmp[p.name1.iloc[i]].append(p.name2.iloc[i])
                else:
                    tmp[p.name1.iloc[i]] = [p.name2.iloc[i]]
            for k,v in tmp.items():
                tmp[k] = v + [y for x in v for y in tmp[x]]
                tmp[k].append(k)
                tmp[k] = sorted(list(set(tmp[k])))
                inx = [len(x) for x in tmp[k]]
                tmp[k] = tmp[k][inx.index(max(inx))]
            tmp = pd.DataFrame([{name_col: k, 'full_name': v} for k,v in tmp.items()])

            out = df.merge(
                tmp, 
                how = 'left', 
                on = name_col,
                validate = 'one_to_one'
            )

            if return_marginal: 
                return out, marginal_out
            return out
            
        frame = df.copy().loc[:, exact].drop_duplicates()

        out = []
        marginal_out = []
        for i in range(len(frame)):
            mn = frame.iloc[[i]].merge(df, how = 'left', on = exact)
            if len(mn)==1:
                mn['full_name'] = mn[name_col].iloc[0]
                out.append(mn)
                continue
            p = self.all_probs(
                mn.loc[:, name_col].unique().tolist(), 
                mn.loc[:, name_col].unique().tolist(), 
                batch_size = batch_size
            )
            if return_marginal:
                marginal_out.append(p[(p.prob>=marginal[0]) & (p.prob<=marginal[1])])
            p = p[p.prob>=merge_threshold]

            tmp = {}
            for i in range(len(p)):
                if p.name1.iloc[i] in tmp:
                    tmp[p.name1.iloc[i]].append(p.name2.iloc[i])
                else:
                    tmp[p.name1.iloc[i]] = [p.name2.iloc[i]]
            for k,v in tmp.items():
                tmp[k] = v + [y for x in v for y in tmp[x]]
                tmp[k].append(k)
                tmp[k] = sorted(list(set(tmp[k])))
                inx = [len(x) for x in tmp[k]]
                tmp[k] = tmp[k][inx.index(max(inx))]
            tmp = pd.DataFrame([{name_col: k, 'full_name': v} for k,v in tmp.items()])

            out.append(mn.merge(
                tmp, 
                how = 'left', 
                on = name_col,
                validate = 'one_to_one'
            ))

        out = pd.concat(out)
        if return_marginal:
            marginal_out = pd.concat(marginal_out)
            return out, marginal_out

        return out
