from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pytorch_lightning
from scipy.sparse import csr_matrix
import torch
from torchmetrics import Metric
from torchmetrics.functional import auroc
from tqdm.auto import tqdm
from typing_extensions import Literal

import collie
from collie.interactions import ExplicitInteractions, Interactions, InteractionsDataLoader
from collie.model import BasePipeline


def _get_user_item_pairs(model: BasePipeline,
                         user_or_item_ids: Union[np.array, torch.tensor],
                         n_items_or_users: int,
                         device: Union[str, torch.device]) -> Tuple[torch.tensor, torch.tensor]:
    """
    Create tensors pairing each input user ID with each item ID or vice versa.

    Parameters
    ----------
    model: collie.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    user_or_item_ids: np.array or torch.tensor, 1-d
        Iterable[int] of users or items to score
    n_items_or_users: int
        Number of items or users in the training data
    device: string
        Device to store tensors on

    Returns
    -------
    users: torch.tensor, 1-d
        Tensor with ``n_items_or_users`` copies of each user ID
    items: torch.tensor, 1-d
        Tensor with ``len(user_or_item_ids)`` copies of each item ID

    Example
    -------
    .. code-block:: python

        >>> user_ids = np.array([10, 11, 12])
        >>> n_items = 4
        >>> user, item = _get_user_item_pairs(user_ids: user_ids, n_items: 4, device: 'cpu'):
        >>> user
        np.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12])
        >>> item
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    """
    try:
        negative_sample_type = model.train_loader.negative_sample_type
    except AttributeError:  # explicit models do not have ``negative_sample_type``
        negative_sample_type = 'item'

    # Added because sometimes we call this function with ``n_items_or_users`` as ``np.int64``
    # type which breaks ``repeat_interleave``.
    if isinstance(n_items_or_users, np.int64):
        n_items_or_users = n_items_or_users.item()

    aabb_tensor = torch.tensor(
        user_or_item_ids,
        dtype=torch.int64,
        requires_grad=False,
        device=device,
    ).repeat_interleave(n_items_or_users)

    cdcd_tensor = torch.arange(
        start=0,
        end=n_items_or_users,
        requires_grad=False,
        device=device,
    ).repeat(len(user_or_item_ids))

    if negative_sample_type == 'item':
        users = aabb_tensor
        items = cdcd_tensor

    elif negative_sample_type == 'user':
        users = cdcd_tensor
        items = aabb_tensor

    return users, items


def get_preds(model: BasePipeline,
              user_or_item_ids: Union[np.array, torch.tensor],
              n_items_or_users: int,
              device: Union[str, torch.device]) -> torch.tensor:
    """
    Returns a ``n_users x n_items`` tensor with the item IDs of recommended products for each user
    ID or a ``n_items x n_users`` tensor with the user IDs of recommended products for each item
    ID depending on the ``model.train_loader.negative_sample_type``

    Parameters
    ----------
    model: collie.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    user_or_item_ids: np.array or torch.tensor
        Iterable[int] of users or items to score
    n_items_or_users: int
        Number of items or users in the training data
    device: string
        Device torch should use

    Returns
    -------
    predicted_scores: torch.tensor
        Tensor of shape ``n_users x n_items`` or ``n_items x n_users``

    """
    user, item = _get_user_item_pairs(model, user_or_item_ids, n_items_or_users, device)

    with torch.no_grad():
        predicted_scores = model(user, item)

    return predicted_scores.view(-1, n_items_or_users)


def _get_labels(targets: csr_matrix,
                user_or_item_ids: Union[np.array, torch.tensor],
                preds: Union[np.array, torch.tensor],
                device: str,
                negative_sample_type: Literal['item', 'user'] = 'item') -> torch.tensor:
    """
    Returns a binary array indicating which of the recommended items or users are in each user's or
    item's target set, respectively.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_or_item_ids: np.array or torch.tensor
        Users or items corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Top ``k`` IDs to recommend to each user or item of shape (n_users_or_items x k)
    device: string
        Device torch should use
    negative_sample_type: str
        Type of negative sampling the Interaction matrix, ``targets``, used

    Returns
    -------
    labels: torch.tensor
        Tensor with the same dimensions as input ``preds``

    """
    if negative_sample_type == 'item':
        return torch.tensor(
            (targets[user_or_item_ids[:, None], np.array(preds.detach().cpu())] > 0)
            .astype('double')
            .toarray(),
            requires_grad=False,
            device=device,
        )
    elif negative_sample_type == 'user':
        return torch.tensor(
            (targets[np.array(preds.detach().cpu()), user_or_item_ids[:, None]] > 0)
            .astype('double')
            .toarray(),
            requires_grad=False,
            device=device,
        )


def mapk(targets: csr_matrix,
         user_or_item_ids: Union[np.array, torch.tensor],
         preds: Union[np.array, torch.tensor],
         k: int = 10,
         negative_sample_type: Literal['item', 'user'] = 'item') -> float:
    """
    Calculate the mean average precision at K (MAP@K) score for each user or item.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_or_item_ids: np.array or torch.tensor
        Users or items corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item or
        tensor of shape (n_items x n_users) with each item's scores for each user
    k: int
        Number of recommendations to consider per user or item
    negative_sample_type: str
        Type of negative sampling the Interaction matrix, ``targets``, used

    Returns
    -------
    mapk_score: float

    """
    device = preds.device
    n_users_or_items = preds.shape[0]

    try:
        predictions = preds.topk(k, dim=1).indices
    except RuntimeError as e:
        raise ValueError(
            f'Ensure ``k`` ({k}) is less than the number of items ({preds.shape[1]}):', str(e)
        )

    topk_labeled = _get_labels(targets, user_or_item_ids, predictions, device, negative_sample_type)
    accuracy = topk_labeled.int()

    weights = (
        1.0 / torch.arange(
            start=1,
            end=k+1,
            dtype=torch.float64,
            requires_grad=False,
            device=device
        )
    ).repeat(n_users_or_items, 1)

    if negative_sample_type == 'item':
        denominator = torch.min(
            torch.tensor(k, device=device, dtype=torch.int).repeat(len(user_or_item_ids)),
            torch.tensor(targets[user_or_item_ids].getnnz(axis=1), device=device)
        )
    elif negative_sample_type == 'user':
        denominator = torch.min(
            torch.tensor(k, device=device, dtype=torch.int).repeat(len(user_or_item_ids)),
            torch.tensor(targets[:, user_or_item_ids].getnnz(axis=1), device=device)
        )

    res = ((accuracy * accuracy.cumsum(axis=1) * weights).sum(axis=1)) / denominator
    res[torch.isnan(res)] = 0

    return res.mean().item()


def mrr(targets: csr_matrix,
        user_or_item_ids: Union[np.array, torch.tensor],
        preds: Union[np.array, torch.tensor],
        k: Optional[Any] = None,
        negative_sample_type: Literal['item', 'user'] = 'item') -> float:
    """
    Calculate the mean reciprocal rank (MRR) of the input predictions.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_or_item_ids: np.array or torch.tensor
        Users or items corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item or
        tensor of shape (n_items x n_users) with each item's scores for each user
    k: Any
        Ignored, included only for compatibility with ``mapk``
    negative_sample_type: str
        Type of negative sampling the Interaction matrix, ``targets``, used

    Returns
    -------
    mrr_score: float

    """
    device = preds.device
    predictions = preds.topk(preds.shape[1], dim=1).indices
    labeled = _get_labels(targets, user_or_item_ids, predictions, device, negative_sample_type)

    if negative_sample_type == 'item':
        # weighting each 0/1 by position so that topk returns index of *first* postive result
        position_weight = 1.0/(
            torch.arange(1, targets.shape[1] + 1, device=device)
            .repeat(len(user_or_item_ids), 1)
            .float()
        )
    elif negative_sample_type == 'user':
        # weighting each 0/1 by position so that topk returns index of *first* postive result
        position_weight = 1.0/(
            torch.arange(1, targets.shape[0] + 1, device=device)
            .repeat(len(user_or_item_ids), 1)
            .float()
        )
    labeled_weighted = (labeled.float() * position_weight)

    highest_score, rank = labeled_weighted.topk(k=1)

    reciprocal_rank = 1.0/(rank.float() + 1)
    reciprocal_rank[highest_score == 0] = 0

    return reciprocal_rank.mean().item()


def auc(targets: csr_matrix,
        user_or_item_ids: Union[np.array, torch.tensor],
        preds: Union[np.array, torch.tensor],
        k: Optional[Any] = None,
        negative_sample_type: Literal['item', 'user'] = 'item') -> float:
    """
    Calculate the area under the ROC curve (AUC) for each user or item and average the results.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_or_item_ids: np.array or torch.tensor
        Users or items corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item or
        tensor of shape (n_items x n_users) with each item's scores for each user
    k: Any
        Ignored, included only for compatibility with ``mapk``
    negative_sample_type: str
        Type of negative sampling the Interaction matrix, ``targets``, used

    Returns
    -------
    auc_score: float

    """
    agg = 0
    if negative_sample_type == 'item':
        for i, user_id in enumerate(user_or_item_ids):
            target_tensor = torch.tensor(
                targets[user_id].toarray(),
                device=preds.device,
                dtype=torch.long
            ).view(-1)
            # many models' ``preds`` may be unbounded if a final activation layer is not applied
            # we have to normalize ``preds`` here to avoid a ``ValueError`` stating that ``preds``
            # should be probabilities, but values were detected outside of [0,1] range
            auc = auroc(torch.sigmoid(preds[i, :]), target=target_tensor, pos_label=1)
            agg += auc
    elif negative_sample_type == 'user':
        for i, item_id in enumerate(user_or_item_ids):
            target_tensor = torch.tensor(
                targets[:, item_id].toarray(),
                device=preds.device,
                dtype=torch.long
            ).view(-1)
            # many models' ``preds`` may be unbounded if a final activation layer is not applied
            # we have to normalize ``preds`` here to avoid a ``ValueError`` stating that ``preds``
            # should be probabilities, but values were detected outside of [0,1] range
            auc = auroc(torch.sigmoid(preds[i, :]), target=target_tensor, pos_label=1)
            agg += auc

    return (agg/len(user_or_item_ids)).item()


def evaluate_in_batches(
    metric_list: Iterable[Callable[
        [csr_matrix, Union[np.array, torch.tensor], Union[np.array, torch.tensor], Optional[int]],
        float
    ]],
    test_interactions: collie.interactions.Interactions,
    model: collie.model.BasePipeline,
    k: int = 10,
    batch_size: int = 20,
    logger: pytorch_lightning.loggers.base.LightningLoggerBase = None,
    verbose: bool = True,
) -> List[float]:
    """
    Evaluate a model with potentially several different metrics.

    Memory constraints require that most test sets will need to be evaluated in batches. This
    function handles the looping and batching boilerplate needed to properly evaluate the model
    without running out of memory.

    Parameters
    ----------
    metric_list: list of functions
        List of evaluation functions to apply. Each function must accept keyword arguments:

        * ``targets``

        * ``user_ids``

        * ``preds``

        * ``k``

    test_interactions: collie.interactions.Interactions
        Interactions to use as labels
    model: collie.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    k: int
        Number of recommendations to consider per user or item. This is ignored by some metrics
    batch_size: int
        Number of users or items to score in a single batch. For best efficiency, this number
        should be as high as possible without running out of memory
    logger: pytorch_lightning.loggers.base.LightningLoggerBase
        If provided, will log outputted metrics dictionary using the ``log_metrics`` method with
        keys being the string representation of ``metric_list`` and values being
        ``evaluation_results``. Additionally, if ``model.hparams.num_epochs_completed`` exists, this
        will be logged as well, making it possible to track metrics progress over the course of
        model training
    verbose: bool
        Display progress bar and print statements during function execution

    Returns
    -------
    evaluation_results: list
        List of floats, with each metric value corresponding to the respective function passed in
        ``metric_list``

    Examples
    --------
    .. code-block:: python

        from collie.metrics import auc, evaluate_in_batches, mapk, mrr


        map_10_score, mrr_score, auc_score = evaluate_in_batches(
            metric_list=[mapk, mrr, auc],
            test_interactions=test,
            model=model,
        )

        print(map_10_score, mrr_score, auc_score)

    """
    if not isinstance(test_interactions, Interactions):
        raise ValueError(
            '``test_interactions`` must be of type ``Interactions``, not '
            f'{type(test_interactions)}. Try using ``explicit_evaluate_in_batches`` instead.'
        )

    device = _get_evaluate_in_batches_device(model=model)
    model.to(device)
    model._move_any_external_data_to_device()

    negative_sample_type = test_interactions.negative_sample_type

    if negative_sample_type == 'item':
        test_users_or_items = np.unique(test_interactions.mat.row)
        n_items_or_users = test_interactions.num_items
    elif negative_sample_type == 'user':
        test_users_or_items = np.unique(test_interactions.mat.col)
        n_items_or_users = test_interactions.num_users

    targets = test_interactions.mat.tocsr()

    if len(test_users_or_items) < batch_size:
        batch_size = len(test_users_or_items)

    accumulators = [0] * len(metric_list)

    data_to_iterate_over = range(int(np.ceil(len(test_users_or_items) / batch_size)))
    if verbose:
        data_to_iterate_over = tqdm(data_to_iterate_over)

    for i in data_to_iterate_over:
        user_or_item_range = test_users_or_items[i * batch_size:(i + 1) * batch_size]
        preds = get_preds(model, user_or_item_range, n_items_or_users, device)
        for metric_ind, metric in enumerate(metric_list):
            score = metric(targets=targets,
                           user_or_item_ids=user_or_item_range,
                           preds=preds,
                           k=k,
                           negative_sample_type=negative_sample_type)
            accumulators[metric_ind] += (score * len(user_or_item_range))

    all_scores = [acc_score / len(test_users_or_items) for acc_score in accumulators]

    if logger is not None:
        _log_metrics(model=model,
                     logger=logger,
                     metric_list=metric_list,
                     all_scores=all_scores,
                     verbose=verbose)

    return all_scores[0] if len(all_scores) == 1 else all_scores


def explicit_evaluate_in_batches(
    metric_list: Iterable[Metric],
    test_interactions: collie.interactions.ExplicitInteractions,
    model: collie.model.BasePipeline,
    logger: pytorch_lightning.loggers.base.LightningLoggerBase = None,
    verbose: bool = True,
    **kwargs,
) -> List[float]:
    """
    Evaluate a model with potentially several different metrics.

    Memory constraints require that most test sets will need to be evaluated in batches. This
    function handles the looping and batching boilerplate needed to properly evaluate the model
    without running out of memory.

    Parameters
    ----------
    metric_list: list of ``torchmetrics.Metric``
        List of evaluation functions to apply. Each function must accept arguments for predictions
        and targets, in order
    test_interactions: collie.interactions.ExplicitInteractions
    model: collie.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    batch_size: int
        Number of users to score in a single batch. For best efficiency, this number should be as
        high as possible without running out of memory
    logger: pytorch_lightning.loggers.base.LightningLoggerBase
        If provided, will log outputted metrics dictionary using the ``log_metrics`` method with
        keys being the string representation of ``metric_list`` and values being
        ``evaluation_results``. Additionally, if ``model.hparams.num_epochs_completed`` exists, this
        will be logged as well, making it possible to track metrics progress over the course of
        model training
    verbose: bool
        Display progress bar and print statements during function execution
    **kwargs: keyword arguments
        Additional arguments sent to the ``InteractionsDataLoader``

    Returns
    ----------
    evaluation_results: list
        List of floats, with each metric value corresponding to the respective function passed in
        ``metric_list``

    Examples
    -------------
    .. code-block:: python

        import torchmetrics

        from collie.metrics import explicit_evaluate_in_batches


        mse_score, mae_score = evaluate_in_batches(
            metric_list=[torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()],
            test_interactions=test,
            model=model,
        )

        print(mse_score, mae_score)

    """
    if not isinstance(test_interactions, ExplicitInteractions):
        raise ValueError(
            '``test_interactions`` must be of type ``ExplicitInteractions``, not '
            f'{type(test_interactions)}. Try using ``evaluate_in_batches`` instead.'
        )

    try:
        device = _get_evaluate_in_batches_device(model=model)
        model.to(device)
        model._move_any_external_data_to_device()

        test_loader = InteractionsDataLoader(interactions=test_interactions,
                                             **kwargs)

        data_to_iterate_over = test_loader
        if verbose:
            data_to_iterate_over = tqdm(test_loader)

        for batch in data_to_iterate_over:
            users, items, ratings = batch

            # move data to batch before sending to model
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.cpu()

            preds = model(users, items)

            for metric in metric_list:
                metric(preds.cpu(), ratings)

        all_scores = [metric.compute() for metric in metric_list]

        if logger is not None:
            _log_metrics(model=model,
                         logger=logger,
                         metric_list=metric_list,
                         all_scores=all_scores,
                         verbose=verbose)

        return all_scores[0] if len(all_scores) == 1 else all_scores
    finally:
        for metric in metric_list:
            metric.reset()


def _get_evaluate_in_batches_device(model: BasePipeline):
    device = getattr(model, 'device', None)

    if torch.cuda.is_available() and str(device) == 'cpu':
        warnings.warn('CUDA available but model device is set to CPU - is this desired?')

    if device is None:
        if torch.cuda.is_available():
            warnings.warn(
                '``model.device`` attribute is ``None``. Since GPU is available, putting model on '
                'GPU.'
            )
            device = 'cuda:0'
        else:
            device = 'cpu'

    return device


def _log_metrics(model: BasePipeline,
                 logger: pytorch_lightning.loggers.base.LightningLoggerBase,
                 metric_list: List[Union[Callable[..., Any], Metric]],
                 all_scores: List[float],
                 verbose: bool):
    try:
        step = model.hparams.get('num_epochs_completed')
    except torch.nn.modules.module.ModuleAttributeError:
        # if, somehow, there is no ``model.hparams`` attribute, this shouldn't fail
        step = None

    try:
        metrics_dict = dict(zip([x.__name__ for x in metric_list], all_scores))
    except AttributeError:
        metrics_dict = dict(zip([type(x).__name__ for x in metric_list], all_scores))

    if verbose:
        print(f'Logging metrics {metrics_dict} to ``logger``...')

    logger.log_metrics(metrics=metrics_dict, step=step)
