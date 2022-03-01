from collections import defaultdict
import functools
import operator
from typing import Any, Iterable, Optional, Tuple, Union

from joblib import delayed, Parallel
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

from collie.interactions import (BaseInteractions,
                                 ExplicitInteractions,
                                 HDF5Interactions,
                                 Interactions)
from collie.utils import get_random_seed


def _subset_interactions(interactions: BaseInteractions,
                         idxs: Iterable[int]) -> Union[ExplicitInteractions, Interactions]:
    idxs = np.array(idxs)

    coo_mat = coo_matrix(
        (interactions.mat.data[idxs], (interactions.mat.row[idxs], interactions.mat.col[idxs])),
        shape=(interactions.num_users, interactions.num_items)
    )

    # disable all ``Interactions`` checks for the data splits, since we assume the initial
    # ``Interactions`` object would have these checks already applied prior to the data split
    if isinstance(interactions, Interactions):
        return Interactions(
            mat=coo_mat,
            negative_sample_type=interactions.negative_sample_type,
            num_negative_samples=interactions.num_negative_samples,
            allow_missing_ids=True,
            remove_duplicate_user_item_pairs=False,
            num_users=interactions.num_users,
            num_items=interactions.num_items,
            check_num_negative_samples_is_valid=False,
            max_number_of_samples_to_consider=interactions.max_number_of_samples_to_consider,
            seed=interactions.seed,
        )
    else:
        return ExplicitInteractions(
            mat=coo_mat,
            allow_missing_ids=True,
            remove_duplicate_user_item_pairs=False,
            num_users=interactions.num_users,
            num_items=interactions.num_items,
        )


def random_split(interactions: BaseInteractions,
                 val_p: float = 0.0,
                 test_p: float = 0.2,
                 processes: Optional[Any] = None,
                 seed: Optional[int] = None) -> Tuple[BaseInteractions, ...]:
    """
    Randomly split interactions into training, validation, and testing sets.

    This split does NOT guarantee that every user will be represented in both the training and
    testing datasets. While much faster than ``stratified_split``, it is not the most representative
    data split because of this.

    Note that this function is not supported for ``HDF5Interactions`` objects, since this data split
    implementation requires all data to fit in memory. A data split for large datasets should be
    done using a big data processing technology, like Spark.

    Parameters
    ----------
    interactions: collie.interactions.BaseInteractions
    val_p: float
        Proportion of data used for validation
    test_p: float
        Proportion of data used for testing
    processes: Any
        Ignored, included only for compatability with ``stratified_split`` API
    seed: int
        Random seed for splits

    Returns
    -------
    train_interactions: collie.interactions.BaseInteractions
        Training data of size proportional to ``1 - val_p - test_p``
    validate_interactions: collie.interactions.BaseInteractions
        Validation data of size proportional to ``val_p``, returned only if ``val_p > 0``
    test_interactions: collie.interactions.BaseInteractions
        Testing data of size proportional to ``test_p``

    Examples
    --------
    .. code-block:: python

        >>> interactions = Interactions(...)
        >>> len(interactions)
        100000
        >>> train, test = random_split(interactions)
        >>> len(train), len(test)
        (80000, 20000)

    """
    assert not isinstance(interactions, HDF5Interactions), (
        '``HDF5Interactions`` data type not supported in cross validation splits!'
    )

    _validate_val_p_and_test_p(val_p=val_p, test_p=test_p)

    if seed is None:
        seed = get_random_seed()

    np.random.seed(seed)

    shuffle_indices = np.arange(len(interactions))
    np.random.shuffle(shuffle_indices)

    interactions = _subset_interactions(interactions=interactions,
                                        idxs=shuffle_indices)

    validate_and_test_p = val_p + test_p
    validate_cutoff = int((1.0 - validate_and_test_p) * len(interactions))
    test_cutoff = int((1.0 - test_p) * len(interactions))

    train_idxs = np.arange(validate_cutoff)
    validate_idxs = np.arange(validate_cutoff, test_cutoff)
    test_idxs = np.arange(test_cutoff, len(interactions))

    train_interactions = _subset_interactions(interactions=interactions,
                                              idxs=train_idxs)
    test_interactions = _subset_interactions(interactions=interactions,
                                             idxs=test_idxs)

    if val_p > 0:
        validate_interactions = _subset_interactions(interactions=interactions,
                                                     idxs=validate_idxs)

        return train_interactions, validate_interactions, test_interactions
    else:
        return train_interactions, test_interactions


def stratified_split(interactions: BaseInteractions,
                     val_p: float = 0.0,
                     test_p: float = 0.2,
                     processes: int = -1,
                     seed: Optional[int] = None,
                     force_split: bool = False) -> Tuple[BaseInteractions, ...]:
    """
    Split an ``Interactions`` instance into train, validate, and test datasets in a stratified
    manner such that each user or item appears at least once in each of the datasets. Whether
    the split is on user of item is determined by the input ``Interactions`` instance.

    This split guarantees that every user or item will be represented in the training, validation,
    and testing datasets given they appear in ``interactions`` at least three times. If ``val_p ==
    0``, they will appear in the training and testing datasets given they appear at least two times.
    If a user or item appears fewer than this number of times, a ``ValueError`` will be raised. To
    ignore this error and force users or items with only a single interaction into the training set
    set ``force_split`` to ``True``. To filter users or items with fewer than ``n`` points out, use
    ``collie.utils.remove_users_or_items_with_fewer_than_n_interactions``.  # TODO name this function

    This is computationally more complex than ``random_split``, but produces a more representative
    data split. Note that when ``val_p > 0``, the algorithm will perform the data split twice,
    once to create the test set and another to create the validation set, essentially doubling the
    computational time.

    Note that this function is not supported for ``HDF5Interactions`` objects, since this data split
    implementation requires all data to fit in memory. A data split for large datasets should be
    done using a big data processing technology, like Spark.

    Parameters
    ----------
    interactions: collie.interactions.BaseInteractions
        ``Interactions`` instance containing the data to split
    val_p: float
        Proportion of data used for validation
    test_p: float
        Proportion of data used for testing
    processes: int
        Number of CPUs to use for parallelization. If ``processes == 0``, this will be run
        sequentially in a single list comprehension, else this function uses ``joblib.delayed`` and
        ``joblib.Parallel`` for parallelization. A value of ``-1`` means that all available cores
        will be used
    seed: int
        Random seed for splits
    force_split: bool
        Ignore error raised when a user or item in the dataset has only a single interaction. Normally,
        a ``ValueError`` is raised when this occurs. When ``force_split=True``, however, users or items
        with a single interaction will be placed in the training set and an error will NOT be
        raised

    Returns
    -------
    train_interactions: collie.interactions.BaseInteractions
        Training data of size proportional to ``1 - val_p - test_p``
    validate_interactions: collie.interactions.BaseInteractions
        Validation data of size proportional to ``val_p``, returned only if ``val_p > 0``
    test_interactions: collie.interactions.BaseInteractions
        Testing data of size proportional to ``test_p``

    Examples
    --------
    .. code-block:: python

        >>> interactions = Interactions(...)
        >>> len(interactions)
        100000
        >>> train, test = stratified_split(interactions)
        >>> len(train), len(test)
        (80000, 20000)

    """
    assert not isinstance(interactions, HDF5Interactions), (
        '``HDF5Interactions`` data types not supported in cross validation splits!'
    )

    _validate_val_p_and_test_p(val_p=val_p, test_p=test_p)

    if seed is None:
        seed = get_random_seed()

    train, test = _stratified_split(interactions=interactions,
                                    test_p=test_p,
                                    processes=processes,
                                    seed=seed,
                                    force_split=force_split)

    if val_p > 0:
        train, validate = _stratified_split(interactions=train,
                                            test_p=val_p / (1 - test_p),
                                            processes=processes,
                                            seed=seed,
                                            force_split=force_split)

        return train, validate, test
    else:
        return train, test


def _stratified_split(interactions: BaseInteractions,
                      test_p: float,
                      processes: int,
                      seed: int,
                      force_split: bool) -> Tuple[Interactions, Interactions]:
    if interactions.negative_sample_type == 'item':
        users_or_items = interactions.mat.row
    elif interactions.negative_sample_type == 'user':
        users_or_items = interactions.mat.col
    unique_users_or_items = set(users_or_items)
    # while we should be able to run ``np.where(users == user)[0]`` to find all items each user
    # interacted with, by building up a dictionary to get these values instead, we can achieve the
    # same result in O(N) complexity rather than O(M * N), a nice timesave to have when working with
    # larger datasets
    all_idxs_for_users_or_items_dict = defaultdict(list)
    for idx, user_or_item in enumerate(users_or_items):
        all_idxs_for_users_or_items_dict[user_or_item].append(idx)

    if processes == 0:
        test_idxs = [
            _stratified_split_parallel_worker(interactions=interactions,
                                              idxs_to_split=all_idxs_for_users_or_items_dict[user_or_item],
                                              test_p=test_p,
                                              seed=(seed + user_or_item),
                                              force_split=force_split)
            for user_or_item in unique_users_or_items
        ]
    else:
        # run the function below in parallel for each user
        # by setting the seed to ``seed + user``, we get a balance between reproducability and
        # actual randomness so users with the same number of interactions are not split the exact
        # same way
        test_idxs = Parallel(n_jobs=processes)(
            delayed(_stratified_split_parallel_worker)(interactions,
                                                       all_idxs_for_users_or_items_dict[user_or_item],
                                                       test_p,
                                                       seed + user_or_item,
                                                       force_split)
            for user_or_item in unique_users_or_items
        )

    # reduce the list of lists down to a 1-d list
    test_idxs = functools.reduce(operator.iconcat, test_idxs, [])
    # find all indices not in test set - they are now train
    train_idxs = list(set(range(len(users_or_items))) - set(test_idxs))

    train_interactions = _subset_interactions(interactions=interactions,
                                              idxs=train_idxs)
    test_interactions = _subset_interactions(interactions=interactions,
                                             idxs=test_idxs)

    return train_interactions, test_interactions


def _stratified_split_parallel_worker(interactions: BaseInteractions,
                                      idxs_to_split: Iterable[Any],
                                      test_p: float,
                                      seed: int,
                                      force_split: bool) -> np.array:
    try:
        _, test_idxs = train_test_split(idxs_to_split,
                                        test_size=test_p,
                                        random_state=seed,
                                        shuffle=True,
                                        stratify=np.ones_like(idxs_to_split))
    except ValueError as ve:
        if 'the resulting train set will be empty' in str(ve):
            if force_split is False:
                raise ValueError(
                    f'Unable to stratify split on {interactions.negative_sample_type}s - the '
                    f'``interactions`` object contains {interactions.negative_sample_type}s'
                    ' with a single interaction. Either set ``force_split = True`` to put all '
                    f'{interactions.negative_sample_type}s with a single interaction in the training'
                    'set or run ``collie.utils.remove_users_or_items_with_fewer_than_n_interactions``' # TODO name this function
                    ' first.'
                )
            else:
                test_idxs = []

    return test_idxs


def _validate_val_p_and_test_p(val_p: float, test_p: float) -> None:
    validate_and_test_p = val_p + test_p

    if val_p >= 1 or val_p < 0:
        raise ValueError('``val_p`` must be in the range [0, 1).')
    if test_p >= 1 or test_p < 0:
        raise ValueError('``test_p`` must be in the range [0, 1).')
    if validate_and_test_p >= 1 or validate_and_test_p <= 0:
        raise ValueError('The sum of ``val_p`` and ``test_p`` must be in the range (0, 1).')
