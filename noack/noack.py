import logging

import numpy as np
import openTSNE
import openTSNE.tsne
from openTSNE import TSNE
from openTSNE.tsne import OptimizationInterrupt

from ._noack import (
    estimate_negative_gradient_bh,
    estimate_positive_gradient_nn,
)
from ._quad_tree import QuadTree

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(openTSNE.tnse.__file__)


def bh_noack(
    embedding,
    P,
    dof,
    bh_params,
    reference_embedding=None,
    attraction=1.0,
    repulsion=-1.0,
    elastic_const=-1,
    dist_eps=0.0,  # a, r, dist_eps are FA2 default
    should_eval_error=False,
    n_jobs=1,
    **_,
):
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself. We've also got to make sure that the points'
    # interactions don't interfere with each other
    pairwise_normalization = reference_embedding is None
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q = estimate_negative_gradient_bh(
        tree,
        embedding,
        gradient,
        **bh_params,
        r=repulsion,
        dist_eps=dist_eps,
        elastic_const=elastic_const,
        num_threads=n_jobs,
        pairwise_normalization=pairwise_normalization,
    )
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = estimate_positive_gradient_nn(
        P.indices,
        P.indptr,
        P.data,
        embedding,
        reference_embedding,
        gradient,
        dof,
        dist_eps=dist_eps,
        a=attraction,
        num_threads=n_jobs,
        should_eval_error=should_eval_error,
    )

    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


class Noack(TSNE):
    def __init__(
        self,
        attraction=1,
        repulsion=-1,
        dist_eps=0.001,
        elastic_const=-1,
        learning_rate=1,
        early_exaggeration_iter=0,
        **kwargs,
    ):
        super().__init__(
            self,
            negative_gradient_method=bh_noack,
            learning_rate=learning_rate,
            early_exaggeration_iter=early_exaggeration_iter,
            **kwargs,
        )
        self.attraction = attraction
        self.repulsion = repulsion
        self.dist_eps = dist_eps
        self.elastic_const = elastic_const

    def fit(self, X=None, affinities=None, initialization=None):
        """Fit a t-SNE embedding for a given data set.

        Runs the standard t-SNE optimization, consisting of the early
        exaggeration phase and a normal optimization phase.

        This function call be called in two ways.

        1.  We can call it in the standard way using a ``np.array``. This will
            compute the affinity matrix and initialization, and run the optimization
            as usual.
        2.  We can also pass in a precomputed ``affinity`` object, which will
            override the affinity-related paramters specified in the constructor.
            This is useful when you wish to use custom affinity objects.

        Please note that some initialization schemes require ``X`` to be specified,
        e.g. PCA. If the initilization is not able to be computed, we default to
        using spectral initilization calculated from the affinity matrix.

        Parameters
        ----------
        X: Optional[np.ndarray]
            The data matrix to be embedded.

        affinities: Optional[openTSNE.affinity.Affinities]
            A precomputed affinity object. If specified, other affinity-related
            parameters are ignored e.g. `perplexity` and anything nearest-neighbor
            search related.

        initialization: Optional[np.ndarray]
            The initial point positions to be used in the embedding space. Can be
            a precomputed numpy array, ``pca``, ``spectral`` or ``random``. Please
            note that when passing in a precomputed positions, it is highly
            recommended that the point positions have small variance
            (std(Y) < 0.0001), otherwise you may get poor embeddings.

        Returns
        -------
        TSNEEmbedding
            A fully optimized t-SNE embedding.

        """
        if self.verbose:
            print("-" * 80, repr(self), "-" * 80, sep="\n")

        embedding = self.prepare_initial(X, affinities, initialization)

        try:
            # Early exaggeration with lower momentum to allow points to find more
            # easily move around and find their neighbors
            embedding.optimize(
                n_iter=self.early_exaggeration_iter,
                exaggeration=self.early_exaggeration,
                momentum=self.initial_momentum,
                inplace=True,
                propagate_exception=True,
                attraction=self.attraction,
                repulsion=self.repulsion,
                dist_eps=self.dist_eps,
                elastic_const=self.elastic_const,
            )

            # Restore actual affinity probabilities and increase momentum to get
            # final, optimized embedding
            embedding.optimize(
                n_iter=self.n_iter,
                exaggeration=self.exaggeration,
                momentum=self.final_momentum,
                inplace=True,
                propagate_exception=True,
                attraction=self.attraction,
                repulsion=self.repulsion,
                dist_eps=self.dist_eps,
                elastic_const=self.elastic_const,
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            embedding = ex.final_embedding

        return embedding
