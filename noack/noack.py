import logging
import multiprocessing
from collections.abc import Iterable
from time import time

import numpy as np
import openTSNE
import openTSNE._tsne
from openTSNE import TSNE
from openTSNE.tsne import OptimizationInterrupt

from ._noack import (
    estimate_negative_gradient_bh,
    estimate_positive_gradient_nn,
)
from .quad_tree import QuadTree

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(openTSNE.tsne.__file__)


def bh_noack(
    embedding,
    P,
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


class gradient_descent(openTSNE.tsne.gradient_descent):
    def __init__(self):
        self.gains = None
        self.update = None

    def copy(self):
        optimizer = self.__class__()
        if self.gains is not None:
            optimizer.gains = np.copy(self.gains)
        if self.update is not None:
            optimizer.update = np.copy(self.update)
        return optimizer

    def __call__(
        self,
        embedding,
        P,
        n_iter,
        objective_function,
        learning_rate="auto",
        momentum=0.8,
        exaggeration=None,
        min_gain=0.01,
        max_grad_norm=None,
        max_step_norm=5,
        theta=0.5,
        attraction=1,
        repulsion=-1,
        dist_eps=0.001,
        elastic_const=-1,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        reference_embedding=None,
        n_jobs=1,
        use_callbacks=False,
        callbacks=None,
        callbacks_every_iters=50,
        verbose=False,
    ):
        """Perform batch gradient descent with momentum and gains.

        Parameters
        ----------
        embedding: np.ndarray
            The embedding :math:`Y`.

        P: array_like
            Joint probability matrix :math:`P`.

        n_iter: int
            The number of iterations to run for.

        objective_function: Callable[..., Tuple[float, np.ndarray]]
            A callable that evaluates the error and gradient for the current
            embedding.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to N / exaggeration as determined in Belkina et al.
            (2019), Nature Communications. Note that this will result in a
            different learning rate during the early exaggeration phase and
            afterwards. This should *not* be used when adding samples into
            existing embeddings, where the learning rate often needs to be much
            lower to obtain convergence.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        min_gain: float
            Minimum individual gain for each parameter.

        max_grad_norm: float
            Maximum gradient norm. If the norm exceeds this value, it will be
            clipped. This is most beneficial when adding points into an existing
            embedding and the new points overlap with the reference points,
            leading to large gradients. This can make points "shoot off" from
            the embedding, causing the interpolation method to compute a very
            large grid, and leads to worse results.

        max_step_norm: float
            Maximum update norm. If the norm exceeds this value, it will be
            clipped. This prevents points from "shooting off" from
            the embedding.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        ints_in_interval: float
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. Indicates how large a grid cell should be e.g. a value of 3
            indicates a grid side length of 3. Lower values provide more
            accurate gradient estimations.

        reference_embedding: np.ndarray
            If we are adding points to an existing embedding, we have to compute
            the gradients and errors w.r.t. the existing embedding.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        use_callbacks: bool

        callbacks: Callable[[int, float, np.ndarray] -> bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        float
            The KL divergence of the optimized embedding.
        np.ndarray
            The optimized embedding Y.

        Raises
        ------
        OptimizationInterrupt
            If the provided callback interrupts the optimization, this is raised.

        """
        assert isinstance(embedding, np.ndarray), (
            "`embedding` must be an instance of `np.ndarray`. Got `%s` instead"
            % type(embedding)
        )

        if reference_embedding is not None:
            assert isinstance(reference_embedding, np.ndarray), (
                "`reference_embedding` must be an instance of `np.ndarray`. Got "
                "`%s` instead" % type(reference_embedding)
            )

        # If we're running transform and using the interpolation scheme, then we
        # should limit the range where new points can go to
        should_limit_range = False
        if reference_embedding is not None:
            if reference_embedding.box_x_lower_bounds is not None:
                should_limit_range = True
                lower_limit = reference_embedding.box_x_lower_bounds[0]
                upper_limit = reference_embedding.box_x_lower_bounds[-1]

        if self.update is None:
            self.update = np.zeros_like(embedding).view(np.ndarray)
        if self.gains is None:
            self.gains = np.ones_like(embedding).view(np.ndarray)

        bh_params = {"theta": theta}
        fft_params = {
            "n_interpolation_points": n_interpolation_points,
            "min_num_intervals": min_num_intervals,
            "ints_in_interval": ints_in_interval,
        }

        # Lie about the P values for bigger attraction forces
        if exaggeration is None:
            exaggeration = 1

        if exaggeration != 1:
            P *= exaggeration

        # Notify the callbacks that the optimization is about to start
        if isinstance(callbacks, Iterable):
            for callback in callbacks:
                # Only call function if present on object
                getattr(callback, "optimization_about_to_start", lambda: ...)()

        timer = openTSNE.utils.Timer(
            "Running optimization with exaggeration=%.2f, lr=%.2f for %d iterations..."
            % (exaggeration, learning_rate, n_iter),
            verbose=verbose,
        )
        timer.__enter__()

        if verbose:
            start_time = time()

        for iteration in range(n_iter):
            should_call_callback = (
                use_callbacks and (iteration + 1) % callbacks_every_iters == 0
            )
            # Evaluate error on 50 iterations for logging, or when callbacks
            should_eval_error = should_call_callback or (
                verbose and (iteration + 1) % 50 == 0
            )

            error, gradient = objective_function(
                embedding,
                P,
                bh_params=bh_params,
                attraction=attraction,
                repulsion=repulsion,
                dist_eps=dist_eps,
                elastic_const=elastic_const,
                fft_params=fft_params,
                reference_embedding=reference_embedding,
                n_jobs=n_jobs,
                should_eval_error=should_eval_error,
            )

            # Clip gradients to avoid points shooting off. This can be an issue
            # when applying transform and points are initialized so that the new
            # points overlap with the reference points, leading to large
            # gradients
            if max_grad_norm is not None:
                norm = np.linalg.norm(gradient, axis=1)
                coeff = max_grad_norm / (norm + 1e-6)
                mask = coeff < 1
                gradient[mask] *= coeff[mask, None]

            # Correct the KL divergence w.r.t. the exaggeration if needed
            if should_eval_error and exaggeration != 1:
                error = error / exaggeration - np.log(exaggeration)

            if should_call_callback:
                # Continue only if all the callbacks say so
                should_stop = any(
                    (
                        bool(c(iteration + 1, error, embedding))
                        for c in callbacks
                    )
                )
                if should_stop:
                    # Make sure to un-exaggerate P so it's not corrupted in future runs
                    if exaggeration != 1:
                        P /= exaggeration
                    raise OptimizationInterrupt(
                        error=error, final_embedding=embedding
                    )

            # Update the embedding using the gradient
            grad_direction_flipped = np.sign(self.update) != np.sign(gradient)
            grad_direction_same = np.invert(grad_direction_flipped)
            self.gains[grad_direction_flipped] += 0.2
            self.gains[grad_direction_same] = (
                self.gains[grad_direction_same] * 0.8 + min_gain
            )
            gradient = gradient.view(np.ndarray)
            self.update = (
                momentum * self.update - learning_rate * self.gains * gradient
            )

            # Clip the update sizes
            if max_step_norm is not None:
                update_norms = np.linalg.norm(
                    self.update, axis=1, keepdims=True
                )
                mask = update_norms.squeeze() > max_step_norm
                self.update[mask] /= update_norms[mask]
                self.update[mask] *= max_step_norm

            embedding += self.update

            # Zero-mean the embedding only if we're not adding new data points,
            # otherwise this will reset point positions
            if reference_embedding is None:
                embedding -= np.mean(embedding, axis=0)

            # Limit any new points within the circle defined by the interpolation grid
            if should_limit_range:
                if embedding.shape[1] == 1:
                    mask = (embedding < lower_limit) | (
                        embedding > upper_limit
                    )
                    np.clip(embedding, lower_limit, upper_limit, out=embedding)
                elif embedding.shape[1] == 2:
                    r_limit = max(abs(lower_limit), abs(upper_limit))
                    embedding, mask = openTSNE.utils.clip_point_to_disc(
                        embedding, r_limit, inplace=True
                    )

                # Zero out the momentum terms for the points that hit the boundary
                self.gains[~mask] = 0

            if verbose and (iteration + 1) % 50 == 0:
                stop_time = time()
                print(
                    "Iteration %4d, KL divergence %6.4f, 50 iterations in %.4f sec"
                    % (iteration + 1, error, stop_time - start_time)
                )
                start_time = time()

        timer.__exit__()

        # Make sure to un-exaggerate P so it's not corrupted in future runs
        if exaggeration != 1:
            P /= exaggeration

        # The error from the loop is the one for the previous, non-updated
        # embedding. We need to return the error for the actual final embedding, so
        # compute that at the end before returning
        error, _ = objective_function(
            embedding,
            P,
            bh_params=bh_params,
            fft_params=fft_params,
            attraction=attraction,
            repulsion=repulsion,
            dist_eps=dist_eps,
            elastic_const=elastic_const,
            reference_embedding=reference_embedding,
            n_jobs=n_jobs,
            should_eval_error=True,
        )

        return error, embedding


def _handle_nice_params(embedding: np.ndarray, optim_params: dict) -> None:
    """Convert the user friendly params into something the optimizer can
    understand."""
    # Handle callbacks
    optim_params["callbacks"] = openTSNE.tsne._check_callbacks(
        optim_params.get("callbacks")
    )
    optim_params["use_callbacks"] = optim_params["callbacks"] is not None

    # Handle negative gradient method
    negative_gradient_method = optim_params.pop("negative_gradient_method")
    if callable(negative_gradient_method):
        negative_gradient_method = negative_gradient_method
    # elif negative_gradient_method in {"umap", "UMAP", "bhumap"}:
    #     negative_gradient_method = bh_umap
    # elif negative_gradient_method in {"elastic"}:
    #     negative_gradient_method = bh_elastic
    elif negative_gradient_method in {"noack"}:
        negative_gradient_method = bh_noack
    elif negative_gradient_method in {"bh", "BH", "barnes-hut"}:
        negative_gradient_method = openTSNE.kl_divergence_bh
    elif negative_gradient_method in {"fft", "FFT", "interpolation"}:
        negative_gradient_method = openTSNE.kl_divergence_fft
    else:
        raise ValueError(
            "Unrecognized gradient method. Please choose one of "
            "the supported methods or provide a valid callback."
        )
    # `gradient_descent` uses the more informative name `objective_function`
    optim_params["objective_function"] = negative_gradient_method

    # Handle number of jobs
    n_jobs = optim_params.get("n_jobs", 1)
    if n_jobs < 0:
        n_cores = multiprocessing.cpu_count()
        # Add negative number of n_jobs to the number of cores, but increment by
        # one because -1 indicates using all cores, -2 all except one, and so on
        n_jobs = n_cores + n_jobs + 1

    # If the number of jobs, after this correction is still <= 0, then the user
    # probably thought they had more cores, so we'll default to 1
    if n_jobs <= 0:
        log.warning(
            "`n_jobs` receieved value %d but only %d cores are available. "
            "Defaulting to single job." % (optim_params["n_jobs"], n_cores)
        )
        n_jobs = 1

    optim_params["n_jobs"] = n_jobs

    # Determine learning rate if requested
    if optim_params.get("learning_rate", "auto") == "auto":
        optim_params["learning_rate"] = max(200, embedding.shape[0] / 12)


class NoackEmbedding(openTSNE.TSNEEmbedding):
    def __new__(
        cls,
        embedding,
        affinities,
        reference_embedding=None,
        negative_gradient_method="noack",
        random_state=None,
        optimizer=None,
        **gradient_descent_params,
    ):
        # init_checks.num_samples(embedding.shape[0], affinities.P.shape[0])

        obj = np.asarray(embedding, dtype=np.float64, order="C").view(
            NoackEmbedding
        )

        obj.reference_embedding = reference_embedding
        obj.P = affinities.P
        obj.gradient_descent_params = gradient_descent_params
        obj.affinities = affinities  # type: Affinities
        obj.gradient_descent_params = gradient_descent_params  # type: dict
        obj.gradient_descent_params.update(
            {"negative_gradient_method": negative_gradient_method}
        )
        obj.random_state = random_state

        if optimizer is None:
            optimizer = gradient_descent()
        elif not isinstance(optimizer, openTSNE.tsne.gradient_descent):
            raise TypeError(
                "`optimizer` must be an instance of `%s`, but got `%s`."
                % (
                    openTSNE.tsne.gradient_descent.__class__.__name__,
                    type(optimizer),
                )
            )
        obj.optimizer = optimizer

        obj.kl_divergence = None

        # Interpolation grid variables
        obj.interp_coeffs = None
        obj.box_x_lower_bounds = None
        obj.box_y_lower_bounds = None

        return obj

    def optimize(
        self,
        n_iter,
        inplace=False,
        propagate_exception=False,
        **gradient_descent_params,
    ):
        """Run optmization on the embedding for a given number of steps.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to max(200, N / 12), as determined in Belkina et al.
            "Automated optimized parameters for t-distributed stochastic
            neighbor embedding improve visualization and analysis of large
            datasets", 2019. Note that this should *not* be used when adding
            samples into existing embeddings, where the learning rate often
            needs to be much lower to obtain convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        negative_gradient_method: str
            Specifies the negative gradient approximation method to use. For
            smaller data sets, the Barnes-Hut approximation is appropriate and
            can be set using one of the following aliases: ``bh``, ``BH`` or
            ``barnes-hut``. For larger data sets, the FFT accelerated
            interpolation method is more appropriate and can be set using one of
            the following aliases: ``fft``, ``FFT`` or ``ìnterpolation``.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.

        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.

        random_state: Union[int, RandomState]
            The random state parameter follows the convention used in
            scikit-learn. If the value is an int, random_state is the seed used
            by the random number generator. If the value is a RandomState
            instance, then it will be used as the random number generator. If
            the value is None, the random number generator is the RandomState
            instance used by `np.random`.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        callbacks: Callable[[int, float, np.ndarray] -> bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        PartialTSNEEmbedding
            An optimized partial t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = NoackEmbedding(
                np.copy(self),
                self.reference_embedding,
                self.P,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        optim_params["n_iter"] = n_iter
        # this calls the function I patched in this file, not the one
        # from openTSNE
        _handle_nice_params(embedding, optim_params)

        try:
            # Run gradient descent with the embedding optimizer so gains are
            # properly updated and kept
            error, embedding = embedding.optimizer(
                embedding=embedding,
                reference_embedding=self.reference_embedding,
                P=self.P,
                **optim_params,
            )

        except openTSNE.OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding


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

    def prepare_initial(self, X=None, affinities=None, initialization=None):
        """Prepare the initial embedding which can be optimized as needed.

        This function call be called in two ways.

        1.  We can call it in the standard way using a ``np.array``. This will
            compute the affinity matrix and initialization as usual.
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
            An unoptimized :class:`TSNEEmbedding` object, prepared for
            optimization.

        """

        # Either `X` or `affinities` must be specified
        if X is None and affinities is None and initialization is None:
            raise ValueError(
                "At least one of the parameters `X` or `affinities` must be specified!"
            )

        # If precomputed affinites are given, use those, otherwise proceed with
        # standard perpelxity-based affinites
        if affinities is None:
            affinities = openTSNE.affinity.MultiscaleMixture(
                X,
                self.perplexity,
                method=self.neighbors,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            if not isinstance(affinities, openTSNE.affinity.Affinities):
                raise ValueError(
                    "`affinities` must be an instance of `openTSNE.affinity.Affinities`"
                )
            log.info(
                "Precomputed affinities provided. Ignoring perplexity-related parameters."
            )

        # If a precomputed initialization was specified, use that, otherwise
        # use the parameters specified in the constructor
        if initialization is None:
            initialization = self.initialization
            log.info(
                "Precomputed initialization provided. Ignoring initalization-related "
                "parameters."
            )

        # If only the affinites have been specified, and the initialization depends
        # on `X`, switch to spectral initalization
        if (
            X is None
            and isinstance(initialization, str)
            and initialization == "pca"
        ):
            log.warning(
                "Attempting to use `pca` initalization, but no `X` matrix specified! "
                "Using `spectral` initilization instead, which doesn't need access "
                "to the data matrix"
            )
            initialization = "spectral"

        # Same spiel for precomputed distance matrices
        if (
            self.metric == "precomputed"
            and isinstance(initialization, str)
            and initialization == "pca"
        ):
            log.warning(
                "Attempting to use `pca` initalization, but using precomputed "
                "distance matrix! Using `spectral` initilization instead, which "
                "doesn't need access to the data matrix."
            )
            initialization = "spectral"

        # Determine the number of samples in the input data set
        if X is not None:
            n_samples = X.shape[0]
        else:
            n_samples = affinities.P.shape[0]

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            openTSNE.tsne.init_checks.num_samples(
                initialization.shape[0], n_samples
            )
            openTSNE.tsne.init_checks.num_dimensions(
                initialization.shape[1], self.n_components
            )

            embedding = np.array(initialization)

        elif initialization == "pca":
            embedding = openTSNE.initialization.pca(
                X,
                self.n_components,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        elif initialization == "random":
            embedding = openTSNE.initialization.random(
                n_samples,
                self.n_components,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        elif initialization == "spectral":
            embedding = openTSNE.initialization.spectral(
                affinities.P,
                self.n_components,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            raise ValueError(
                f"Unrecognized initialization scheme `{initialization}`."
            )

        gradient_descent_params = {
            "negative_gradient_method": self.negative_gradient_method,
            "learning_rate": self.learning_rate,
            # By default, use the momentum used in unexaggerated phase
            "momentum": self.final_momentum,
            # Barnes-Hut params
            "theta": self.theta,
            "max_grad_norm": self.max_grad_norm,
            "max_step_norm": self.max_step_norm,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            # Callback params
            "callbacks": self.callbacks,
            "callbacks_every_iters": self.callbacks_every_iters,
        }

        return NoackEmbedding(
            embedding,
            affinities=affinities,
            random_state=self.random_state,
            **gradient_descent_params,
        )
