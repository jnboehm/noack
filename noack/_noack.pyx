# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()
import numpy as np
from cython.parallel import prange, parallel
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free

from .quad_tree cimport QuadTree, Node, is_close


cdef double EPSILON = np.finfo(np.float64).eps


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil
    double exp(double x) nogil
    double fabs(double x) nogil
    double pow(double x, double y) nogil
    double powf(float x, float y) nogil
    double fmax(double x, double y) nogil
    double isinf(long double) nogil
    double INFINITY


cpdef tuple estimate_positive_gradient_nn(
    sparse_index_type[:] indices,
    sparse_index_type[:] indptr,
    double[:] P_data,
    double[:, ::1] embedding,
    double[:, ::1] reference_embedding,
    double[:, ::1] gradient,
    double a=1,
    double dist_eps=0.001,
    Py_ssize_t num_threads=1,
    bint should_eval_error=False,
):
    cdef:
        Py_ssize_t n_samples = gradient.shape[0]
        Py_ssize_t n_dims = gradient.shape[1]
        double * diff
        double w_ij, d_ij, p_ij, q_ij, kl_divergence = 0, sum_P = 0
        double power, tmp

        Py_ssize_t i, j, k, d

    if num_threads < 1:
        num_threads = 1

    with nogil, parallel(num_threads=num_threads):
        # Use `malloc` here instead of `PyMem_Malloc` because we're in a
        # `nogil` clause and we won't be allocating much memory
        diff = <double *>malloc(n_dims * sizeof(double))
        if not diff:
            with gil:
                raise MemoryError()

        power = fabs(a - 1)
        for i in prange(n_samples, schedule="guided"):
            # Iterate over all the neighbors `j` and sum up their contribution
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                p_ij = P_data[k]
                # Compute the direction of the points attraction and the
                # squared euclidean distance between the points
                for d in range(n_dims):
                    diff[d] = embedding[i, d] - reference_embedding[j, d]
                    # w_ij = w_ij + powf(fabs(diff[d]), a - 1)
                    tmp = fabs(diff[d])
                    if power == 0:
                        d_ij = d_ij + 1
                    elif power == 1:
                        d_ij = d_ij + tmp
                    elif power == 2:
                        d_ij = d_ij + tmp * tmp
                    else:
                        d_ij = d_ij + pow(tmp, power)

                if a - 1 < 0:
                    w_ij = 1 / (d_ij + dist_eps)
                else:
                    w_ij = d_ij + dist_eps

                # Compute F_{attr} of point `j` on point `i`
                for d in range(n_dims):
                    gradient[i, d] = gradient[i, d] + w_ij * p_ij * diff[d]

                # Evaluating the following expressions can slow things down
                # considerably if evaluated every iteration. Note that the q_ij
                # is unnormalized, so we need to normalize once the sum of q_ij
                # is known
                if should_eval_error:
                    sum_P += p_ij
                    # Now we need to do the exp by dof
                    kl_divergence += p_ij * log((p_ij / (w_ij + EPSILON)) + EPSILON)

        free(diff)

    return sum_P, kl_divergence


cpdef double estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double theta=0.5,
    double r=-1.0,
    double dist_eps=0.001,
    double elastic_const=-1.0,

    Py_ssize_t num_threads=1,
    bint pairwise_normalization=True,
):
    """Estimate the negative force gradient using the Barnes Hut approximation.
    The parameter `r` corresponds to the parameter in the attraction-repulsion
    spectrum by Noack (2009), i. e. the repulsive exponent of the energy model,
    of which we calculate the gradient here.

    Notes
    -----
    Changes the gradient inplace to avoid needless memory allocation. As
    such, this must be run before estimating the positive gradients, since
    the negative gradient must be normalized at the end with the sum of
    q_{ij}s.

    """
    cdef:
        Py_ssize_t i, j, num_points = embedding.shape[0]
        double sum_Q = 0
        double[::1] sum_Qi = np.zeros(num_points, dtype=float)
        double Z

    if num_threads < 1:
        num_threads = 1

    # In order to run gradient estimation in parallel, we need to pass each
    # worker its own memory slot to write sum_Qs
    for i in prange(num_points, nogil=True, num_threads=num_threads, schedule="guided"):
        _estimate_negative_gradient_single(
            &tree.root, &embedding[i, 0], &gradient[i, 0], &sum_Qi[i], theta, r, dist_eps)

    for i in range(num_points):
        sum_Q += sum_Qi[i]

    # Normalize q_{ij}s
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            if elastic_const > 0:
                Z = elastic_const
            elif pairwise_normalization:
                Z = sum_Q + EPSILON
            else:
                Z = sum_Qi[i] + EPSILON

            gradient[i, j] /= Z

    return sum_Q


cdef void _estimate_negative_gradient_single(
    Node * node,
    double * point,
    double * gradient,
    double * sum_Q,
    double theta,
    double r,
    double dist_eps,
) nogil:
    # Make sure that we spend no time on empty nodes or simple self-interactions
    if node.num_points == 0 or node.is_leaf and is_close(node, point, EPSILON):
        return

    cdef:
        double sqdistance = EPSILON
        double diff, power, w_ij = dist_eps
        Py_ssize_t d

    power = fabs(r - 1)
    # Compute the squared euclidean distance in the embedding space from the
    # new point to the center of mass
    for d in range(node.n_dims):
        tmp = fabs(node.center_of_mass[d] - point[d])
        sqdistance += tmp * tmp
        if power == 0:
            w_ij += 1.0
        if power == 1:
            w_ij += tmp
        elif power == 2:
            # copy sqdistance
            w_ij = sqdistance
        elif power == 3:
            w_ij += tmp * tmp * tmp
        else:
            w_ij += powf(tmp, power)

    if r - 1 < 0:
        w_ij = 1 / w_ij

    # Check whether we can use this node as a summary
    if node.is_leaf or node.length / sqrt(sqdistance) < theta:
        sum_Q[0] += node.num_points * w_ij

        for d in range(node.n_dims):
            gradient[d] -= node.num_points * w_ij * (point[d] - node.center_of_mass[d])

        return

    # Otherwise we have to look for summaries in the children
    for d in range(1 << node.n_dims):
        _estimate_negative_gradient_single(&node.children[d], point, gradient, sum_Q, theta, r, dist_eps)
