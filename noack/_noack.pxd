# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()

from .quad_tree cimport QuadTree


ctypedef fused sparse_index_type:
    cnp.int32_t
    cnp.int64_t


cpdef tuple estimate_positive_gradient_nn(
    sparse_index_type[:] indices,
    sparse_index_type[:] indptr,
    double[:] P_data,
    double[:, ::1] embedding,
    double[:, ::1] reference_embedding,
    double[:, ::1] gradient,
    double a=*,
    double dis_eps=*,
    Py_ssize_t num_threads=*,
    bint should_eval_error=*,
)

cpdef double estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double theta=*,
    double r=*,
    double dist_eps=*,
    double elastic_const=*,
    Py_ssize_t num_threads=*,
    bint pairwise_normalization=*,
)
