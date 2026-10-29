"""Declarations of the C interface of Ipopt."""


cdef extern from "coin-or/IpStdCInterface.h" nogil:
    # The header names these types `bool`, `ipindex` and `ipnumber`. Declaring `bool` as an
    # integer type keeps Cython from treating it as the Python truth value, while the
    # generated C still uses the type of the header, so callbacks have the right signature.
    ctypedef unsigned char ipbool "bool"
    ctypedef int ipindex
    ctypedef double ipnumber
    ctypedef void* UserDataPtr

    ctypedef struct IpoptProblemInfo:
        pass

    ctypedef IpoptProblemInfo* IpoptProblem

    ctypedef ipbool (*Eval_F_CB)(
        ipindex n, ipnumber* x, ipbool new_x, ipnumber* obj_value, UserDataPtr user_data
    ) noexcept nogil

    ctypedef ipbool (*Eval_Grad_F_CB)(
        ipindex n, ipnumber* x, ipbool new_x, ipnumber* grad_f, UserDataPtr user_data
    ) noexcept nogil

    ctypedef ipbool (*Eval_G_CB)(
        ipindex n, ipnumber* x, ipbool new_x, ipindex m, ipnumber* g, UserDataPtr user_data
    ) noexcept nogil

    ctypedef ipbool (*Eval_Jac_G_CB)(
        ipindex n,
        ipnumber* x,
        ipbool new_x,
        ipindex m,
        ipindex nele_jac,
        ipindex* iRow,
        ipindex* jCol,
        ipnumber* values,
        UserDataPtr user_data,
    ) noexcept nogil

    ctypedef ipbool (*Eval_H_CB)(
        ipindex n,
        ipnumber* x,
        ipbool new_x,
        ipnumber obj_factor,
        ipindex m,
        ipnumber* lambda_,
        ipbool new_lambda,
        ipindex nele_hess,
        ipindex* iRow,
        ipindex* jCol,
        ipnumber* values,
        UserDataPtr user_data,
    ) noexcept nogil

    ctypedef ipbool (*Intermediate_CB)(
        ipindex alg_mod,
        ipindex iter_count,
        ipnumber obj_value,
        ipnumber inf_pr,
        ipnumber inf_du,
        ipnumber mu,
        ipnumber d_norm,
        ipnumber regularization_size,
        ipnumber alpha_du,
        ipnumber alpha_pr,
        ipindex ls_trials,
        UserDataPtr user_data,
    ) noexcept nogil

    IpoptProblem CreateIpoptProblem(
        ipindex n,
        ipnumber* x_L,
        ipnumber* x_U,
        ipindex m,
        ipnumber* g_L,
        ipnumber* g_U,
        ipindex nele_jac,
        ipindex nele_hess,
        ipindex index_style,
        Eval_F_CB eval_f,
        Eval_G_CB eval_g,
        Eval_Grad_F_CB eval_grad_f,
        Eval_Jac_G_CB eval_jac_g,
        Eval_H_CB eval_h,
    )
    void FreeIpoptProblem(IpoptProblem ipopt_problem)
    ipbool AddIpoptStrOption(IpoptProblem ipopt_problem, char* keyword, char* val)
    ipbool AddIpoptNumOption(IpoptProblem ipopt_problem, char* keyword, ipnumber val)
    ipbool AddIpoptIntOption(IpoptProblem ipopt_problem, char* keyword, int val)
    ipbool SetIntermediateCallback(IpoptProblem ipopt_problem, Intermediate_CB intermediate_cb)
    int IpoptSolve(
        IpoptProblem ipopt_problem,
        ipnumber* x,
        ipnumber* g,
        ipnumber* obj_val,
        ipnumber* mult_g,
        ipnumber* mult_x_L,
        ipnumber* mult_x_U,
        UserDataPtr user_data,
    )
    ipbool GetIpoptCurrentIterate(
        IpoptProblem ipopt_problem,
        ipbool scaled,
        ipindex n,
        ipnumber* x,
        ipnumber* z_L,
        ipnumber* z_U,
        ipindex m,
        ipnumber* g,
        ipnumber* lambda_,
    )
    void GetIpoptVersion(int* major, int* minor, int* release)
