import torch
def X_reg_inverse_M_adaptive_damping(U,D,M,lambdda, n_kfactor_update, rho, damping_type): # damping_type is just an artefact now
    # X = UDU^T; want to compute (X + lambda I)^{-1}M
    # X is low rank! X is square: X is either AA^T or GG^T
    # This is actually for G as G sits before
    # the damping here is adaptive! - it adjusts based on the amxium eigenvalue !
    lbd_continue = torch.min(D) # torch.min(D) # 0 #<---possible choices
    #if damping_type == 'adaptive':
    lambdda = lambdda * torch.max(D) + rho**n_kfactor_update #+ rho**n_kfactor_update is the identity initialization of kfactors moved to reg
    lambdda = lambdda + lbd_continue
    #### effective computations :
    U_T_M = torch.matmul(U.T, M)
    U_times_reg_D_times_U_T_M = torch.matmul( U * ( 1/(D + lambdda - lbd_continue) - 1/lambdda), U_T_M)
    return U_times_reg_D_times_U_T_M + (1/lambdda) * M
    
def M_X_reg_inverse_adaptive_damping(U,D,M,lambdda, n_kfactor_update, rho, damping_type): # damping_type is just an artefact now
    # X = UDU^T; want to compute (X + lambda I)^{-1}M
    # X is low rank! X is square: X is either AA^T or GG^T
    # This is actually for A as A sits after M
    # the damping here is adaptive! - it adjusts based on the amxium eigenvalue !
    lbd_continue = torch.min(D) # torch.min(D) # 0 #<---possible choices
    #if damping_type == 'adaptive':
    lambdda = lambdda * torch.max(D) + rho**n_kfactor_update #+ rho**n_kfactor_update is the identity initialization of kfactors moved to reg
    lambdda = lambdda + lbd_continue
    #### effective computations :
    M_times_U_times_reg_D_times_U_T = M @ ( U * ( 1/(D + lambdda - lbd_continue) - 1/lambdda) ) @ U.T
    return M_times_U_times_reg_D_times_U_T + (1/lambdda) * M

def RSVD_lowrank(M, oversampled_rank, target_rank, niter, start_matrix = None):
    U, D, V = torch.svd_lowrank(M, q = oversampled_rank, niter = niter, M = None) # RSVD returns SVs in descending order !
    # we're flipping because we want the eigenvalues in ASCENDING order ! s.t. we can work with the brand subroutine which uses eigh with ascending order evals
    return torch.flip(D[:target_rank] + 0.0, dims=(0,)), torch.flip(V[:, :target_rank] + 0.0, dims=(1,)).contiguous()  # OMEGA IS u - overwritten for efficiency
