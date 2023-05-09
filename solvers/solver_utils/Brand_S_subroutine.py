import torch

def Brand_S_update(U, D, A, r_target, device):
    # M = U D U^T is the current "on the fly matrix"
    # A is the incming A form M + AA^T
    # r_target is where we'll crop the new RSCD representation, if needed - set to None to avoid cropping
    ############ get sizes #############
    r = U.shape[1]
    c = A.shape[1]
    n = A.shape[0]    
    
    ########## force the UDU^T rank to be rtarget. #########################
    ### Note the rank of the matrix we invert will be rtarget + N_bs #######
    if r_target is not None: # if we want to crop down to a specified rank
        #eigenvalues seem to be sorted ascendingly
        if r_target < r:
            D = D[-r_target:]
            U = U[:, -r_target:]
            r = r_target
        else: # if we ask for a bigger rank than what we actually ahve, pass !
            pass
    #########################################################################
    
    ########### compute required linalg quantities #########
    U_T_A = torch.matmul(U.T, A)
    UU_T_A = torch.matmul(U, U_T_A)
    A_minus_UU_T_A = A - UU_T_A
    Q, R = torch.linalg.qr(A_minus_UU_T_A, mode='reduced')
    U_tilde = torch.hstack( (U, Q) )
    
    ######## forming M_s: ##################
    M_11 = torch.matmul(U_T_A, U_T_A.T)
    idx_list_range = list(range(0, r))
    M_11[idx_list_range, idx_list_range] = M_11[idx_list_range, idx_list_range] + D
    M_12 = torch.matmul(U_T_A, R.T)
    M_22 = torch.matmul(R, R.T)
    
    
    #### SPED UP VERSION USING ZEROS REQUIRES CONSTUCTING A NEW TENSOR to "device" 
    # WHICH MAY CAUSE DEADLOCK AT A LATER STAGE IN dist.all_reduce
    """M = torch.vstack((   torch.hstack( (M_11, M_12) ),
                      torch.hstack( (torch.zeros([c,r], device = device), M_22) ) ))"""
    # Try to spped up: (used zeros instead of M_12.T)
    # NOT using M_21 = M_12^T anymore as we are onyl really using the upper-part of the matrix
    M = torch.vstack((   torch.hstack( (M_11, M_12) ),
                      torch.hstack( (M_12.T, M_22) ) ))
    ### USE OLDSCHOOL IMPLEMENTATION TO AVOID the zeros to device problem
    
    # when doing the eigendecompositon. THUS we save the transposition and storage cost!
    S, hat_U = torch.linalg.eigh(M, UPLO='U')
    hat_U = torch.matmul(U_tilde, hat_U)
    
    return S, hat_U

