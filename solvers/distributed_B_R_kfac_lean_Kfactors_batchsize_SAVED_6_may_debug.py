import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from kfac_utils_for_vgg16_bn import (ComputeCovA, ComputeCovG)
from kfac_utils_for_vgg16_bn import update_running_stat
from kfac_utils_for_vgg16_bn import fct_split_list_of_modules

from Brand_S_subroutine import Brand_S_update

def X_reg_inverse_M_adaptive_damping(U,D,M,lambdda, damping_type): # damping_type is just an artefact now
    # X = UDU^T; want to compute (X + lambda I)^{-1}M
    # X is low rank! X is square: X is either AA^T or GG^T
    # This is actually for G as G sits before
    # the damping here is adaptive! - it adjusts based on the amxium eigenvalue !
    lbd_continue = torch.min(D) # torch.min(D) # 0 #<---possible choices
    #if damping_type == 'adaptive':
    lambdda = lambdda * torch.max(D)
    lambdda = lambdda + lbd_continue
    #### effective computations :
    U_T_M = torch.matmul(U.T, M)
    U_times_reg_D_times_U_T_M = torch.matmul( U * ( 1/(D + lambdda - lbd_continue) - 1/lambdda), U_T_M)
    return U_times_reg_D_times_U_T_M + (1/lambdda) * M
    
def M_X_reg_inverse_adaptive_damping(U,D,M,lambdda, damping_type): # damping_type is just an artefact now
    # X = UDU^T; want to compute (X + lambda I)^{-1}M
    # X is low rank! X is square: X is either AA^T or GG^T
    # This is actually for A as A sits after M
    # the damping here is adaptive! - it adjusts based on the amxium eigenvalue !
    lbd_continue = torch.min(D) # torch.min(D) # 0 #<---possible choices
    #if damping_type == 'adaptive':
    lambdda = lambdda * torch.max(D)
    lambdda = lambdda + lbd_continue
    #### effective computations :
    M_times_U_times_reg_D_times_U_T = M @ ( U * ( 1/(D + lambdda - lbd_continue) - 1/lambdda) ) @ U.T
    return M_times_U_times_reg_D_times_U_T + (1/lambdda) * M

def RSVD_lowrank(M, oversampled_rank, target_rank, niter, start_matrix = None):
    U, D, V = torch.svd_lowrank(M, q = oversampled_rank, niter = niter, M = None) # RSVD returns SVs in descending order !
    # we're flipping because we want the eigenvalues in ASCENDING order ! s.t. we can work with the brand subroutine which uses eigh with ascending order evals
    return torch.flip(D[:target_rank] + 0.0, dims=(0,)), torch.flip(V[:, :target_rank] + 0.0, dims=(1,))  # OMEGA IS u - overwritten for efficiency

class B_R_KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 rank, world_size,
                 lr_function = lambda epoch_n, iteration_n: 0.1,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 rsvd_rank = 220,
                 oversampling_parameter = 10,
                 rsvd_niter = 3,
                 damping_type = 'adaptive',
                 clip_type = 'non_standard',
                 brand_period = 10, 
                 brand_r_target_excess = 0,
                 brand_update_multiplier_to_TCov = 1):
        
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.epoch_number = 1
        defaults = dict(lr = lr_function(self.epoch_number, 0), 
                        momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(B_R_KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model

        self.lr_function = lr_function
        self.lr = self.lr_function(self.epoch_number, 0)
        
        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {} 
        self.d_a, self.d_g = {}, {}
        self.size_of_missing_m_aa = {}
        self.size_of_missing_m_gg = {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        
        # parallelism flags
        self.data_parallel = False
        self.layewise_parallel = True
        
        # distributed stuff
        # self.modules_for_this_rank is initialized here!
        self.rank = rank
        self.world_size = world_size
        self._prepare_model()
        #for it_rank in range(0, world_size):
        #    self.modules_for_this_rank[it_rank] = [] # a dictionary of lists 
        # modules will get appended to list accordingly, later at preparing model
        
        # debug flags
        self.K_fac_incoming_info_debugger_mode = False
        self.Dist_communication_debugger = False
        self.dist_communication_2nd_version_debugger = False
        self.dist_comm_for_layers_debugger = True
        self.dist_debugger_testing_leanness_thing = False
        
        ### R-KFAC specific or introduced with RKFAC for te 1st time
        #rsvd_params
        self.rsvd_rank = rsvd_rank
        self.total_rsvd_rank = oversampling_parameter + rsvd_rank
        self.rsvd_niter = rsvd_niter
        # introduced with RKFAC for te 1st time but also relevant to simple KFAC
        self.damping_type = damping_type
        self.clip_type = clip_type
        ## RKFAC specific init
        self.U_aa, self.D_aa, self.V_aa = {}, {}, {}
        self.U_gg, self.D_gg, self.V_gg = {}, {}, {}
        
        ####### Brand-specific hyperparams ####################################
        self.brand_period = brand_period # this is in numebr of "updating stages" multiply by TCov to get in number fo steps!
        #print('type(brand_r_target_excess) = {}, brand_r_target_excess = {}'.format(type(brand_r_target_excess),brand_r_target_excess))
        #print('type of rsvd_rank is {}'.format(type(rsvd_rank)))
        self.brand_r_target = rsvd_rank + brand_r_target_excess
        self.brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov
        self.sqr_1_minus_stat_decay = (1 - stat_decay)**(0.5) # to avoid recomputations
        #######################################################################
        
    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            if module in self.modules_for_this_rank[self.rank]: # ONLY compute the Kfactor and update it if the GPU parsing this
                #is responsible for this aprticular module
                # try concatenate reduction
                aa = self.CovAHandler(input[0].data, module)
                
                ############ DEBUG ONLY #########
                if self.Dist_communication_debugger:
                    print('RANK {} WORLDSIZE {}. At module {}. AA^T Value BEFORE reducing is = {}\n'.format(
                        self.rank, self.world_size, module, aa))
                
                if self.K_fac_incoming_info_debugger_mode or self.dist_debugger_testing_leanness_thing:
                    print('RANK {} WORLDSIZE {}. At module {} \n ... the A size is {}\n'.format(self.rank, self.world_size, module, input[0].data.shape))
                ############ END DEBUG ONLY #########
                
                # Initialize buffers
                if self.steps == 0:
                    self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
                    # here we initialize with identity and we'll move this to the reg term for R-KFAC and B-KFAC
                
                ############ DEBUG ONLY #########
                if self.Dist_communication_debugger:
                    aa_local = aa + 0
                    aa = aa/self.world_size # divide by worldsize because we allreduce with sum not average!
                    dist.all_reduce(aa, dist.ReduceOp.SUM, async_op = False)
                    print('RANK {} WORLDSIZE {}. At module {}. ||aa - aa.all_reduced_avg||_2/||(1/2)aa||_2 = {} , ||(1/2)aa - aa.all_reduced_avg||_2/||(1/2)aa||_2 = {}\n'.format(
                        self.rank, self.world_size, module, torch.norm(aa_local - aa)/torch.norm(aa_local), torch.norm(aa_local/2 - aa)/torch.norm(aa_local/2) ) )
                    print('RANK {} WORLDSIZE {}. At module {}. AA^T Value after reducing is = {}\n'.format(
                        self.rank, self.world_size, module,aa))
                ############ END DEBUG ONLY #########
                
                ############ also update the real AA for alter correction ########
                ## TODO:make this NOT happen when we're doing PURE B-KFAC
                update_running_stat(aa, self.m_aa[module], self.stat_decay)
                
                ''' TO DO: make it possible to accumulate in incoming A (from AA^T)
                and only invert when the time comes : i.e. make it work for for TCov < TInv!'''
                
                ############ Brand - Update the RSVD of \bar{AA} ##############
                # if layer is linear and it's not time to do RSVD and it's time to do brand update
                if isinstance(module, nn.Linear) and self.steps % (self.TInv * self.brand_period) != 0 and self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0:
                    n_dim = input[0].data.shape[1]; c_dim = input[0].data.shape[0]; r_dim = self.Q_a[module].shape[1]
                    if r_dim + c_dim < n_dim: # only if the new B_updated AA^T matrix is lowrank
                        # If the layer is linear! and it's time to do the update
                        # g: batch_size * out_dim
                        batch_size = c_dim
                        
                        A = input[0].data / (batch_size + 0)**(0.5)
                        if module.bias is not None:
                            #print('\n Updating AA^T: we have bias in linear layer!')
                            A = torch.cat([A, A.new(A.size(0), 1).fill_(1)], 1).T
        
                        self.d_a[module], self.Q_a[module] = Brand_S_update(self.Q_a[module], self.stat_decay * self.d_a[module],
                                                                            A = self.sqr_1_minus_stat_decay * A, r_target = self.brand_r_target, 
                                                                            device = torch.device('cuda:{}'.format(self.rank)) )
                    else: # if doing the brand update would give a Raw representation of higher rank than Max possible one, just do the rsvd
                        oversampled_rank = min(self.m_aa[module].shape[0], self.total_rsvd_rank)
                        actual_rank = min(self.m_aa[module].shape[0], self.rsvd_rank)
                        self.d_a[module], self.Q_a[module] = RSVD_lowrank(M = self.m_aa[module], oversampled_rank = oversampled_rank, target_rank = actual_rank, niter = self.rsvd_niter, start_matrix = None)
                    # Ensure tensor Q_a is contiguous s.t. the allreduce can work! the tensor becoming noncontiguous can be due to transposition and it occurs in practice
                    self.Q_a[module] = self.Q_a[module].contiguous()
                ####### END : Brand - Update the RSVD of \bar{AA} ############## 
                
                ############ DEBUG ONLY #############
                if self.dist_communication_2nd_version_debugger and (self.steps % self.TCov == 0):
                    print('RANK {} WORLDSIZE {}. At module {}. FINISHED doing updatestats\n at step {}\n'.format(self.rank, self.world_size, module, self.steps))
                ############ END DEBUG ONLY #########
            else:  #this part is done only at the init (once per module) to get us the correct dimensions we need to use later
                # the approach can be improved to get the size w/o computing the matrix, which is faster
                # but not a big deal: done only once
                if self.steps == 0:
                    aa = self.CovAHandler(input[0].data, module)
                    # save the size
                    self.size_of_missing_m_aa[module] = aa.size(0)
                    # initialize required EVD quantities correctly as zero 
                    #(could do this in the inversion funciton but it's best done here to avoid using torch.zeros)
                    actual_rank = min(aa.shape[0], self.rsvd_rank)
                    self.d_a[module] = 0 * aa[0,:actual_rank]; self.Q_a[module] = 0 * aa[:,:actual_rank] # Now we'll have Q_a's as skinnytall because
                    # we are using RSVD representation(lowrank) and thus we need to initialize our zeros accordngly
                
                elif isinstance(module, nn.Linear) and self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0:
                    # clear the Linear Layer B-representaiton to zeros to get proper result of allreduce sum
                    # if it's a linear layer and it's at the time we do the B-update on <<ONE>> of the GPU for afterwards sharing
                    actual_rank = min(self.d_a[module].shape[0], self.rsvd_rank)
                    self.d_a[module] = 0 * self.d_a[module]; self.Q_a[module] = 0 * self.Q_a[module]
                    # the conv modules are reset at _update_inv (and also the Linear modules at "R" update time)
                    

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            if module in self.modules_for_this_rank[self.rank]: # ONLY compute the Kfactor and update it if the GPU parsing this
                #is responsible for this aprticular module
                if self.K_fac_incoming_info_debugger_mode or self.dist_debugger_testing_leanness_thing:
                    print('RANK {} WORLDSIZE {}. At module {} \n ... the G size is {}\n'.format(self.rank, self.world_size, module, grad_output[0].data.shape))
                gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
                # Initialize buffers
                if self.steps == 0:
                    self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
                # here we initialize with identity and we'll move this to the reg term for R-KFAC and B-KFAC
                
                ############ also update the real AA for alter correction ########
                update_running_stat(gg, self.m_gg[module], self.stat_decay)
                
                ''' TO DO: make it possible to accumulate in incoming A (from AA^T)
                and only invert when the time comes : i.e. make it work for for TCov < TInv!'''
                
                ############ Brand - Update the RSVD of \bar{GG} ##############
                # if layer is linear and it's not time to do RSVD and it's time to do brand update
                if isinstance(module, nn.Linear) and self.steps % (self.TInv * self.brand_period) != 0 and self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0:
                    # If the layer is linear! and it's time to do the update
                    # g: batch_size * out_dim
                    #print('Updating GG^T')
                    n_dim = grad_output[0].data.shape[1]; c_dim = grad_output[0].data.shape[0]; r_dim = self.Q_g[module].shape[1]
                    if r_dim + c_dim < n_dim: # only if the new B_updated AA^T matrix is lowrank
                        batch_size = c_dim
                        if self.batch_averaged:
                            G = grad_output[0].data.T * (batch_size + 0.0)**0.5
                        else:
                            G = grad_output[0].data.T / (batch_size + 0.0)**0.5
                    
                        self.d_g[module], self.Q_g[module] = Brand_S_update(self.Q_g[module], self.stat_decay * self.d_g[module],
                                                                            A = self.sqr_1_minus_stat_decay * G, r_target = self.brand_r_target,
                                                                            device = torch.device('cuda:{}'.format(self.rank)) )
                    else:
                        oversampled_rank = min(self.m_gg[module].shape[0], self.total_rsvd_rank)
                        actual_rank = min(self.m_gg[module].shape[0], self.rsvd_rank)
                        self.d_g[module], self.Q_g[module] = RSVD_lowrank(M = self.m_gg[module], oversampled_rank = oversampled_rank, target_rank = actual_rank, niter = self.rsvd_niter, start_matrix = None)
            
                    # Ensure tensor Q_a is contiguous s.t. the allreduce can work! the tensor becoming noncontiguous can be due to transposition and it occurs in practice
                    self.Q_g[module] = self.Q_g[module].contiguous()
            else: # this part is done only at the init (once per module) to get us the correct dimensions we need to use later
                if self.steps == 0:
                    gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
                    # save the size
                    self.size_of_missing_m_gg[module] = gg.size(0)
                    # initialize required EVD quantities correctly as zero 
                    #(could do this in the inversion funciton but it's best done here to avoid using torch.zeros)
                    actual_rank = min(gg.shape[0], self.rsvd_rank)
                    self.d_g[module] = 0 * gg[0,:actual_rank]; self.Q_g[module] = 0 * gg[:,:actual_rank] # Now we'll have Q_g's as skinnytall because
                    # we are using RSVD representation(lowrank) and thus we need to initialize our zeros accordngly
                elif isinstance(module, nn.Linear) and self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0:
                    # clear the Linear Layer B-representaiton to zeros to get proper result of allreduce sum
                    # if it's a linear layer and it's at the time we do the B-update on <<ONE>> of the GPU for afterwards sharing
                    actual_rank = min(self.d_g[module].shape[0], self.rsvd_rank)
                    self.d_g[module] = 0 * self.d_g[module]; self.Q_g[module] = 0 * self.Q_g[module]
                    # the conv modules are reset at _update_inv (and also the Linear modules at "R" update time)
                    

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1
        
        # construct self.modules_for_this_rank (a dictonary of lists] - which tells us which modules's EVD  are computed by which GPU
        self.modules_for_this_rank = fct_split_list_of_modules(self.modules, self.world_size)
        # returns a dictionary of lists!

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        if m in self.modules_for_this_rank[self.rank]:
            ### PARALLELIZE OVER layers: for each GPU-RANK compute only the EVD's of the "some" layers
            """Do eigen decomposition for computing inverse of the ~ fisher.
            :param m: The layer
            :return: no returns.
            """
            eps = 1e-10  # for numerical stability
            #self.d_a[m], self.Q_a[m] = torch.symeig( self.m_aa[m] + 0.01* torch.eye(self.m_aa[m].shape[0], device = torch.device('cuda:0'))  , eigenvectors=True)
            oversampled_rank = min(self.m_aa[m].shape[0], self.total_rsvd_rank)
            actual_rank = min(self.m_aa[m].shape[0], self.rsvd_rank)
            self.d_a[m], self.Q_a[m] = RSVD_lowrank(M = self.m_aa[m], oversampled_rank = oversampled_rank, target_rank = actual_rank, niter = self.rsvd_niter, start_matrix = None)
        
            #self.d_g[m], self.Q_g[m] = torch.symeig( self.m_gg[m] , eigenvectors=True) # computes the eigen decomposition of bar_A and G matrices
            oversampled_rank = min(self.m_gg[m].shape[0], self.total_rsvd_rank)
            actual_rank = min(self.m_gg[m].shape[0], self.rsvd_rank)
            self.d_g[m], self.Q_g[m] = RSVD_lowrank(M = self.m_gg[m], oversampled_rank = oversampled_rank, target_rank = actual_rank, niter = self.rsvd_niter, start_matrix = None)
    
            self.d_a[m].mul_((self.d_a[m] > eps).float())
            self.d_g[m].mul_((self.d_g[m] > eps).float())
            
            #### MAKE TENSORS CONTIGUOUS s.t. the ALLREDUCE OPERATION CAN WORK (does nto take that much!)
            self.Q_a[m] = self.Q_a[m].contiguous()
            self.Q_g[m] = self.Q_g[m].contiguous() # D's are already contiguous as tey were not transposed!
            if self.dist_comm_for_layers_debugger:
                print('RANK {} WORLDSIZE {}. computed EVD of module {} \n'.format(self.rank, self.world_size, m))
                print('The shapes are Q_a.shape = {}, d_a.shape = {}, Q_q.shape = {}, d_g.shape = {}'. format(self.Q_a[m].shape, self.d_a[m].shape, self.Q_g[m].shape,self.d_g[m].shape))
        else:
            ### PARALLELIZE OVER layers: Set uncomputed quantities to zero to allreduce with SUM 
            #if len(self.d_a) == 0: # if it's the 1st time we encouter these guys (i.e. at init during 1st evd computation before 1st allreduction)
            self.d_a[m] = 0 * self.d_a[m];  self.Q_a[m] = 0 * self.Q_a[m]
            self.d_g[m] = 0 * self.d_g[m];  self.Q_g[m] = 0 * self.Q_g[m]

        
    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        ######
        v1 = X_reg_inverse_M_adaptive_damping(U = self.Q_g[m], D = self.d_g[m], M = p_grad_mat, lambdda = damping, damping_type = self.damping_type) # the damping here is adaptive!
        v = M_X_reg_inverse_adaptive_damping(U = self.Q_a[m], D = self.d_a[m], M = v1, lambdda = damping, damping_type = self.damping_type)  # the damping here is adaptive!
        
        '''v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()'''
        
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        if self.clip_type == 'standard':
                # do kl clip
            vg_sum = 0
            for m in self.modules:
                v = updates[m]
                vg_sum += torch.abs( (v[0] * m.weight.grad.data * lr ** 2).sum()) #.item()
                if m.bias is not None:
                    vg_sum += torch.abs( (v[1] * m.bias.grad.data * lr ** 2).sum() ) #.item()
            nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))
    
            for m in self.modules:
                v = updates[m]
                m.weight.grad.data.copy_(v[0])
                m.weight.grad.data.mul_(nu)
                if m.bias is not None:
                    m.bias.grad.data.copy_(v[1])
                    m.bias.grad.data.mul_(nu)
        else:
            for m in self.modules:
                v = updates[m]
                # ipdb.set_trace(context = 7)
                if m.bias is not None:
                    concat_V = torch.cat( (v[0].flatten(), v[1].flatten() ) )
                    numel_v = torch.numel(concat_V)
                    nu = min(1, self.kl_clip/(torch.norm( concat_V, p=2 )/math.sqrt(numel_v)))
                else:
                    nu = min(1, self.kl_clip/(torch.norm(v[0], p = 2)/math.sqrt(torch.numel(v[0]))))
                m.weight.grad.data.copy_(v[0])
                m.weight.grad.data.mul_(nu)
                if m.bias is not None:
                    m.bias.grad.data.copy_(v[1])
                    m.bias.grad.data.mul_(nu)
                
    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_( - group['lr'], d_p)

    def step(self, epoch_number, error_savepath, closure = None):
        
        #############################################################################################
        #### NO MORE NEED TO allreduce if AA^T and GG^T statistics have been updated locally
        #############################################################################################
        
        self.epoch_number = epoch_number
        self.lr = self.lr_function(epoch_number, self.steps)
        for g in self.param_groups:
            g['lr'] = self.lr_function(epoch_number, self.steps)
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        
        ##### these 2 loops used to be just 1 when no distributed
        ## 1) if it's time to recompute EVD, recompute EVD (set zeros to parts where I don't recompute for a given rank)
        if self.steps % self.TInv == 0:
            for m in self.modules:
                if isinstance(m, nn.Linear): # if the layer at hand is linear, we ony do RSVD at a frequency x(self.brand_period) slower
                    # than what we do in Conv layers. Set self.brand_period=1 to get pure RS-KFAC, and self.brand_period = inf to get pure B-KFAC, else we're in between the two
                    if self.steps % (self.TInv * self.brand_period) == 0: # only every (self.TInv * self.brand_period) we do RS-KFAC update to Kfactors
                        # but we still have lightler B-updates whenever we get new K-factors info
                        self._update_inv(m)
                    else:
                        pass
                else: # if it's not a linear layer, do RSVD every TInv iterations (No brand update for Conv Layers)
                    self._update_inv(m)
                    
                
        
        # take the step and allreduce across evd's if the inverses were updated    
        for m in self.modules:
            classname = m.__class__.__name__
            if (not isinstance(m, nn.Linear)) and (self.steps % self.TInv == 0):
                # if the inversion was done locally this turn, allreduce to disseminate inverse representation
                #if it's time to recompute inverse for Conv layers or for liear (BRAND) layers
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. Before Allreduce d_a={}, Q_g = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_a[m], self.Q_g[m]))
                print('RANK {}. Doing line: dist.all_reduce(self.d_a[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                dist.all_reduce(self.d_a[m], dist.ReduceOp.SUM, async_op = False)
                #self.d_a[m] = 0 * self.d_a[m] + 1

                print('RANK {}. Doing line : dist.all_reduce(self.Q_a[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)
                #self.Q_a[m] = 0 * self.Q_a[m]; Q_debug_size = min(self.Q_a[m].shape[0],self.Q_a[m].shape[1])
                #self.Q_a[m][torch.arange(Q_debug_size),torch.arange(Q_debug_size)] = 1 # make Q_g identity to avoid comunication

                print('RANK {}. Doing line : dist.all_reduce(self.d_g[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                dist.all_reduce(self.d_g[m], dist.ReduceOp.SUM, async_op = False)
                #self.d_g[m] = 0 * self.d_g[m] + 1

                print('RANK {}. DOING LINE: dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)
                #self.Q_g[m] = 0 * self.Q_g[m]; Q_debug_size = min(self.Q_g[m].shape[0],self.Q_g[m].shape[1])
                #self.Q_g[m][torch.arange(Q_debug_size),torch.arange(Q_debug_size)] = 1 # make Q_g identity to avoid comunication
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. AFTER Allreduce d_a={}, Q_g = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_a[m], self.Q_g[m]))
            elif isinstance(m, nn.Linear) and (self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0):
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. Before Allreduce d_a={}, Q_g = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_a[m], self.Q_g[m]))
                print('RANK {}. Doing line: dist.all_reduce(self.d_a[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                dist.all_reduce(self.d_a[m], dist.ReduceOp.SUM, async_op = False)
                #self.d_a[m] = 0 * self.d_a[m] + 1

                print('RANK {}. Doing line : dist.all_reduce(self.Q_a[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                #dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)
                self.Q_a[m] = 0 * self.Q_a[m]; Q_debug_size = min(self.Q_a[m].shape[0],self.Q_a[m].shape[1])
                self.Q_a[m][torch.arange(Q_debug_size),torch.arange(Q_debug_size)] = 1 # make Q_g identity to avoid comunication

                print('RANK {}. Doing line : dist.all_reduce(self.d_g[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                #dist.all_reduce(self.d_g[m], dist.ReduceOp.SUM, async_op = False)
                self.d_g[m] = 0 * self.d_g[m] + 1

                print('RANK {}. DOING LINE: dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                #dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)
                self.Q_g[m] = 0 * self.Q_g[m]; Q_debug_size = min(self.Q_g[m].shape[0],self.Q_g[m].shape[1])
                self.Q_g[m][torch.arange(Q_debug_size),torch.arange(Q_debug_size)] = 1 # make Q_g identity to avoid comunication
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. AFTER Allreduce d_a={}, Q_g = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_a[m], self.Q_g[m]))
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1


