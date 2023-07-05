import math

import torch
import torch.optim as optim
import torch.distributed as dist

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!
#sys.path.append('/home/chri5570/Distributed_Brand_and_Randomized_KFACs/') 

from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.kfac_utils import (ComputeCovA, ComputeCovG)
from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.kfac_utils import update_running_stat
from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.kfac_utils import fct_split_list_of_modules
from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.solver_workload_allocation_utils import allocate_EVD_inversion_work


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 rank, world_size, batch_size,
                 lr_function = lambda epoch_n, iteration_n: 0.1,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 work_alloc_propto_EVD_cost = True):
        
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.epoch_number = 1
        
        self.batch_size = batch_size
        self.lr_function = lr_function
        self.lr = self.lr_function(epoch_n = self.epoch_number, n_GPUs = world_size,
                                   batch_size = batch_size, iter_n = 0 )
        defaults = dict(lr = self.lr, 
                        momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        
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
        
        #for it_rank in range(0, world_size):
        #    self.modules_for_this_rank[it_rank] = [] # a dictionary of lists 
        # modules will get appended to list accordingly, later at preparing model
        
        # debug flags
        self.K_fac_incoming_info_debugger_mode = False
        self.Dist_communication_debugger = False
        self.dist_communication_2nd_version_debugger = False
        self.dist_comm_for_layers_debugger = False
        self.dist_debugger_testing_leanness_thing = False
        
        ## efficient wor allocation stuff
        self.work_alloc_propto_EVD_cost = work_alloc_propto_EVD_cost
        self.size_0_of_all_Kfactors_A = {} #once obtained, save for later usage
        self.size_0_of_all_Kfactors_G = {} #once obtained, save for later usage
        self.modules_for_this_rank_A = {} # the output of work-schedulling across GPUs for A KFACTORS
        self.modules_for_this_rank_G = {} # the output of work-schedulling across GPUs for G KFACTORS

        self._prepare_model()

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            if module in self.modules_for_this_rank_A[self.rank]: # ONLY compute the Kfactor and update it if the GPU parsing this
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
                    self.size_0_of_all_Kfactors_A[module] = aa.size(0)
                elif self.steps == self.TCov and (module not in self.old_modules_for_this_rank_A[self.rank]): # we could also say "not in self.m_aa[module], but this has less control over the situation
                    #the first time we enter here after the efficient allocation is at step number self.TCov 
                    self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
                    # we are reinitializing the modules which got newly allocated to *this GPU but were not allocated to it before
                    # we could instead choose to communicate the self.m_aa from the GPU that took care of it before, but we avoid doing so to minimize communication.
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
                
                update_running_stat(aa, self.m_aa[module], self.stat_decay)
                
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
                    self.size_of_missing_m_aa[module] = aa.size(0) # this may be a slight redudacy, may remove in later versions
                    self.size_0_of_all_Kfactors_A[module] = aa.size(0)
                    # initialize required EVD quantities correctly as zero 
                    #(could do this in the inversion funciton but it's best done here to avoid using torch.zeros)
                    self.d_a[module] = 0 * aa[0,:]; self.Q_a[module] = 0 * aa
                    

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            if module in self.modules_for_this_rank_G[self.rank]: # ONLY compute the Kfactor and update it if the GPU parsing this
                #is responsible for this aprticular module
                if self.K_fac_incoming_info_debugger_mode or self.dist_debugger_testing_leanness_thing:
                    print('RANK {} WORLDSIZE {}. At module {} \n ... the G size is {}\n'.format(self.rank, self.world_size, module, grad_output[0].data.shape))
                gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
                # Initialize buffers
                if self.steps == 0:
                    self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
                    self.size_0_of_all_Kfactors_G[module] = gg.size(0)
                elif self.steps == self.TCov and (module not in self.old_modules_for_this_rank_G[self.rank]): # we could also say "not in self.m_aa[module], but this has less control over the situation
                    #the first time we enter here after the efficient allocation is at step number self.TCov 
                    self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
                    # we are reinitializing the modules which got newly allocated to *this GPU but were not allocated to it before
                    # we could instead choose to communicate the self.m_aa from the GPU that took care of it before, but we avoid doing so to minimize communication.
                    # here we initialize with identity and we'll move this to the reg term for R-KFAC and B-KFAC
                # here we initialize with identity and we'll move this to the reg term for R-KFAC and B-KFAC
                update_running_stat(gg, self.m_gg[module], self.stat_decay)
            else: # this part is done only at the init (once per module) to get us the correct dimensions we need to use later
                if self.steps == 0:
                    gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
                    # save the size
                    self.size_of_missing_m_gg[module] = gg.size(0) # this may be a slight redudacy, may remove in later versions
                    self.size_0_of_all_Kfactors_G[module] = gg.size(0)
                    # initialize required EVD quantities correctly as zero 
                    #(could do this in the inversion funciton but it's best done here to avoid using torch.zeros)
                    self.d_g[module] = 0 * gg[0,:]; self.Q_g[module] = 0 * gg
                    

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
                module.register_full_backward_hook(self._save_grad_output) # deprecated: # module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1
        
        ### WORK ALLOCATION temporary (for self.steps ==0 only)... or not! depending on choice
        # IMPORTANT: USING THE TRIVIAL ALLOCATION MECHANISM ON THE 1st FACTOR COMPUTATION
        # HTIS IS BECAUSE WE CAN'T ACCESS DIMENSIONS UNLESS A PASS HAS BEEN DONE: can't access layer params if net is "black-box" (not defined by us)
        # THUS, we first do a trivial number-of layers based allocation (i.e. assuming they're all the same) 
        # and after that, IF SELECTED SO THROUGH HYPERPARAM work_alloc_propto_RSVD_cost, we'll swtich to a more efficient one, once a pass has been done
        # construct self.modules_for_this_rank (a dictonary of lists] - which tells us which modules's EVD  are computed by which GPU
        self.modules_for_this_rank_A = self.modules_for_this_rank_G = fct_split_list_of_modules(self.modules, self.world_size) # returns a dictionary of lists!
        # call the same fct for A and G to get the same TRIVIAL split in both A and G: that boils down to being a module-split rather than a KFACTOR split
        # returns a dictionary of lists!
        ### Commented the below for speed, uncomment or add verboise option to better see how work allocation is performed
        #print('Split work in TRIVIAL fashion as: self.modules_for_this_rank_A = {} \n self.modules_for_this_rank_G = {}'.format(self.modules_for_this_rank_A, self.modules_for_this_rank_G))
        #print('The following sentece is {} : We will also improve the allocation from the 2nd KFACTOR work onwards (at end of step 0)'.format(self.work_alloc_propto_EVD_cost))
        

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        ### PARALLELIZE OVER layers: for each GPU-RANK compute only the EVD's of the "some" KFACTORS
        # ================ AA^T KFACTORS ===================================
        if m in self.modules_for_this_rank_A[self.rank]:
            if self.dist_comm_for_layers_debugger:
                print('RANK {} WORLDSIZE {}. computed "A"-EVD of module {} \n'.format(self.rank, self.world_size, m))
            
            eps = 1e-10  # for numerical stability
            self.d_a[m], self.Q_a[m] = torch.linalg.eigh(self.m_aa[m], UPLO = 'U') #deprecated # torch.symeig( self.m_aa[m], eigenvectors=True)
            self.d_a[m].mul_((self.d_a[m] > eps).float())
            #### MAKE TENSORS CONTIGUOUS s.t. the ALLREDUCE OPERATION CAN WORK (does nto take that much!)
            self.Q_a[m] = self.Q_a[m].contiguous()
        else:
            ### PARALLELIZE OVER layers: Set uncomputed quantities to zero to allreduce with SUM 
            #if len(self.d_a) == 0: # if it's the 1st time we encouter these guys (i.e. at init during 1st evd computation before 1st allreduction)
            self.d_a[m] = 0 * self.d_a[m];  self.Q_a[m] = 0 * self.Q_a[m]
        # ==== END ======= AA^T KFACTORS ===================================
        
        # ================ GG^T KFACTORS ===================================
        if m in self.modules_for_this_rank_G[self.rank]:  
            if self.dist_comm_for_layers_debugger:
                print('RANK {} WORLDSIZE {}. computed "G"-EVD of module {} \n'.format(self.rank, self.world_size, m))
            eps = 1e-10  # for numerical stability
            self.d_g[m], self.Q_g[m] = torch.linalg.eigh(self.m_gg[m], UPLO = 'U') #deprecated #torch.symeig( self.m_gg[m], eigenvectors=True)
            self.d_g[m].mul_((self.d_g[m] > eps).float())
            #### MAKE TENSORS CONTIGUOUS s.t. the ALLREDUCE OPERATION CAN WORK (does nto take that much!)
            self.Q_g[m] = self.Q_g[m].contiguous() # D's are already contiguous as tey were not transposed!
        else:
            ### PARALLELIZE OVER layers: Set uncomputed quantities to zero to allreduce with SUM 
            #if len(self.d_a) == 0: # if it's the 1st time we encouter these guys (i.e. at init during 1st evd computation before 1st allreduction)
            self.d_g[m] = 0 * self.d_g[m];  self.Q_g[m] = 0 * self.Q_g[m]
        # ====== END ===== GG^T KFACTORS ===================================

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
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        damping = damping * torch.max(self.d_g[m]) * torch.max(self.d_a[m])
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
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
        '''# do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += torch.abs((v[0] * m.weight.grad.data * lr ** 2).sum()) #.item()
            if m.bias is not None:
                vg_sum += torch.abs((v[1] * m.bias.grad.data * lr ** 2).sum()) #.item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))'''
        
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
                    d_p.add_(p.data, alpha = weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p , alpha = 1)
                    d_p = buf

                p.data.add_( d_p , alpha = - group['lr'])

    def step(self, epoch_number, error_savepath, closure = None):
        
        #############################################################################################
        #### NO MORE NEED TO allreduce if AA^T and GG^T statistics have been updated locally
        #############################################################################################
        
        self.epoch_number = epoch_number
        self.lr = self.lr_function(epoch_n = self.epoch_number, n_GPUs = self.world_size,
                                   batch_size = self.batch_size, iter_n = self.steps )
        for g in self.param_groups:
            g['lr'] = self.lr
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        
        ##### these 2 loops used to be just 1 when no distributed
        ## 1) if it's time to recompute EVD, recompute EVD (set zeros to parts where I don't recompute for a given rank)
        if self.steps % self.TInv == 0:
            for m in self.modules:
                self._update_inv(m)
        
        # take the step and allreduce across evd's if the inverses were updated    
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0: # if the inversion was done locally this turn, allreduce to disseminate inverse representation
                if self.dist_comm_for_layers_debugger:
                    print('RANK {} WORLDSIZE {} MODULE {}. Before Allreduce d_a={}, Q_g = {} \n'.format(self.rank, self.world_size, m, self.d_a[m], self.Q_g[m]))
                dist.all_reduce(self.d_a[m], dist.ReduceOp.SUM, async_op = False)
                dist.all_reduce(self.Q_a[m], dist.ReduceOp.SUM, async_op = False)
                dist.all_reduce(self.d_g[m], dist.ReduceOp.SUM, async_op = False)
                dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)
                if self.dist_comm_for_layers_debugger:
                    print('RANK {} WORLDSIZE {} MODULE {}. AFTER Allreduce d_a={}, Q_g = {} \n'.format(self.rank, self.world_size, m, self.d_a[m], self.Q_g[m]))
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)
        
        #### change work allocation to dimension-based for RSVD
        if self.steps == 0 and self.work_alloc_propto_EVD_cost == True: # allocate work over KFACTORS in proportion to RSVD cost
                # output of allocate_RSVD_inversion_work_same_fixed_r
                # dict_of_lists_of_responsibilities_A = a dictionary where the key is the wwrker number 
                # and the value is the list of all modules that particular worker is responsible for at Kfactor AA^T
                # dict_of_lists_of_responsibilities_G = a dictionary where the key is the wwrker number 
                # and the value is the list of all modules that particular worker is responsible for at Kfactor GG^T
                new_modules_for_this_rank_A, new_modules_for_this_rank_G, _ = allocate_EVD_inversion_work(number_of_workers = self.world_size, 
                                                                                    size_0_of_all_Kfactors_G = self.size_0_of_all_Kfactors_G,
                                                                                    size_0_of_all_Kfactors_A = self.size_0_of_all_Kfactors_A)
                ### delete and initialize Q[m], d[m] and m_aa/m_gg[m] to accommodate reallocation
                #### 1. delete what's in OLD but NOT in new
                for key_A_old in self.modules_for_this_rank_A[self.rank]:
                    if key_A_old not in new_modules_for_this_rank_A[self.rank]:
                        # the next line CAN be omitted because we zero them out anyway during _update_inv; but we leave it here to remind ourselves which qunatities are relevant
                        # self.d_a[key_A_old] = 0 * self.d_a[key_A_old]; self.Q_a[key_A_old] = 0 * self.Q_a[key_A_old]
                        del self.m_aa[key_A_old]
                for key_G_old in self.modules_for_this_rank_G[self.rank]:
                    if key_G_old not in new_modules_for_this_rank_G[self.rank]:
                        # the next line CAN be omitted because we zero them out anyway during _update_inv; but we leave it here to remind ourselves which qunatities are relevant
                        # self.d_g[key_G_old] = 0 * self.d_g[key_G_old]; self.Q_g[key_G_old] = 0 * self.Q_g[key_G_old]
                        del self.m_gg[key_G_old]
                #### 2. initialize what's in NEW but NOT in old (and thus does nto exist)
                ## we do this in save_inuput and _save_grad_output hooks
                ## but in order to do that we need to rememeber the old keys first
                self.old_modules_for_this_rank_A = self.modules_for_this_rank_A
                self.old_modules_for_this_rank_G = self.modules_for_this_rank_G
                self.modules_for_this_rank_A = new_modules_for_this_rank_A
                self.modules_for_this_rank_G = new_modules_for_this_rank_G
                
                #comment the below prints for less spam and slightly better speed
                #print(' self.work_alloc_propto_EVD_cost was set to TRUE, so at the very end of self.steps == 0, we reallocated work in proportion to squared-size')
                #print(' as given by: self.modules_for_this_rank_A = {} \n self.modules_for_this_rank_G = {}'.format(self.modules_for_this_rank_A, self.modules_for_this_rank_G))
        #### END : change work allocation to dimension-based for RSVD
        
        self._step(closure)
        self.steps += 1


