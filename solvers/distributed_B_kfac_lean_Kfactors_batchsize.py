import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import sys
sys.path.append('/home/chri5570/') # add your own path to *this github repo here!
#sys.path.append('/home/chri5570/Distributed_Brand_and_Randomized_KFACs/') 

from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.kfac_utils_for_vgg16_bn import (ComputeCovA, ComputeCovG)
from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.kfac_utils_for_vgg16_bn import update_running_stat
from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.kfac_utils_for_vgg16_bn import fct_split_list_of_modules

from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.Brand_S_subroutine import Brand_S_update
from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.solver_LA_utils import (X_reg_inverse_M_adaptive_damping, M_X_reg_inverse_adaptive_damping, RSVD_lowrank)

from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.solver_workload_allocation_utils import allocate_RSVD_inversion_work_same_fixed_r
from Distributed_Brand_and_Randomized_KFACs.solvers.solver_utils.solver_workload_allocation_utils import allocate_B_inversion_work_same_fixed_r_and_batchsize

class B_KFACOptimizer(optim.Optimizer):
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
                 brand_r_target_excess = 0,
                 brand_update_multiplier_to_TCov = 1,
                 work_alloc_propto_RSVD_and_B_cost = True):
        
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.epoch_number = 1
        defaults = dict(lr = lr_function(self.epoch_number, 0), 
                        momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(B_KFACOptimizer, self).__init__(model.parameters(), defaults)
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

        self.m_aa, self.m_gg = {}, {} # these dictionaries will be populated just for VERY tiny linear layers, or for CONV layers (below r_target + n_BS_per_GPU) K-factors
        # the other k-factors (large ones) will not be stored, and only B-update will be done to them.
        self.Q_a, self.Q_g = {}, {} 
        self.d_a, self.d_g = {}, {}
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
        self.debug_size_for_B = False
        
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
        #print('type(brand_r_target_excess) = {}, brand_r_target_excess = {}'.format(type(brand_r_target_excess),brand_r_target_excess))
        #print('type of rsvd_rank is {}'.format(type(rsvd_rank)))
        self.brand_r_target = rsvd_rank + brand_r_target_excess
        self.brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov
        self.sqr_1_minus_stat_decay = (1 - stat_decay)**(0.5) # to avoid recomputations
        self.batch_size = None
        #######################################################################
        
        #### for tracking which modules are on Brand track and whicha ren't
        self.size_of_missing_m_aa = {} # dictionary tracking the size of lazy AA^T kfactors 
        self.size_of_missing_m_gg = {} # dictionary tracking the size of lazy GG^T kfactors 
        self.size_of_nonlazy_Kfactors_a = {} # dictionary tracking the size of NON-lazy & non-brand-tracked AA^T kfactors 
        self.size_of_nonlazy_Kfactors_g = {} # dictionary tracking the size of NON-lazy & non-brand-tracked AA^T kfactors 
        
        ################ for efficient work allocation
        self.work_alloc_propto_RSVD_and_B_cost = work_alloc_propto_RSVD_and_B_cost
        ### dict for sizes of all modules, split by LL vs CaSL cathegory
        self.size_0_of_LL_Kfactors_A = {} # here, LL refers to large linear layers, large is in the sense that "B-update makes sense", loosely speaking
        self.size_0_of_LL_Kfactors_G = {} # and the B-update makes sense if brand_target_rank +nBS < shape of KFACTOR
        self.size_0_of_CaSL_Kfactors_A = {} # hare CaSL = "Conv and Small linear layers"
        self.size_0_of_CaSL_Kfactors_G = {} # small is defined as non-large for large defined as 2 lines above
        
        ### allocation of work for Casl and LL KFACTORS sepparately
        self.LL_modules_for_this_rank_A = {} # the output of work-schedulling across GPUs for A KFACTORS: relevant to  B-updates
        self.LL_modules_for_this_rank_G = {} # the output of work-schedulling across GPUs for G KFACTORS: relevant to  B-updates
        self.CaSL_modules_for_this_rank_A = {} # the output of work-schedulling across GPUs for A KFACTORS: relevant to R-updates
        self.CaSL_modules_for_this_rank_G = {} # the output of work-schedulling across GPUs for G KFACTORS: relevant to R-updates
        self.initalloc_modules_for_this_rank_A = {}
        self.initalloc_modules_for_this_rank_G = {} # these latter 2 dictionaries are only used at the 0th step: to deal with the fact that we don't knwo what layer is LL and what layer is not LL at 0th step (unless another pass is made, which we avoid as it's cosntly)
        
        ### number of Kfactors updates stored for each kfactor: dictionaries
        self.nkfu_dict_a = {} # these will be different on each GPU, but that's not a concern. Each GPU will have the correct values for its tracked modules
        self.nkfu_dict_g = {}
        
        self._prepare_model()
        
    def _save_input(self, module, input):
        if self.batch_size == None and isinstance(module, nn.Linear): ### save batchsize
            self.batch_size = input[0].data.shape[0]
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            #########################################################
            if self.steps == 0:
                # at the 0th step, we don;t know yet which layers are LL and which are otherwise
                # so we have to treat this case sepparately: 1. we perform "trivial allocation" across modules rather than KFACTORS
                # then 2. if we wisht o stick to the trivial allocation, we merely split this trivial allocation dict into the 2
                # dictionaries self.LL_modules_for_this_rank_A and self.CaSL_modules_for_this_rank_A at the very end of step 0
                # otherwise, we recompute "efficient" realllocations at the end of step 0
                # both these approaches in 2 require us to know which layer is LL and whic isn;t, and the latter requires the size of the layer, which we get at self.steps ==0, and that's why we treat it sepparately
                aa = self.CovAHandler(input[0].data, module)
                self.nkfu_dict_a[module] = 1
                
                #### see whether the module is LL or not, and save the size to the corresponding dicitonary  ###########
                # we save sizes for ALL modules, not just for the ones allocated to *this GPU
                # note that by saving the size in the correct dictionary we also implicitly sve info about whether one layer is LL or is CaSL
                if isinstance(module, nn.Linear) and (self.brand_r_target + input[0].data.shape[0] < aa.shape[0]):
                    self.size_0_of_LL_Kfactors_A[module] = aa.shape[0] 
                else:
                    self.size_0_of_CaSL_Kfactors_A[module] = aa.shape[0] 
                ##### end: save module size to the correct dictioanry depending on whether "is LL" or not ##############
                    
                if module in self.size_0_of_LL_Kfactors_A: # the keys of this dict are all the LL layers
                    if module in self.initalloc_modules_for_this_rank_A[self.rank]: # for LL layers alloc to *this GPU, perform B update
                        # initialize for brand-update first
                        self.d_a[module] = 0 * aa[0, :self.brand_r_target] # initialize with a rank-brand_target_rank null tensor for minila computational effort!
                        self.Q_a[module] = 0 * aa[:, :self.brand_r_target] # can't do a mere rank 1 as that would give wrong size at communication
                        self.Q_a[module][range(0, self.brand_r_target), range(0, self.brand_r_target)] = 1
                        ############ BRAND UPDATE ############################
                        batch_size = input[0].data.shape[0] #  c_dim = input[0].data.shape[0]
                        A = input[0].data / (batch_size + 0)**(0.5)
                        if module.bias is not None:
                            #print('\n Updating AA^T: we have bias in linear layer!')
                            A = torch.cat([A, A.new(A.size(0), 1).fill_(1)], 1).T
        
                        self.d_a[module], self.Q_a[module] = Brand_S_update(self.Q_a[module], self.stat_decay * self.d_a[module],
                                                                            A = self.sqr_1_minus_stat_decay * A, r_target = self.brand_r_target, 
                                                                            device = torch.device('cuda:{}'.format(self.rank)) )
                        ########### END BRAND UPDATE #########################
                    else: # else, if the layer is LL but not alloc to *this GPU, just initilalize correct shapes
                        actual_rank = self.rsvd_rank + input[0].data.shape[0]# NOTE: # batch_size = input[0].data.shape[0]
                        self.d_a[module] = 0 * aa[0,:actual_rank]; self.Q_a[module] = 0 * aa[:,:actual_rank] # Now we'll have Q_a's as skinnytall because
                        # we are using RSVD representation(lowrank) and thus we need to initialize our zeros accordngly
                            
                elif module in self.size_0_of_CaSL_Kfactors_A: # the keys of this dict are all the CaSL layers
                    # for non LL layers, save m_aa and pass (inversion will happen later)
                    #strictly speaking this elif could be simple "else" if all works correctly elsewhere (because the 2 sets are a partition of the total set of registered modules), but we use elif to make it explicit
                    if module in self.initalloc_modules_for_this_rank_A[self.rank]:
                        self.m_aa[module] = (1 - self.stat_decay) * aa + 0
                    else:
                        actual_rank = min(aa.shape[0], self.rsvd_rank)
                        self.d_a[module] = 0 * aa[0,:actual_rank]; self.Q_a[module] = 0 * aa[:,:actual_rank] # Now we'll have Q_a's as skinnytall because
                        # we are using RSVD representation(lowrank) and thus we need to initialize our zeros accordngly
               
            elif module in self.LL_modules_for_this_rank_A[self.rank]:
                if self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0:
                    # if the module is the responsibility of *this GPU, AND the module is on Brand-track, and it's time to Brand-update
                    ############ BRAND UPDATE ############################
                    batch_size = input[0].data.shape[0] #  c_dim = input[0].data.shape[0]
                    A = input[0].data / (batch_size + 0)**(0.5)
                    if module.bias is not None:
                        #print('\n Updating AA^T: we have bias in linear layer!')
                        A = torch.cat([A, A.new(A.size(0), 1).fill_(1)], 1).T
    
                    self.d_a[module], self.Q_a[module] = Brand_S_update(self.Q_a[module], self.stat_decay * self.d_a[module],
                                                                        A = self.sqr_1_minus_stat_decay * A, r_target = self.brand_r_target, 
                                                                        device = torch.device('cuda:{}'.format(self.rank)) )
                    self.nkfu_dict_a[module] += 1
                    ########### END BRAND UPDATE #########################
            elif module in self.CaSL_modules_for_this_rank_A[self.rank]:
                # if the module is the responsibility of *this GPU, AND the module is NOT on Brand-track
                #### update m_aa###############################################
                aa = self.CovAHandler(input[0].data, module)
                update_running_stat(aa, self.m_aa[module], self.stat_decay)
                self.nkfu_dict_a[module] += 1
            else: 
                # if the module is NOT the responsibility of *this GPU AT ALL!
                if module in self.size_0_of_LL_Kfactors_A:
                    if self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0: 
                        # NOTE: the keys to self.size_0_of_LL_Kfactors_A are all the brand-tacked linear layers, ie "LL" layers
                        # if the KFACTOR is LL and some other GPU does the brand-update of it: restart Q_a and d_a to zeros
                        self.d_a[module] = 0 * self.d_a[module]; self.Q_a[module] = 0 * self.Q_a[module]
                        self.nkfu_dict_a[module] += 1
                elif module in self.size_0_of_CaSL_Kfactors_A: #elif module not in self.size_0_of_LL_Kfactors_A:
                    self.nkfu_dict_a[module] += 1 # do nothing if the KFACTOR is not LL, as this gets reset in update_inv metod at the correct time
            #########################################################

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            if self.steps == 0:
                # at the 0th step, we don;t know yet which layers are LL and which are otherwise
                # so we have to treat this case sepparately: 1. we perform "trivial allocation" across modules rather than KFACTORS
                # then 2. if we wisht o stick to the trivial allocation, we merely split this trivial allocation dict into the 2
                # dictionaries self.LL_modules_for_this_rank_A and self.CaSL_modules_for_this_rank_A at the very end of step 0
                # otherwise, we recompute "efficient" realllocations at the end of step 0
                # both these approaches in 2 require us to know which layer is LL and whic isn;t, and the latter requires the size of the layer, which we get at self.steps ==0, and that's why we treat it sepparately
                gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
                self.nkfu_dict_g[module] = 1
                #### see whether the module is LL or not, and save the size to the corresponding dicitonary  ###########
                # we save sizes for ALL modules, not just for the ones allocated to *this GPU
                # note that by saving the size in the correct dictionary we also implicitly sve info about whether one layer is LL or is CaSL
                if isinstance(module, nn.Linear) and (self.brand_r_target + grad_output[0].data.shape[0] < gg.shape[0]):
                    self.size_0_of_LL_Kfactors_G[module] = gg.shape[0] 
                else:
                    self.size_0_of_CaSL_Kfactors_G[module] = gg.shape[0] 
                ##### end: save module size to the correct dictioanry depending on whether "is LL" or not ##############
                    
                if module in self.size_0_of_LL_Kfactors_G: 
                    if module in self.initalloc_modules_for_this_rank_G[self.rank]: # for LL layers alloc to *this GPU, perform B update
                        # initialize for brand-update first
                        self.d_g[module] = 0 * gg[0, :self.brand_r_target] # initialize with a rank-brand_target_rank null tensor for minila computational effort!
                        self.Q_g[module] = 0 * gg[:, :self.brand_r_target] # can't do a mere rank 1 as that would give wrong size at communication
                        self.Q_g[module][range(0, self.brand_r_target), range(0, self.brand_r_target)] = 1
                        ############ BRAND UPDATE ############################
                        batch_size =  grad_output[0].data.shape[0] # c_dim =batch_size
                        if self.batch_averaged:
                            G = grad_output[0].data.T * (batch_size + 0.0)**0.5
                        else:
                            G = grad_output[0].data.T / (batch_size + 0.0)**0.5
                    
                        self.d_g[module], self.Q_g[module] = Brand_S_update(self.Q_g[module], self.stat_decay * self.d_g[module],
                                                                            A = self.sqr_1_minus_stat_decay * G, r_target = self.brand_r_target,
                                                                            device = torch.device('cuda:{}'.format(self.rank)) )
                        
                        ########### END BRAND UPDATE #########################
                    else: # else, if the layer is LL but not alloc to *this GPU, just initilalize correct shapes
                        actual_rank = self.rsvd_rank + grad_output[0].data.shape[0]# NOTE: # batch_size = input[0].data.shape[0]
                        self.d_g[module] = 0 * gg[0,:actual_rank]; self.Q_g[module] = 0 * gg[:,:actual_rank] # Now we'll have Q_a's as skinnytall because
                        # we are using RSVD representation(lowrank) and thus we need to initialize our zeros accordngly
                            
                elif module in self.size_0_of_CaSL_Kfactors_G: # for non LL layers, save m_gg and pass (inversion will happen later)
                    #strictly speaking this elif could be simple "else" if all works correctly elsewhere (because the 2 sets are a partition of the total set of registered modules), but we use elif to make it explicit
                    if module in self.initalloc_modules_for_this_rank_G[self.rank]:
                        self.m_gg[module] = (1 - self.stat_decay) * gg + 0
                    else:
                        actual_rank = min(gg.shape[0], self.rsvd_rank)
                        self.d_g[module] = 0 * gg[0,:actual_rank]; self.Q_g[module] = 0 * gg[:,:actual_rank] # Now we'll have Q_a's as skinnytall because
                        # we are using RSVD representation(lowrank) and thus we need to initialize our zeros accordngly
               
            elif module in self.LL_modules_for_this_rank_G[self.rank]:
                if self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0:
                    # if the module is the responsibility of *this GPU, AND the module is on Brand-track, and it's time to Brand-update
                    ############ BRAND UPDATE ############################
                    batch_size =  grad_output[0].data.shape[0] # c_dim =batch_size
                    if self.batch_averaged:
                        G = grad_output[0].data.T * (batch_size + 0.0)**0.5
                    else:
                        G = grad_output[0].data.T / (batch_size + 0.0)**0.5
                
                    self.d_g[module], self.Q_g[module] = Brand_S_update(self.Q_g[module], self.stat_decay * self.d_g[module],
                                                                        A = self.sqr_1_minus_stat_decay * G, r_target = self.brand_r_target,
                                                                        device = torch.device('cuda:{}'.format(self.rank)) )
                    self.nkfu_dict_g[module] += 1
                    ########### END BRAND UPDATE #########################
            elif module in self.CaSL_modules_for_this_rank_A[self.rank]:
                # if the module is the responsibility of *this GPU, AND the module is NOT on Brand-track
                #### update m_gg ###############################################
                gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
                update_running_stat(gg, self.m_gg[module], self.stat_decay)
                self.nkfu_dict_g[module] += 1
            else: 
                # if the module is NOT the responsibility of *this GPU AT ALL!
                if module in self.size_0_of_LL_Kfactors_G:
                    if self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0: 
                        # NOTE: the keys to self.size_0_of_LL_Kfactors_A are all the brand-tacked linear layers, ie "LL" layers
                        # if the KFACTOR is LL and some other GPU does the brand-update of it: restart Q_a and d_a to zeros
                        self.d_g[module] = 0 * self.d_g[module]; self.Q_g[module] = 0 * self.Q_g[module]
                        self.nkfu_dict_g[module] += 1
                elif module in self.size_0_of_CaSL_Kfactors_G: #elif module not in self.size_0_of_LL_Kfactors_A:
                    self.nkfu_dict_g[module] += 1 # do nothing if the KFACTOR is not LL, as this gets reset in update_inv metod at the correct time
            #########################################################
            ##########################################################
            
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
        
        ### WORK ALLOCATION temporary (for self.steps ==0 only)... or not! depending on choice
        # IMPORTANT: USING THE TRIVIAL ALLOCATION MECHANISM ON THE 1st FACTOR COMPUTATION
        # HTIS IS BECAUSE WE CAN'T ACCESS DIMENSIONS UNLESS A PASS HAS BEEN DONE: can't access layer params if net is "black-box" (not defined by us)
        # THUS, we first do a trivial number-of layers based allocation (i.e. assuming they're all the same) 
        # and after that, IF SELECTED SO THROUGH HYPERPARAM work_alloc_propto_RSVD_cost, we'll swtich to a more efficient one, once a pass has been done
        # construct self.modules_for_this_rank (a dictonary of lists] - which tells us which modules's EVD  are computed by which GPU
        self.initalloc_modules_for_this_rank_A = self.initalloc_modules_for_this_rank_G = fct_split_list_of_modules(self.modules, self.world_size)
        # call the same fct for A and G to get the same TRIVIAL split in both A and G: that boils down to being a module-split rather than a KFACTOR split
        # returns a dictionary of lists!
        print('Split work in TRIVIAL fashion as: self.modules_for_this_rank_A = {} \n\n self.modules_for_this_rank_G = {}'.format(self.initalloc_modules_for_this_rank_A, self.initalloc_modules_for_this_rank_G))
        print('The following sentece is {} : We will also improve the allocation from the 2nd KFACTOR work onwards (at end of step 0)'.format(self.work_alloc_propto_RSVD_and_B_cost))

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        # we now can have a GPU doing only the A or only the G KFACTOR of a layer/module
         ### PARALLELIZE OVER layers: for each GPU-RANK compute only the EVD's of the "some" KFACTORS
        ##### choose which list to loo into #############################
        # this operation is required because self.CaSL_modules_for_this_rank_A // G is not defined at self.steps ==0
        if self.steps == 0:
            list_to_check_in_A = self.initalloc_modules_for_this_rank_A[self.rank]
            list_to_check_in_G = self.initalloc_modules_for_this_rank_G[self.rank]
        else:
            list_to_check_in_A = self.CaSL_modules_for_this_rank_A[self.rank]
            list_to_check_in_G = self.CaSL_modules_for_this_rank_G[self.rank]
        ##### end choose hich list to look into #############################
            
        # ================ AA^T KFACTORS ===================================
        if (m in list_to_check_in_A) and (m in self.size_0_of_CaSL_Kfactors_A):
            """Do eigen decomposition for computing inverse of the ~ fisher.
            :param m: The layer
            :return: no returns.
            """
            eps = 1e-10  # for numerical stability
            oversampled_rank = min(self.m_aa[m].shape[0], self.total_rsvd_rank)
            actual_rank = min(self.m_aa[m].shape[0], self.rsvd_rank)
            self.U_aa[m], self.D_aa[m], self.V_aa[m] = torch.svd_lowrank(self.m_aa[m], q = oversampled_rank, niter = self.rsvd_niter, M = None) # this is rsvd
            self.Q_a[m] = self.V_aa[m][:,:actual_rank] + 0.0 # 0.5*(self.U_aa[m][:,:actual_rank] + self.V_aa[m][:,:actual_rank]); 
            del self.U_aa[m]; del self.V_aa[m]
            self.d_a[m] = self.D_aa[m][:actual_rank]; # self.d_a[m][ self.d_a[m] < self.damping] = self.damping
            
            self.d_a[m].mul_((self.d_a[m] > eps).float())
            #### MAKE TENSORS CONTIGUOUS s.t. the ALLREDUCE OPERATION CAN WORK (does nto take that much!)
            self.Q_a[m] = self.Q_a[m].contiguous()
            if self.dist_comm_for_layers_debugger:
                print('RANK {} WORLDSIZE {}. computed EVD of module {} \n'.format(self.rank, self.world_size, m))
                print('The shapes are Q_a.shape = {}, d_a.shape = {}'. format(self.Q_a[m].shape, self.d_a[m].shape))
        elif m in self.size_0_of_CaSL_Kfactors_A: # the keys of this dictionary are all the CASL modules
            ### PARALLELIZE OVER layers: Set uncomputed quantities to zero to allreduce with SUM 
            #if len(self.d_a) == 0: # if it's the 1st time we encouter these guys (i.e. at init during 1st evd computation before 1st allreduction)
            self.d_a[m] = 0 * self.d_a[m];  self.Q_a[m] = 0 * self.Q_a[m]
        # ====  END  ======== AA^T KFACTORS ===================================
        
        # ================ GG^T KFACTORS ===================================
        if (m in list_to_check_in_G) and (m in self.size_0_of_CaSL_Kfactors_G):
            eps = 1e-10  # for numerical stability
            oversampled_rank = min(self.m_gg[m].shape[0], self.total_rsvd_rank)
            actual_rank = min(self.m_gg[m].shape[0], self.rsvd_rank)
            self.U_gg[m], self.D_gg[m], self.V_gg[m] = torch.svd_lowrank(self.m_gg[m], q = oversampled_rank, niter = self.rsvd_niter, M=None) # this is rsvd
            self.Q_g[m] = self.V_gg[m][:,:actual_rank] + 0.0 # 0.5 * ( self.U_gg[m][:,:actual_rank] + self.V_gg[m][:,:actual_rank]);
            del self.U_gg[m]; del self.V_gg[m]
            self.d_g[m] = self.D_gg[m][ : actual_rank ]; # d_g[m][ d_g[m] < self.damping ] = self.damping
    
            self.d_g[m].mul_((self.d_g[m] > eps).float())
            #### MAKE TENSORS CONTIGUOUS s.t. the ALLREDUCE OPERATION CAN WORK (does nto take that much!)
            self.Q_g[m] = self.Q_g[m].contiguous() # D's are already contiguous as tey were not transposed!
            if self.dist_comm_for_layers_debugger:
                print('RANK {} WORLDSIZE {}. computed EVD of module {} \n'.format(self.rank, self.world_size, m))
                print('The shapes are Q_a.shape = {}, d_a.shape = {}'. format(self.Q_g[m].shape,self.d_g[m].shape))
        elif m in self.size_0_of_CaSL_Kfactors_G:
            ### PARALLELIZE OVER layers: Set uncomputed quantities to zero to allreduce with SUM 
            #if len(self.d_a) == 0: # if it's the 1st time we encouter these guys (i.e. at init during 1st evd computation before 1st allreduction)
            self.d_g[m] = 0 * self.d_g[m];  self.Q_g[m] = 0 * self.Q_g[m]
        # ====== END ======= GG^T KFACTORS ===================================
        
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
        ###### get nkfu #################
        nkfu_a = self.nkfu_dict_a[m]
        nkfu_g = self.nkfu_dict_g[m]
        ###### END: get nkfu ############
        v1 = X_reg_inverse_M_adaptive_damping(U = self.Q_g[m], D = self.d_g[m], M = p_grad_mat, lambdda = damping, 
                                               n_kfactor_update = nkfu_g, rho = self.stat_decay, damping_type = self.damping_type) # the damping here is adaptive!
        v = M_X_reg_inverse_adaptive_damping(U = self.Q_a[m], D = self.d_a[m], M = v1, lambdda = damping,
                                              n_kfactor_update = nkfu_a, rho = self.stat_decay, damping_type = self.damping_type)  # the damping here is adaptive!
        
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
        if self.dist_comm_for_layers_debugger:
            print('rank = {}, at step = {}, after the sav_inpt and grad hooks\n'.format(self.rank, self.steps))

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
                #if isinstance(m, nn.Linear): # if the layer at hand is linear, 
                #pass # we never do an RSVD (R-update), only ever do B_updates for LARGE enough linear layers!
                # for smaller linear layers we do RSVD and never brand (as they go on the brand track) - but it is simpler to
                # perform this RSVD operation at the B-update phase, as there we have per- AA vs GG individual control while here the control is bundled AA and GG so harder to segment on ifs over this
                #else: # if it's not a linear layer, do RSVD every TInv iterations (No brand update for Conv Layers)
                self._update_inv(m)
                    
                
        
        # take the step and allreduce across evd's if the inverses were updated    
        for m in self.modules:
            classname = m.__class__.__name__
            if ((m not in self.size_0_of_LL_Kfactors_A) and (self.steps % self.TInv == 0)) or ((m in self.size_0_of_LL_Kfactors_A) and (self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0)):
                # if the inversion was done locally this turn, allreduce to disseminate inverse representation
                #if it's time to recompute inverse for Conv layers or for liear (BRAND) layers
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. Before Allreduce d_a={}, size_d_a = {}, Q_a = {}, size_Q_a = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_a[m], self.d_a[m].shape, self.Q_a[m], self.Q_a[m].shape))
                #print('RANK {}. Doing line: dist.all_reduce(self.d_a[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                handle = dist.all_reduce(self.d_a[m], dist.ReduceOp.SUM, async_op = True)
                handle.wait()
                #self.d_a[m] = 0 * self.d_a[m] + 1

                #print('RANK {}. Doing line : dist.all_reduce(self.Q_a[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                handle = dist.all_reduce(self.Q_a[m], dist.ReduceOp.SUM, async_op = True)
                handle.wait()
                #self.Q_a[m] = 0 * self.Q_a[m]; Q_debug_size = min(self.Q_a[m].shape[0],self.Q_a[m].shape[1])
                #self.Q_a[m][torch.arange(Q_debug_size),torch.arange(Q_debug_size)] = 1 # make Q_g identity to avoid comunication
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. AFTER Allreduce d_a={}, size_d_a = {}, Q_a = {}, size_Q_a = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_a[m], self.d_a[m].shape, self.Q_a[m], self.Q_a[m].shape))
            
            if ((m not in self.size_0_of_LL_Kfactors_G) and (self.steps % self.TInv == 0)) or ((m in self.size_0_of_LL_Kfactors_G) and (self.steps % (self.TCov * self.brand_update_multiplier_to_TCov) == 0)):
                #print('RANK {}. Doing line : dist.all_reduce(self.d_g[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. Before Allreduce d_g={}, size_d_g = {}, Q_g = {}, size_Q_g = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_g[m], self.d_g[m].shape, self.Q_g[m], self.Q_g[m].shape))
                handle = dist.all_reduce(self.d_g[m], dist.ReduceOp.SUM, async_op = True)
                handle.wait()
                #self.d_g[m] = 0 * self.d_g[m] + 1

                #print('RANK {}. DOING LINE: dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = False)'.format(self.rank))
                handle = dist.all_reduce(self.Q_g[m], dist.ReduceOp.SUM, async_op = True)
                handle.wait()
                #self.Q_g[m] = 0 * self.Q_g[m]; Q_debug_size = min(self.Q_g[m].shape[0],self.Q_g[m].shape[1])
                #self.Q_g[m][torch.arange(Q_debug_size),torch.arange(Q_debug_size)] = 1 # make Q_g identity to avoid comunication
                if self.dist_comm_for_layers_debugger:
                    print('RANK {}. STEP {}. WORLDSIZE {}. MODULE {}. AFTER Allreduce d_g={}, size_d_g = {}, Q_g = {}, size_Q_g = {} \n'.format(self.rank, self.steps, self.world_size, m, self.d_g[m], self.d_g[m].shape, self.Q_g[m], self.Q_g[m].shape))
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)
        
        #### 1. Change work allocation or 2. rearrange variables maintaining the same wokr alloation (2 is there to simply integrate the case of choosing trivial alloc vs efficient allocation)
        #### change work allocation to dimension-based for RSVD
        if self.steps == 0 and self.work_alloc_propto_RSVD_and_B_cost == True:
            raise NotImplementedError('self.work_alloc_propto_RSVD_and_B_cost == True case not implemented yet! Coming in next commit or so!')
        elif self.steps == 0 and self.work_alloc_propto_RSVD_and_B_cost == False:
            ## if we don't want to reallocate in an efficient way, we still need to split each allocated list we have into 1 for LL and 1 for CaSL
            ### 1. construct self.LL_modules_for_this_rank_A and self.CaSL_modules_for_this_rank_A dictionaries
            for key_rank in self.initalloc_modules_for_this_rank_A:
                self.LL_modules_for_this_rank_A[key_rank] = []
                self.CaSL_modules_for_this_rank_A[key_rank] = []
                for module_allocated in self.initalloc_modules_for_this_rank_A[key_rank]:
                    if module_allocated in self.size_0_of_LL_Kfactors_A: # the keys of this dict are all the LL layers
                        #if it's LL, allocate to the same rank, but put in the correct dictionary
                        self.LL_modules_for_this_rank_A[key_rank].append(module_allocated)
                    elif module_allocated in self.size_0_of_CaSL_Kfactors_A: # the keys of this dict are all the CaSL layers
                        #if it's CaSL, allocate to the same rank, but put in the correct dictionary
                        self.CaSL_modules_for_this_rank_A[key_rank].append(module_allocated)
                        
            ### 2. construct self.LL_modules_for_this_rank_G and self.CaSL_modules_for_this_rank_G dictionaries
            for key_rank in self.initalloc_modules_for_this_rank_G:
                self.LL_modules_for_this_rank_G[key_rank] = []
                self.CaSL_modules_for_this_rank_G[key_rank] = []
                for module_allocated in self.initalloc_modules_for_this_rank_G[key_rank]:
                    if module_allocated in self.size_0_of_LL_Kfactors_G: # the keys of this dict are all the LL layers
                        #if it's LL, allocate to the same rank, but put in the correct dictionary
                        self.LL_modules_for_this_rank_G[key_rank].append(module_allocated)
                    elif module_allocated in self.size_0_of_CaSL_Kfactors_G: # the keys of this dict are all the CaSL layers
                        #if it's CaSL, allocate to the same rank, but put in the correct dictionary
                        self.CaSL_modules_for_this_rank_G[key_rank].append(module_allocated)
        
        self._step(closure)
        self.steps += 1




