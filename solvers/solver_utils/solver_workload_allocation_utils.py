import torch

# wrappers ###############
## NOTE: for TESNOR ONLY use: allocate_work_timebased_tensors
#for all 4 fct below  already_alloc_time_list is an optional argument a list wiht len = num workers where each element is how much load each worker starts with for this allocation
 # if left blank it is assumed each worker starts with 0 load. Useful for B-R-KFAC allocations in particular
def allocate_B_inversion_work_same_fixed_r_and_batchsize(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_RSVD, batch_size, already_alloc_time_list = None):
    return allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, 
                                                                  size_0_of_all_Kfactors_G, 
                                                                  size_0_of_all_Kfactors_A, 
                                                                  target_rank_ = target_rank_RSVD,
                                                                  oversampling_to_rank_ = 20, 
                                                                  batch_size_ = batch_size, 
                                                                  type_of_cost = 'B',
                                                                  already_alloc_time_list = already_alloc_time_list) 

def allocate_RSVD_inversion_work_same_fixed_r(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_RSVD, oversampling_to_rank = 20, already_alloc_time_list = None):
    return allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, 
                                                                  size_0_of_all_Kfactors_G, 
                                                                  size_0_of_all_Kfactors_A, 
                                                                  target_rank_ = target_rank_RSVD, 
                                                                  oversampling_to_rank_ = oversampling_to_rank, # not required 
                                                                  batch_size_ = None, #not required
                                                                  type_of_cost = 'RSVD',
                                                                  already_alloc_time_list = already_alloc_time_list)

def allocate_EVD_inversion_work(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, already_alloc_time_list = None):
    return allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, 
                                                                  size_0_of_all_Kfactors_G, 
                                                                  size_0_of_all_Kfactors_A, 
                                                                  target_rank_ = None, #not required
                                                                  oversampling_to_rank_ = None, #not required
                                                                  batch_size_ = None, #not required
                                                                  type_of_cost = 'EVD', 
                                                                  already_alloc_time_list = already_alloc_time_list)

def allocate_ANYTHING_in_prop_to_MEASURED_time(number_of_workers, measured_invtime_of_all_Kfactors_G, measured_invtime_of_all_Kfactors_A, already_alloc_time_list = None):
    return allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, 
                                                                  size_0_of_all_Kfactors_G = measured_invtime_of_all_Kfactors_G, 
                                                                  size_0_of_all_Kfactors_A = measured_invtime_of_all_Kfactors_A, 
                                                                  target_rank_ = None, #not required
                                                                  batch_size_ = None, #not required
                                                                  type_of_cost = 'time_given_instead_of_size',
                                                                  already_alloc_time_list = already_alloc_time_list)

# end wrappers ###############

###########################################################################################################################
####### Helper functions for: predicting RSVD time based on size with measurements nd more sophssticated computation ######
###########################################################################################################################
theta_results_dict = {30: [0.00255482, 0.01016938, 0.05377455],
                      80: [0.00499274, 0.01908004, 0.14648095],
                    130: [0.00711607, 0.02932301, 0.26814728],
                    180: [0.00887165, 0.04546348, 0.30530023],
                    240: [0.02178314, 0.03073283, 0.39171789],
                    370: [0.01602652, 0.12918453, 0.54051409],
                    520: [0.02564055, 0.10353079, 0.98908084]
                    } # these numbers were obtained by linear regressions on features: 1, size, size^2 - with data measured on V100-SXM2 GPU
                    # different GPUs give different numbers and it is recommended the user does his own measurements and linear reg and replaces the numbers here for their own GPU
# the dict keys are TOTAL rank (used oversampling of 20 in test but it doesn t really matter!)

# instead of using the dictionary with interpolation we linearly-regressed again keys = target rank vs theta[i], for all i \in {0,1,2} and use that
omega_0 = [2.20764030e-03, 4.61496825e-05] # recoveringtheta_0_params
omega_1 = [0.00099798, 0.00023258] # recoveringtheta_1_params
omega_2 = [-0.0058709, 0.00176523] # recoveringtheta_2_params

def pred_theta_0_from_omega(total_rank):
    return omega_0[0] + total_rank *omega_0[1]
def pred_theta_1_from_omega(total_rank):
    return omega_1[0] + total_rank * omega_1[1]
def pred_theta_2_from_omega(total_rank):
    return omega_2[0] + total_rank *omega_2[1]
# then, after we recover theta based on target rank we get the cpu time based on size and theta.
############## end RSVD helper learnt (from measurements) quantities - for time prediciton #################################

def predict_RSVD_comptime_from_size_and_targrank(size, total_rank):
    #measurements on Tesla V100-SXM2
    if total_rank < 30: # make sure we fall within the 30,520 interval
        total_rank = 30
    elif total_rank > 520:
        total_rank = 520
    
    # first we predict the lin-reg parameters of (size, time) which are diffefrent for each target rank
    thetta_0 = pred_theta_0_from_omega(total_rank)
    thetta_1 = pred_theta_1_from_omega(total_rank)
    thetta_2 = pred_theta_2_from_omega(total_rank)
    
    #### then given these parameters we predict the cpu time based on size and return
    def predict_time_from_pred_theta_and_size(sizze):
        sizze = sizze / 33000 # the 33000 factor was used at learning so also using at prediciton
        return thetta_0 + thetta_1 * sizze + thetta_2 *sizze **2
    
    return predict_time_from_pred_theta_and_size(size)
##########################################################################################################################
####### END: Helper functions for: predicting time based on size with measurements nd more sophssticated computation #####
##########################################################################################################################


#############################################################################################################################
####### Helper functions for: predicting B-updt time based on size with measurements nd more sophssticated computation ######
#############################################################################################################################
def predict_B_comptime_from_size_and_targrank(size, starting_rank, incoming_rank):
    #measurements on Tesla V100-SXM2
    # only did the regression for starting rank 220 (this is also the target rank) and 256 incoming rank (also the batchsize)
    # so it's hardcoded for now as in the next 2 lines, will change in the future
    if size < starting_rank + incoming_rank: # then return the RSVD time prediciton: because that's what's it doing!
        return predict_RSVD_comptime_from_size_and_targrank(size, total_rank = starting_rank + incoming_rank)
    
    # (else):
    theta_0 = 0.02012786; theta_1 =  0.01055792 # (regression based on measurements with div_data_factor = 33000, 220 starting rank and 256 incoming rank)
    return theta_0 + theta_1 * size / 33000
#################################################################################################################################
####### END Helper functions for: predicting B-updt time based on size with measurements nd more sophssticated computation ######
#################################################################################################################################



#################################################################################################################################
####### Helper functions for: predicting EVD time based on size with measurements nd more sophssticated computation #############
#################################################################################################################################
def predict_EVD_comptime_from_size_and_targrank(size):
    # measurements on Tesla V100-SXM2
    
    """
    max_size_measured_threshold = 2e04
    if size > max_size_measured_threshold:
        print('Warning, we are extrapolating. Only measured until max_size == {} but size to predict for was {}. The extrapolation should be pretty good though.'.format(max_size_measured_threshold, size))
    """
    theta_1 =  1.23393439; theta_3 =  58.67534088 # other thetas theta_0 and theta_2 were practially 0 in our measurements+regression
    size = size/ 33000 # same factor as with all
    return theta_1 * size + theta_3 * size**3
#################################################################################################################################
####### END Helper functions for: predicting EVD time based on size with measurements nd more sophssticated computation #########
#################################################################################################################################    
    
def allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_, 
                                                           oversampling_to_rank_, batch_size_, type_of_cost, already_alloc_time_list = None):
    #### input:
    #number_of_workers = number of total workers we have ie worldsize
    # size_0_of_all_Kfactors_G - a dictionary where the key is the module, and the value is the size[0] of GG^T K-factor for that module
    # size_0_of_all_Kfactors_A - a dictionary where the key is the module, and the value is the size[0] of GG^T K-factor for that module
    # target_rank_RSVD = the FIXED IN ITERATIONA AND ACROSS KFACTORS RSVD target rank
    #### output:
    # dict_of_lists_of_responsibilities_A = a dictionary where the key is the wwrker number 
    # and the value is the list of all modules that particular worker is responsible for at Kfactor AA^T
    # dict_of_lists_of_responsibilities_G = a dictionary where the key is the wwrker number 
    # and the value is the list of all modules that particular worker is responsible for at Kfactor GG^T
    
    # allocation is done to be efficient if all the target ranks are the same across the Kfactors
    dict_of_lists_of_responsibilities_A = {}; dict_of_lists_of_responsibilities_G = {}
    
    #initialize lists as desired
    for i in range(0, number_of_workers):
        dict_of_lists_of_responsibilities_A[i] = []; dict_of_lists_of_responsibilities_G[i] = []
    
    # construct adjoint lists
    computation_time_for_A = []; computation_time_for_G = []
    modules_as_keys_list_A = []; modules_as_keys_list_G = []
    for key in size_0_of_all_Kfactors_A:
        modules_as_keys_list_A.append(key)
        # the cost is m^2 r if m > r, and m^3 otherwise. 
        #To avoid large numbers, we divide by r, so m^2 and m^3/r are comapred
        if type_of_cost == 'EVD':
            computation_time_for_A.append(size_0_of_all_Kfactors_A[key]**3)
        elif type_of_cost == 'RSVD': # RSVD is theoretically O(m^2 n ) but on GPUs it seems to scale more like O(mn)
            total_rank_ = target_rank_ + oversampling_to_rank_
            computation_time_for_A.append(predict_RSVD_comptime_from_size_and_targrank(size_0_of_all_Kfactors_A[key], total_rank_))#**2)
            # old and inaccurate below
            #computation_time_for_A.append( min(1, size_0_of_all_Kfactors_A[key]/target_rank_) * size_0_of_all_Kfactors_A[key])#**2)
        elif type_of_cost == 'B':
            computation_time_for_A.append(predict_B_comptime_from_size_and_targrank(size = size_0_of_all_Kfactors_A[key], starting_rank = 220, incoming_rank = 256))
            #if size_0_of_all_Kfactors_A[key] / (target_rank_ + batch_size_) > 1: # if we perform a true B-update fr this layer
            #    computation_time_for_A.append( size_0_of_all_Kfactors_A[key]) #  (target_rank_ + batch_size_) * size_0_of_all_Kfactors_G[key]
            #else: # else, we perform an rsvd
            #    # here we're assuming the rsvd target rank and the brand target rank are the same
            #    computation_time_for_A.append( min(target_rank_, size_0_of_all_Kfactors_G[key] ) * size_0_of_all_Kfactors_G[key] / (target_rank_ + batch_size_) ) # (target_rank_, size_0_of_all_Kfactors_G[key] ) * size_0_of_all_Kfactors_G[key]**2
            #    # the reasn we don't use  size_0_of_all_Kfactors_G[key]**2 in the line just above is because we assume (target_rank_ + batch_size_)  are very low 
            #    #(which is always the case tbh!) AND in that situation the cost scales linearly rather than quadratically on GPUs 
            #    #(a quadratic scaling is kept down to linear for small sizes on GPU due to efficient use of many cores, 
            #    #- but eventually the quadratic scaling kicks in: we observe this happens at around 400-800 size for RSVD - so we can use 256 + 220 as being in the linear regime with minimal problems)
        
        elif type_of_cost == 'time_given_instead_of_size':
            computation_time_for_A.append(size_0_of_all_Kfactors_A[key])
        
    for key in size_0_of_all_Kfactors_G:
        modules_as_keys_list_G.append(key)
        # the cost is m^2 r if m > r, and m^3 otherwise. 
        #To avoid large numbers, we divide by r, so m^2 and m^3/r are comapred
        if type_of_cost == 'EVD':
            computation_time_for_G.append( predict_EVD_comptime_from_size_and_targrank(size_0_of_all_Kfactors_G[key]**3) )
        elif type_of_cost == 'RSVD': # RSVD is theoretically O(m^2 n ) but on GPUs it seems to scale more like O(mn)
            total_rank_ = target_rank_ + oversampling_to_rank_
            computation_time_for_G.append(predict_RSVD_comptime_from_size_and_targrank(size_0_of_all_Kfactors_G[key], total_rank_))#**2)
            # old and inaccurate below
            #computation_time_for_G.append(min(1, size_0_of_all_Kfactors_G[key] / target_rank_) * size_0_of_all_Kfactors_G[key])#**2)
        elif type_of_cost == 'B':
            computation_time_for_G.append(predict_B_comptime_from_size_and_targrank(size = size_0_of_all_Kfactors_G[key], starting_rank = 220, incoming_rank = 256))
            # in this case target_rank_ is actually target_rank_B + N_BS but leaving same variable name for simplicity
            # diving cost by (target_rank_ + batch_size_) to make numbers more manageable (arguably not needed)
            #if size_0_of_all_Kfactors_G[key] / (target_rank_ + batch_size_) > 1: # if we perform a true B-update fr this layer
            #    computation_time_for_G.append( size_0_of_all_Kfactors_G[key]) #  (target_rank_ + batch_size_) * size_0_of_all_Kfactors_G[key]
            #else: # else, we perform an rsvd
            #    # here we're assuming the rsvd target rank and the brand target rank are the same
            #    computation_time_for_G.append( min(target_rank_, size_0_of_all_Kfactors_G[key] ) * size_0_of_all_Kfactors_G[key]**2 / (target_rank_ + batch_size_) ) # (target_rank_, size_0_of_all_Kfactors_G[key] ) * size_0_of_all_Kfactors_G[key]**2
        
        elif type_of_cost == 'time_given_instead_of_size':
            computation_time_for_G.append(size_0_of_all_Kfactors_G[key])
        
    ### compute raw allocations
    optimal_allocation_a, optimal_allocation_g, sum_loads_each_worker = optimal_most_allocation(number_of_workers, computation_time_for_A, computation_time_for_G, already_alloc_time_list = already_alloc_time_list)
    
    #print(modules_as_keys_list_A); print(modules_as_keys_list_G); print(optimal_allocation_a); print(optimal_allocation_g)
    #print(computation_time_for_A); print(computation_time_for_G)
    ### construct allocatons in desired form
    for module_idx_a, worker_number_a in enumerate(optimal_allocation_a):
        dict_of_lists_of_responsibilities_A[worker_number_a].append(modules_as_keys_list_A[module_idx_a])
        #print('A: Allocated module name = {}, and with index = {} to worker # {}'.format(modules_as_keys_list_A[module_idx_a],module_idx_a ,worker_number_a))
        
    for module_idx_g, worker_number_g in enumerate(optimal_allocation_g):
        dict_of_lists_of_responsibilities_G[worker_number_g].append(modules_as_keys_list_G[module_idx_g])
        #print('G: Allocated module name = {}, and with index = {} to worker # {}'.format(modules_as_keys_list_G[module_idx_g],module_idx_g ,worker_number_g))
    
    return dict_of_lists_of_responsibilities_A, dict_of_lists_of_responsibilities_G, sum_loads_each_worker

def allocate_RSVD_inversion_work_same_fixed_r_tensor(number_of_workers, size_0_of_all_Kfactors_A_tensor, 
                                                     size_0_of_all_Kfactors_G_tensor, target_rank_RSVD, modules_list):
    return allocate_sizebased_tensor_raw(number_of_workers = number_of_workers,
                                         size_0_of_all_Kfactors_A_tensor = size_0_of_all_Kfactors_A_tensor, 
                                         size_0_of_all_Kfactors_G_tensor = size_0_of_all_Kfactors_G_tensor, 
                                         target_rank_RSVD = target_rank_RSVD, modules_list = modules_list,
                                         cost_type = 'RSVDgpu')

def allocate_sizebased_tensor_raw(number_of_workers, size_0_of_all_Kfactors_A_tensor, 
                                  size_0_of_all_Kfactors_G_tensor, target_rank_RSVD, modules_list,
                                  cost_type):
    #### get TIME(cost) estimated based on size
    if cost_type == 'RSVDgpu':
        tensor_computation_time_for_A = torch.min(size_0_of_all_Kfactors_A_tensor, size_0_of_all_Kfactors_A_tensor**2/target_rank_RSVD)
        tensor_computation_time_for_G = torch.min(size_0_of_all_Kfactors_G_tensor, size_0_of_all_Kfactors_G_tensor**2/target_rank_RSVD)
    elif cost_type == 'RSVDcpu':
        tensor_computation_time_for_A = torch.min(size_0_of_all_Kfactors_A_tensor**2, size_0_of_all_Kfactors_A_tensor**3/target_rank_RSVD)
        tensor_computation_time_for_G = torch.min(size_0_of_all_Kfactors_G_tensor**2, size_0_of_all_Kfactors_G_tensor**3/target_rank_RSVD)
    elif cost_type == 'Bgpu':
        raise NotImplementedError('for function allocate_sizebased_tensor_raw cost type = {} NOT IMPLEMENTED YET'.format(cost_type))
        #tensor_computation_time_for_A = size_0_of_all_Kfactors_A_tensor
        #tensor_computation_time_for_G = size_0_of_all_Kfactors_G_tensor
    else:
        raise NotImplementedError('for function allocate_sizebased_tensor_raw cost type = {} NOT IMPLEMENTED'.format(cost_type))
    
    ### call time_based function  (time is now ESTIMATED not measured)
    return allocate_work_timebased_tensors(number_of_workers = number_of_workers, 
                                           tensor_computation_time_for_A = tensor_computation_time_for_A,
                                           tensor_computation_time_for_G = tensor_computation_time_for_G,
                                           modules_list = modules_list)


def allocate_work_timebased_tensors(number_of_workers, tensor_computation_time_for_A, tensor_computation_time_for_G, modules_list):
    # number_of_workers - self explanatory
    # tensor_computation_time_for_A - tensor format, the 1st element is the time for Kfactor A of the 1st module in modules_list
    # tensor_computation_time_for_G - tensor format, the 1st element is the time for Kfactor G of the 1st module in modules_list
    # modules_list - self explanatory
    number_of_modules = tensor_computation_time_for_A.shape[0]
    tensor_times_concat = torch.concat([tensor_computation_time_for_A, tensor_computation_time_for_G])
    allocation_tensor_for_ranks = 0 * tensor_times_concat # this tensor will hold 
    argsort = torch.argsort( - tensor_times_concat) # sort descendingly
    
    if 2 * number_of_modules <= number_of_workers: ### solve particular case sepparately: 1. TRIVIALish: more workers than KFACTORS
        #note that the number of Kfactors is 2x the number of modules
        allocation_tensor_for_ranks = torch.arange(0, 2 * number_of_modules)
    else: # 2. solve the case when there's fewer workers than KFACTORS
        #### initialize per-module sums to zero
        tensor_sumtimes_for_each_module = 0 * allocation_tensor_for_ranks[0 : number_of_workers]
        #### allocate in tensor (and concatenated) format
        for argsort_idx in range(0, argsort.shape[0]):
            idx = argsort[argsort_idx]
            time_value = tensor_times_concat[idx]
            #### find out which worker to allocate this work to
            worker_to_allocate_to = torch.argmin(tensor_sumtimes_for_each_module)
            #### allocate work (save)
            allocation_tensor_for_ranks[ idx ] = worker_to_allocate_to
            #### ammend sum for future allocations
            tensor_sumtimes_for_each_module[ worker_to_allocate_to ] += time_value
    
    #### translate tensor-format (and concantenated) allocation in list of module allocation
    #convert relevant quantities to numpy to avoid wrong keys (because they sit on certain GPUs)
    allocation_tensor_for_ranks = allocation_tensor_for_ranks.cpu().numpy()
    #### initialize dictionaries
    dict_of_lists_of_responsibilities_A = {}; dict_of_lists_of_responsibilities_G = {}
    for rank in range(0, number_of_workers): # for rank in torch.range(0, number_of_workers):
        dict_of_lists_of_responsibilities_A[rank] = []; dict_of_lists_of_responsibilities_G[rank] = []
    # append
    for i, module in enumerate(modules_list):
        #### extract which rank is allocated to this module
        rank_alloc_to_m_and_A = allocation_tensor_for_ranks[i]
        rank_alloc_to_m_and_G = allocation_tensor_for_ranks[i + number_of_modules]
        ### append to the correct dicionaries
        dict_of_lists_of_responsibilities_A[rank_alloc_to_m_and_A].append(module)
        dict_of_lists_of_responsibilities_G[rank_alloc_to_m_and_G].append(module)
    ##### Return
    ### note that the gpu_rank (keys) to dictionaries are TENSORS rather than typical int: we need to ammend remainig code to deal with it
    return dict_of_lists_of_responsibilities_A, dict_of_lists_of_responsibilities_G #, tensor_sumtimes_for_each_module#, allocation_tensor_for_ranks#, tensor_sumtimes_for_each_module


def optimal_most_allocation(number_of_workers, computation_time_for_A, computation_time_for_G, already_alloc_time_list = None):
    # returns the most optimal possible allocation given the desired num workes and the comp times for A and G modules
    # return type is a tupe (list_a, list_g) where for each module-KFCTOR-type pair we say which worker will be allocated ot it
    # the module is inferred by the position in the list, and will be recovered using modules_as_keys_list called with the same idx as the idx of our output
    
    # the RELEVANT optimal defn here is minimize the difference between minimal and maximal total load per worker
    len_A = len(computation_time_for_A) ; len_G = len(computation_time_for_G); total_len = len_A + len_G
    list_alloc_module_a = [];  list_alloc_module_g = []
    
    ### Initialize sum of current work-load of all workers  ###############
    if already_alloc_time_list is not None and len(already_alloc_time_list) == number_of_workers:
        sum_loads_each_worker = already_alloc_time_list
    else:
        sum_loads_each_worker = [0] * number_of_workers
    ### END: Initialize sum of current work-load of all workers  ###############
    
    if number_of_workers >= total_len: # special case 1
        # do trivial allocation with 1 each
        for idx in range(0, len_A):
            list_alloc_module_a.append(idx)
            sum_loads_each_worker[idx] += computation_time_for_A[idx]
        for idx in range(0, len_G):
            list_alloc_module_g.append(idx + len_A)
            sum_loads_each_worker[idx] += computation_time_for_G[idx]
        """elif number_of_workers == total_len - 1: #special case 2: NOT STRICTLY REQUIRED AS IT IS INCLUDED IN THE NEXT CASE
        lowest_idx = None; lowest_num = 1e16; second_lowest_idx = None; second_lowest_num = 1e16
        for idx, num in enumerate(computation_time_for_A + computation_time_for_G):
            if num < max(lowest_num, second_lowest_num):
                if lowest_idx == None:
                    lowest_idx = idx; lowest_num = num
                elif second_lowest_idx == None:
                    second_lowest_idx = idx; second_lowest_num = num
                    if lowest_num > second_lowest_num: # make sure the lowest is indeed the lowest
                        ccc = lowest_idx; lowest_idx = second_lowest_idx; second_lowest_idx = ccc
                        ccc = lowest_num; lowest_num = second_lowest_num; second_lowest_num = ccc
                elif num < lowest_num:
                    second_lowest_idx = lowest_idx; second_lowest_num = lowest_num
                    lowest_idx = idx; lowest_num = num
                else:
                    second_lowest_idx = idx; second_lowest_num = num
        #then cluster the 2 cheapest together for the same guy, and everyone else gets 1
        list_alloc_module_a = [-1] * len_A # -1 means not allocate yet
        list_alloc_module_g = [-1] * len_G # -1 means not allocate yet
        #### process lowest index
        if lowest_idx < len_A:
            list_alloc_module_a[lowest_idx] = number_of_workers - 1
        else:
            list_alloc_module_g[lowest_idx - len_A] = number_of_workers - 1
        #### process second lowest index
        if second_lowest_idx < len_A:
            list_alloc_module_a[second_lowest_idx] = number_of_workers - 1
        else:
            list_alloc_module_g[second_lowest_idx - len_A] = number_of_workers - 1
        ## now fill the others
        current_worker_to_allocate = 0
        for el_a_idx, el_a in enumerate(list_alloc_module_a):
            if el_a == -1:
                list_alloc_module_a[el_a_idx] = current_worker_to_allocate
                current_worker_to_allocate += 1
        for el_g_idx, el_g in enumerate(list_alloc_module_g):
            if el_g == -1:
                list_alloc_module_g[el_g_idx] = current_worker_to_allocate
                current_worker_to_allocate += 1    """
    else: # in the general case, we should really try all cases which has a # of (num_KFACT choose num_worers) * (num_KFAC - num_workers)^num_workers
        # then choose the one where max_sum_load - min_sum_load is minimal
        # hwever the number can get as large as 10^40 for 50 layers and 20 workers
        # instead, we use a greedy approach which is reasonably close to optimal
        ##### 1. sort each piece of effort into descending order in a bigger list #########
        cpu_time_a_then_g_list = computation_time_for_A + computation_time_for_G
        def argsort_descending(seq):
            argsort_ = sorted(range(len(seq)), key=seq.__getitem__)
            argsort_.reverse()
            return argsort_
        idxsort_dsc = argsort_descending(cpu_time_a_then_g_list)
        ##### END 1. sort each piece of effort into descending order in a bigger list #####
        
        ##### 2. Initialize lists to be output ###############
        list_alloc_module_a = [-1] * len_A # -1 means not allocate yet
        list_alloc_module_g = [-1] * len_G # -1 means not allocate yet
        ##### END 2. Initialize sum of current work-load of all workers and lists to be output ###########
        
        ##### 3. Loop over idxsort_dsc and allocate in a greedy fashion ##################################
        for idxx in idxsort_dsc:
            ########### check which worker has smallest load
            worker_smallest_load = argsort_descending(sum_loads_each_worker)[-1]
            
            ########### allocate that worker and update sum
            if idxx < len_A: # we're on list A
                list_alloc_module_a[idxx] = worker_smallest_load
            else: # we're on list G
                list_alloc_module_g[idxx - len_A] = worker_smallest_load
            sum_loads_each_worker[worker_smallest_load] += cpu_time_a_then_g_list[idxx]
        
        ## debugger return
        #return list_alloc_module_a, list_alloc_module_g, sum_loads_each_worker, max(sum_loads_each_worker) - min(sum_loads_each_worker)
        ##### END : 3. Loop over idxsort_dsc and allocate in a greedy fashion ############################
    # sum_loads_each_worker is returned to know how many load-units each worker has been allocated - this can be used later to start allocation from here, for eg like in B-R-KFAC
    return list_alloc_module_a, list_alloc_module_g, sum_loads_each_worker

if __name__ == '__main__': ### testing
    """
    computation_time_for_A = [170.00000000000003, 350.00000000000006 ,70, 150, 90]
    computation_time_for_G = [190, 300.00000000000006, 200.00000000000003, 50.00000000000001, 100]
    ### defined numbers above to have the same outcome as squareing the sqrt (which loses an eps-precision and might change selection)
    size_0_of_all_Kfactors_A = {'C1': 170**(0.5), 'C2': 350**(0.5) ,'C3' : 70**(0.5), 'C4': 150**(0.5), 'L1': 90**(0.5)}
    size_0_of_all_Kfactors_G = {'C1':  190**(0.5), 'C2': 300**(0.5) ,'C3' : 200**(0.5), 'C4': 50**(0.5), 'L1': 100**(0.5)}
    
    # realisitc values from my VGG16_bn with less ooling
    """
    computation_time_for_A = [2305**2, 4609**2 ,28**2, 577**2, 1153**2, 4609**2, 16385**2, 2049**2]
    computation_time_for_G = [512**2, 512**2, 64**2, 64**2, 128**2, 512**2, 2048**2, 10**2]
    size_0_of_all_Kfactors_A = {'C1': 2305, 'C2': 4609 ,'C3' : 28, 'C4': 577, 'C5': 1153, 'C6': 4609, 'L1': 16385, 'L2': 2049}
    size_0_of_all_Kfactors_G = {'C1': 512, 'C2': 512, 'C3': 64, 'C4': 64, 'C5': 128, 'C6': 512, 'L1': 2048, 'L2': 10}
    #"""

    print('TESTING optimal_most_allocation. \n input lists')
    print(computation_time_for_A); print(computation_time_for_G)
    print('Output')
    print(optimal_most_allocation(2, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(2, computation_time_for_A, computation_time_for_G, [268468225, 0]))
    268468225
    print(optimal_most_allocation(3, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(4, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(5, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(6, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(7, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(8, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(9, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(9, computation_time_for_A, computation_time_for_G, [268468225] * 8 + [0] ))
    print(optimal_most_allocation(9, computation_time_for_A, computation_time_for_G, [268468225] + [0] * 8))
    print(optimal_most_allocation(10, computation_time_for_A, computation_time_for_G))
    
    print('TESTING outer function')
    number_of_workers = 2
    
    target_rank_RSVD = 20 #220 # set to very low to avoid the x min(1,m/r) effect!
    print('Output:')
    print(allocate_RSVD_inversion_work_same_fixed_r(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_RSVD))

def invert_rank_to_modules_dict(rank_to_modules_dict):
    D = {}
    for key_rank in rank_to_modules_dict.keys():
        for module in rank_to_modules_dict[key_rank]:
            D[module] = key_rank
    return D