# wrappers ###############
def allocate_B_inversion_work_same_fixed_r_and_batchsize(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_RSVD, batch_size):
    return allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, 
                                                                  size_0_of_all_Kfactors_G, 
                                                                  size_0_of_all_Kfactors_A, 
                                                                  target_rank_ = target_rank_RSVD,
                                                                  batch_size_ = batch_size, 
                                                                  type_of_cost = 'RSVD') 

def allocate_RSVD_inversion_work_same_fixed_r(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_RSVD):
    return allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, 
                                                                  size_0_of_all_Kfactors_G, 
                                                                  size_0_of_all_Kfactors_A, 
                                                                  target_rank_ = target_rank_RSVD, 
                                                                  batch_size_ = None, #not required
                                                                  type_of_cost = 'RSVD')

def allocate_EVD_inversion_work(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A):
    return allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, 
                                                                  size_0_of_all_Kfactors_G, 
                                                                  size_0_of_all_Kfactors_A, 
                                                                  target_rank_ = None, #not required
                                                                  batch_size_ = None, #not required
                                                                  type_of_cost = 'EVD')

# end wrappers ###############

def allocate_inversion_work_same_fixed_sizes_any_cost_type(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_, batch_size_, type_of_cost):
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
        elif type_of_cost == 'RSVD':
            computation_time_for_A.append( min(1, size_0_of_all_Kfactors_A[key]/target_rank_) * size_0_of_all_Kfactors_A[key]**2)
        elif type_of_cost == 'B':
            pass
    for key in size_0_of_all_Kfactors_G:
        modules_as_keys_list_G.append(key)
        # the cost is m^2 r if m > r, and m^3 otherwise. 
        #To avoid large numbers, we divide by r, so m^2 and m^3/r are comapred
        if type_of_cost == 'EVD':
            computation_time_for_G.append(size_0_of_all_Kfactors_G[key]**3)
        elif type_of_cost == 'RSVD':
            computation_time_for_G.append(min(1, size_0_of_all_Kfactors_G[key] / target_rank_) * size_0_of_all_Kfactors_G[key]**2)
        elif type_of_cost == 'B':
            # in this case target_rank_ is actually target_rank_B + N_BS but leaving same variable name for simplicity
            # diving cost by (target_rank_ + batch_size_) to make numbers more manageable (arguably not needed)
            if size_0_of_all_Kfactors_G[key] / (target_rank_ + batch_size_) > 1: # if we perform a true B-update fr this layer
                computation_time_for_G.append( size_0_of_all_Kfactors_G[key]) #  (target_rank_ + batch_size_) * size_0_of_all_Kfactors_G[key]
            else: # else, we perform an rsvd
                # here we're assuming the rsvd target rank and the brand target rank are the same
                computation_time_for_G.append( min(target_rank_, size_0_of_all_Kfactors_G[key] ) * size_0_of_all_Kfactors_G[key]**2 / (target_rank_ + batch_size_) ) # (target_rank_, size_0_of_all_Kfactors_G[key] ) * size_0_of_all_Kfactors_G[key]**2
    
    ### compute raw allocations
    optimal_allocation_a, optimal_allocation_g = optimal_most_allocation(number_of_workers, computation_time_for_A, computation_time_for_G)
    
    #print(modules_as_keys_list_A); print(modules_as_keys_list_G); print(optimal_allocation_a); print(optimal_allocation_g)
    #print(computation_time_for_A); print(computation_time_for_G)
    ### construct allocatons in desired form
    for module_idx_a, worker_number_a in enumerate(optimal_allocation_a):
        dict_of_lists_of_responsibilities_A[worker_number_a].append(modules_as_keys_list_A[module_idx_a])
        #print('A: Allocated module name = {}, and with index = {} to worker # {}'.format(modules_as_keys_list_A[module_idx_a],module_idx_a ,worker_number_a))
        
    for module_idx_g, worker_number_g in enumerate(optimal_allocation_g):
        dict_of_lists_of_responsibilities_G[worker_number_g].append(modules_as_keys_list_G[module_idx_g])
        #print('G: Allocated module name = {}, and with index = {} to worker # {}'.format(modules_as_keys_list_G[module_idx_g],module_idx_g ,worker_number_g))
    
    return dict_of_lists_of_responsibilities_A, dict_of_lists_of_responsibilities_G

def optimal_most_allocation(number_of_workers, computation_time_for_A, computation_time_for_G):
    # returns the most optimal possible allocation given the desired num workes and the comp times for A and G modules
    # return type is a tupe (list_a, list_g) where for each module-KFCTOR-type pair we say which worker will be allocated ot it
    # the module is inferred by the position in the list, and will be recovered using modules_as_keys_list called with the same idx as the idx of our output
    
    # the RELEVANT optimal defn here is minimize the difference between minimal and maximal total load per worker
    len_A = len(computation_time_for_A) ; len_G = len(computation_time_for_G); total_len = len_A + len_G
    list_alloc_module_a = [];  list_alloc_module_g = []
    if number_of_workers >= total_len: # special case 1
        # do trivial allocation with 1 each
        for idx in range(0, len_A):
            list_alloc_module_a.append(idx)
        for idx in range(0, len_G):
            list_alloc_module_g.append(idx + len_A)
    elif number_of_workers == total_len - 1: #special case 2
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
                current_worker_to_allocate += 1    
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
        
        ##### 2. Initialize sum of current work-load of all workers and lists to be output ###############
        sum_loads_each_worker = [0] * number_of_workers
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
        
    return list_alloc_module_a, list_alloc_module_g

if __name__ == '__main__': ### testing
    computation_time_for_A = [170.00000000000003, 350.00000000000006 ,70, 150, 90]
    computation_time_for_G = [190, 300.00000000000006, 200.00000000000003, 50.00000000000001, 100]
    ### defined numbers above to have the same outcome as squareing the sqrt (which loses an eps-precision and might change selection)
    size_0_of_all_Kfactors_A = {'C1': 170**(0.5), 'C2': 350**(0.5) ,'C3' : 70**(0.5), 'C4': 150**(0.5), 'L1': 90**(0.5)}
    size_0_of_all_Kfactors_G = {'C1':  190**(0.5), 'C2': 300**(0.5) ,'C3' : 200**(0.5), 'C4': 50**(0.5), 'L1': 100**(0.5)}
    
    # realisitc values from my VGG16_bn with less ooling
    """
    computation_time_for_A = [2305**2, 4609**2 ,28**2, 577**2, 1153**2, 4609**2, 16385**2, 2049*82]
    computation_time_for_G = [512**2, 512**2, 64**2, 64**2, 128**2, 512**2, 2048**2, 10**2]
    size_0_of_all_Kfactors_A = {'C1': 2305, 'C2': 4609 ,'C3' : 28, 'C4': 577, 'C5': 1153, 'C6': 4609, 'L1': 16385, 'L2': 2049}
    size_0_of_all_Kfactors_G = {'C1': 512, 'C2': 512, 'C3': 64, 'C4': 64, 'C5': 128, 'C6': 512, 'L1': 2048, 'L2': 10}
    """

    print('TESTING optimal_most_allocation. \n input lists')
    print(computation_time_for_A); print(computation_time_for_G)
    print('Output')
    print(optimal_most_allocation(2, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(3, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(4, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(5, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(6, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(7, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(8, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(9, computation_time_for_A, computation_time_for_G))
    print(optimal_most_allocation(10, computation_time_for_A, computation_time_for_G))
    
    print('TESTING outer function')
    number_of_workers = 7
    
    target_rank_RSVD = 20 #220 # set to very low to avoid the x min(1,m/r) effect!
    print('Output:')
    print(allocate_RSVD_inversion_work_same_fixed_r(number_of_workers, size_0_of_all_Kfactors_G, size_0_of_all_Kfactors_A, target_rank_RSVD))

