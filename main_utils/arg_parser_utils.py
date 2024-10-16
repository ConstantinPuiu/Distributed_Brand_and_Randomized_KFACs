import argparse

def parse_args(solver_name):
    # rank here is the GPU rank (i.e. GPU index)
    ## LL = Large Linear Layers
    ## CaSL = Conv and small linear layers
    parser = argparse.ArgumentParser()
    parser = arg_parser_add_arguments(parser, solver_name = solver_name)
    
    # parse args
    args = parser.parse_args()
            
    return args

def adjust_args_for_0_1_and_compatibility(args, rank, solver_name):
    ###########################################################################
    #### transform 0-1 attributes = variables into True/False #################
    ###########################################################################
    
    ############## helper fct defn ####################
    def turn_0_1_var_into_T_F(v):
        if v == 0:
            return False
        else:
            return True
    ############## end helper fct defn #################
    
    args.test_at_end = turn_0_1_var_into_T_F(args.test_at_end)
    args.store_and_save_metrics = turn_0_1_var_into_T_F(args.store_and_save_metrics)
    args.print_tqdm_progress_bar = turn_0_1_var_into_T_F(args.print_tqdm_progress_bar)
    args.auto_scale_forGPUs_and_BS = turn_0_1_var_into_T_F(args.auto_scale_forGPUs_and_BS)
    args.stop_at_test_acc = turn_0_1_var_into_T_F(args.stop_at_test_acc)
    
    if solver_name == 'SGD':
        args.use_nesterov = turn_0_1_var_into_T_F(args.use_nesterov)
    elif solver_name == 'KFAC':
        args.work_alloc_propto_EVD_cost = turn_0_1_var_into_T_F(args.work_alloc_propto_EVD_cost)
        
    elif solver_name == 'R-KFAC':
        args.work_alloc_propto_RSVD_cost = turn_0_1_var_into_T_F(args.work_alloc_propto_RSVD_cost)
        args.work_eff_alloc_with_time_measurement = turn_0_1_var_into_T_F(args.work_eff_alloc_with_time_measurement)
        args.adaptable_rsvd_rank = turn_0_1_var_into_T_F(args.adaptable_rsvd_rank)
        
    elif solver_name == 'B-KFAC':
        args.adaptable_rsvd_rank = turn_0_1_var_into_T_F(args.adaptable_rsvd_rank)
        args.work_alloc_propto_RSVD_and_B_cost = turn_0_1_var_into_T_F(args.work_alloc_propto_RSVD_and_B_cost)
        args.adaptable_B_rank = turn_0_1_var_into_T_F(args.adaptable_B_rank)
        args.B_truncate_before_inversion = turn_0_1_var_into_T_F(args.B_truncate_before_inversion)
        
    elif solver_name == 'BR-KFAC':
        args.adaptable_rsvd_rank = turn_0_1_var_into_T_F(args.adaptable_rsvd_rank)
        args.work_alloc_propto_RSVD_and_B_cost = turn_0_1_var_into_T_F(args.work_alloc_propto_RSVD_and_B_cost)
        args.adaptable_B_rank = turn_0_1_var_into_T_F(args.adaptable_B_rank)
        args.B_truncate_before_inversion = turn_0_1_var_into_T_F(args.B_truncate_before_inversion)
        
    elif solver_name == 'BRC-KFAC':
        args.adaptable_rsvd_rank = turn_0_1_var_into_T_F(args.adaptable_rsvd_rank)
        args.work_alloc_propto_RSVD_and_B_cost = turn_0_1_var_into_T_F(args.work_alloc_propto_RSVD_and_B_cost)
        args.adaptable_B_rank = turn_0_1_var_into_T_F(args.adaptable_B_rank)
        args.B_truncate_before_inversion = turn_0_1_var_into_T_F(args.B_truncate_before_inversion)
    else:
        raise ValueError('solver_name = {} is not a valid choice, see source code and implement if required !'.format(solver_name))
    
    ###########################################################################
    #### END: transform 0-1 attributes = variables into True/False ############
    ###########################################################################
    
    # ======== check if args.dataset, args.net_type) pair can go together correctly #===
    # and change net if not so==========================================================
    if args.dataset == 'imagenet': # for imagenet, if we selected the corrected version of VGG (1hich is only for CIFAR10, ignore the corrected part)
        if '_corrected' in args.net_type and 'resnet' in args.net_type:
            args.net_type = args.net_type.replace('_corrected', '')
    
    if args.dataset == 'MNIST':
        # make sure we did not select a net which cna't run with MNIST< namely anything apart form the simple MNIST net
        if args.net_type != 'Simple_net_for_MNIST':
            print('rank:{}. Because dataset == MNIST we can only use the Simple_net_for_MNIST net, so overwriting given parameter as such'.format(rank))
        args.net_type = 'Simple_net_for_MNIST'
    else:
        if args.net_type == 'Simple_net_for_MNIST':
            print('args.net_type = Simple_net_for_MNIST is only possible when args.dataset == MNIST, but args.dataset is {}. Changing args.net_type to default: VGG16_bn_lmxp'.format(args.dataset))
            args.net_type = 'VGG16_bn_lmxp'
    # ===================================================================================
        
    return args

def adjust_args_for_schedules(args, solver_name):
    # run  as: args, ... (dependend on solver_name) = adjust_args_for_schedules(args, solver_name)
    if solver_name == 'SGD':
        ################################ SGD SCHEDULES ######################################################################
        if args.momentum_dampening_schedule_flag == 0: # then it's False
            momentum_dampening_schedule = {} # empty dictionary - no scheduling "enforcement"
        else:# if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.SGD_schedules import momentum_dampening_schedule
            if 0 in momentum_dampening_schedule.keys():
                print('Because --momentum_dampening_schedule_flag was set to non-zero (True) and momentum_dampening_schedule[0] exists, we overwrite momentum_dampening = {} (as passed in --momentum_dampening) to TCov_schedule[0] = {}'.format(args.momentum_dampening, momentum_dampening_schedule[0]))
                args.momentum_dampening = momentum_dampening_schedule[0]
        return args, momentum_dampening_schedule
        ################################ END: SGD SCHEDULES ##################################################################
    
    elif solver_name == 'KFAC':
        ################################ KFAC SCHEDULES ######################################################################
        ### for dealing with PERIOD SCHEDULES
        if args.TInv_schedule_flag == 0: # then it's False
            TInv_schedule = {} # empty dictionary - no scheduling "enforcement"
        else:# if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.KFAC_schedules import TInv_schedule
            if 0 in TInv_schedule.keys(): # overwrite TInv_period
                print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(args.TInv_period, TInv_schedule[0]))
                args.TInv_period = TInv_schedule[0]
        
        if args.TCov_schedule_flag == 0: # then it's False
            TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.KFAC_schedules import TCov_schedule
            if 0 in TCov_schedule.keys(): # overwrite TInv_period
                print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(args.TCov_period, TCov_schedule[0]))
                args.TCov_period = TCov_schedule[0]
        #########################################
                
        ### for dealing with other parameters SCHEDULES ####
        if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
            KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
        else:
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.KFAC_schedules import KFAC_damping_schedule
        KFAC_damping = KFAC_damping_schedule[0]
        ################################ END KFAC SCHEDULES ###################################################################
        return args, TInv_schedule, TCov_schedule, KFAC_damping_schedule, KFAC_damping
    
    elif solver_name == 'R-KFAC':
        ################################ R-KFAC SCHEDULES ######################################################################
        ### for dealing with PERIOD SCHEDULES
        if args.TInv_schedule_flag == 0: # then it's False
            TInv_schedule = {} # empty dictionary - no scheduling "enforcement"
        else:# if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.R_schedules import TInv_schedule
            if 0 in TInv_schedule.keys(): # overwrite TInv_period
                print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(args.TInv_period, TInv_schedule[0]))
                args.TInv_period = TInv_schedule[0]
        
        if args.TCov_schedule_flag == 0: # then it's False
            TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.R_schedules import TCov_schedule
            if 0 in TCov_schedule.keys(): # overwrite TInv_period
                print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(args.TCov_period, TCov_schedule[0]))
                args.TCov_period = TCov_schedule[0]
        #########################################
                
        ### for dealing with other parameters SCHEDULES ####
        if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
            KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
        else:
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.R_schedules import KFAC_damping_schedule
        KFAC_damping = KFAC_damping_schedule[0]
        ################################ END R-KFAC SCHEDULES ###################################################################
        return args, TInv_schedule, TCov_schedule, KFAC_damping_schedule, KFAC_damping
    
    elif solver_name == 'B-KFAC':
        ################################  B SCHEDULES ######################################################################
        ### for dealing with PERIOD SCHEDULES
        if args.TInv_schedule_flag == 0: # then it's False
            TInv_schedule = {} # empty dictionary - no scheduling "enforcement"
        else:# if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import TInv_schedule
            if 0 in TInv_schedule.keys(): # overwrite TInv_period
                print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(args.TInv_period, TInv_schedule[0]))
                args.TInv_period = TInv_schedule[0]
        
        if args.TCov_schedule_flag == 0: # then it's False
            TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import TCov_schedule
            if 0 in TCov_schedule.keys(): # overwrite TInv_period
                print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(args.TCov_period, TCov_schedule[0]))
                args.TCov_period = TCov_schedule[0]
        
        if args.brand_update_multiplier_to_TCov_schedule_flag == 0: # then it's False
            brand_update_multiplier_to_TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import brand_update_multiplier_to_TCov_schedule
            if 0 in brand_update_multiplier_to_TCov_schedule.keys(): # overwrite TInv_period
                print('Because --brand_update_multiplier_to_TCov_schedule_flag was set to non-zero (True) and brand_update_multiplier_to_TCov_schedule[0] exists, we overwrite brand_update_multiplier_to_TCov = {} (as passed in --brand_update_multiplier_to_TCov) to brand_update_multiplier_to_TCov_schedule[0] = {}'.format(args.brand_update_multiplier_to_TCov, brand_update_multiplier_to_TCov_schedule[0]))
                args.brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov_schedule[0]
        #########################################
                
        ### for dealing with other parameters SCHEDULES ####
        if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
            KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
        else:
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.B_schedules import KFAC_damping_schedule
        KFAC_damping = KFAC_damping_schedule[0]
        ################################ END B SCHEDULES ###################################################################
        return args, TInv_schedule, TCov_schedule, brand_update_multiplier_to_TCov_schedule, KFAC_damping_schedule, KFAC_damping
        
    elif solver_name == 'BR-KFAC':
        ################################  BR SCHEDULES ######################################################################
        ### for dealing with PERIOD SCHEDULES
        if args.TInv_schedule_flag == 0: # then it's False
            TInv_schedule = {} # empty dictionary - no scheduling "enforcement"
        else:# if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BR_schedules import TInv_schedule
            if 0 in TInv_schedule.keys(): # overwrite TInv_period
                print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(args.TInv_period, TInv_schedule[0]))
                args.TInv_period = TInv_schedule[0]
        
        if args.TCov_schedule_flag == 0: # then it's False
            TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BR_schedules import TCov_schedule
            if 0 in TCov_schedule.keys(): # overwrite TInv_period
                print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(args.TCov_period, TCov_schedule[0]))
                args.TCov_period = TCov_schedule[0]
        
        if args.brand_update_multiplier_to_TCov_schedule_flag == 0: # then it's False
            brand_update_multiplier_to_TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BR_schedules import brand_update_multiplier_to_TCov_schedule
            if 0 in brand_update_multiplier_to_TCov_schedule.keys(): # overwrite TInv_period
                print('Because --brand_update_multiplier_to_TCov_schedule_flag was set to non-zero (True) and brand_update_multiplier_to_TCov_schedule[0] exists, we overwrite brand_update_multiplier_to_TCov = {} (as passed in --brand_update_multiplier_to_TCov) to brand_update_multiplier_to_TCov_schedule[0] = {}'.format(args.brand_update_multiplier_to_TCov, brand_update_multiplier_to_TCov_schedule[0]))
                args.brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov_schedule[0]
        
        if args.B_R_period_schedule_flag == 0: # then it's False
            B_R_period_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BR_schedules import B_R_period_schedule
            if 0 in B_R_period_schedule.keys(): # overwrite TInv_period
                print('Because --B_R_period_schedule_flag was set to non-zero (True) and B_R_period_schedule[0] exists, we overwrite B_R_period = {} (as passed in --B_R_period) to B_R_period_schedule[0] = {}'.format(args.B_R_period, B_R_period_schedule[0]))
                args.B_R_period = B_R_period_schedule[0]
        #########################################
            
        ### for dealing with other parameters SCHEDULES ####
        if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
            KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
        else:
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BR_schedules import KFAC_damping_schedule
        KFAC_damping = KFAC_damping_schedule[0]
        ################################ END BR SCHEDULES ###################################################################
        return args, TInv_schedule, TCov_schedule, brand_update_multiplier_to_TCov_schedule, B_R_period_schedule, KFAC_damping_schedule, KFAC_damping
    
    elif solver_name == 'BRC-KFAC':
        ################################  BRC SCHEDULES ######################################################################
        ### for dealing with PERIOD SCHEDULES
        if args.TInv_schedule_flag == 0: # then it's False
            TInv_schedule = {} # empty dictionary - no scheduling "enforcement"
        else:# if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BRC_schedules import TInv_schedule
            if 0 in TInv_schedule.keys(): # overwrite TInv_period
                print('Because --TInv_schedule_flag was set to non-zero (True) and TInv_schedule[0] exists, we overwrite TInv_period = {} (as passed in --TInv_period) to TInv_schedule[0] = {}'.format(args.TInv_period, TInv_schedule[0]))
                args.TInv_period = TInv_schedule[0]
        
        if args.TCov_schedule_flag == 0: # then it's False
            TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BRC_schedules import TCov_schedule
            if 0 in TCov_schedule.keys(): # overwrite TInv_period
                print('Because --TCov_schedule_flag was set to non-zero (True) and TCov_schedule[0] exists, we overwrite TCov_period = {} (as passed in --TCov_period) to TCov_schedule[0] = {}'.format(args.TCov_period, TCov_schedule[0]))
                args.TCov_period = TCov_schedule[0]
        
        if args.brand_update_multiplier_to_TCov_schedule_flag == 0: # then it's False
            brand_update_multiplier_to_TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BRC_schedules import brand_update_multiplier_to_TCov_schedule
            if 0 in brand_update_multiplier_to_TCov_schedule.keys(): # overwrite TInv_period
                print('Because --brand_update_multiplier_to_TCov_schedule_flag was set to non-zero (True) and brand_update_multiplier_to_TCov_schedule[0] exists, we overwrite brand_update_multiplier_to_TCov = {} (as passed in --brand_update_multiplier_to_TCov) to brand_update_multiplier_to_TCov_schedule[0] = {}'.format(args.brand_update_multiplier_to_TCov, brand_update_multiplier_to_TCov_schedule[0]))
                args.brand_update_multiplier_to_TCov = brand_update_multiplier_to_TCov_schedule[0]
                
        if args.correction_multiplier_TCov_schedule_flag == 0: # then it's False
            correction_multiplier_TCov_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BRC_schedules import correction_multiplier_TCov_schedule
            if 0 in brand_update_multiplier_to_TCov_schedule.keys(): # overwrite TInv_period
                print('Because --correction_multiplier_TCov_schedule_flag was set to non-zero (True) and correction_multiplier_TCov_schedule[0] exists, we overwrite correction_multiplier_TCov = {} (as passed in --correction_multiplier_TCov) to correction_multiplier_TCov_schedule[0] = {}'.format(args.correction_multiplier_TCov, correction_multiplier_TCov_schedule[0]))
                args.correction_multiplier_TCov = correction_multiplier_TCov_schedule[0]
        
        if args.B_R_period_schedule_flag == 0: # then it's False
            B_R_period_schedule = {} # empty dictionary - no scheduling "enforcement"
        else: # if the flag is True
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BRC_schedules import B_R_period_schedule
            if 0 in B_R_period_schedule.keys(): # overwrite TInv_period
                print('Because --B_R_period_schedule_flag was set to non-zero (True) and B_R_period_schedule[0] exists, we overwrite B_R_period = {} (as passed in --B_R_period) to B_R_period_schedule[0] = {}'.format(args.B_R_period, B_R_period_schedule[0]))
                args.B_R_period = B_R_period_schedule[0]
        #########################################
                
        ### for dealing with other parameters SCHEDULES ####
        if args.KFAC_damping_schedule_flag == 0: # if we don't set the damping shcedule in R_schedules.py, use DEFAULT (as below)
            KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
        else:
            from Distributed_Brand_and_Randomized_KFACs.solvers.schedules.BR_schedules import KFAC_damping_schedule
        KFAC_damping = KFAC_damping_schedule[0]
        ################################ END BRC SCHEDULES ###################################################################
        return args, TInv_schedule, TCov_schedule, brand_update_multiplier_to_TCov_schedule, correction_multiplier_TCov_schedule, B_R_period_schedule, KFAC_damping_schedule, KFAC_damping
    
    else:
        raise ValueError('solver_name = {} is not a valid choice, see source code and implement if required !'.format(solver_name))

def arg_parser_add_arguments(parser, solver_name):
    if solver_name == 'SGD':
        parser = parse_basic_arguments(parser)
        parser = parse_SGD_specific_arguments(parser)
    
    elif solver_name == 'KFAC_save_eigenspectrums':
        parser = parse_basic_arguments(parser)
        parser = parse_KFAC_specific_arguments(parser)
        #### for efficient work allocaiton selection
        parser.add_argument('--work_alloc_propto_EVD_cost', type=bool, default = True, help = 'Set to True if allocation in proportion to EVD cost is desired. Else naive allocation of equal number of modules for each GPU is done!' )
        # for saving eigenspectrums
        parser.add_argument('--Kfactor_spectrum_savepath', type = str, default = './', help = 'path for saving eigenspectrums')
        parser.add_argument('--Network_scalefactor', type = float, default = 1.0, help = 'network FATTENING (scaling) parameter - all layers are made so many times as big to help with eigenspectrum behavior investigation')
    
    elif solver_name == 'KFAC':
        parser = parse_basic_arguments(parser)
        parser = parse_KFAC_specific_arguments(parser)
        #### for efficient work allocaiton selection
        parser.add_argument('--work_alloc_propto_EVD_cost', type=bool, default = True, help = 'Set to True if allocation in proportion to EVD cost is desired. Else naive allocation of equal number of modules for each GPU is done!' )
        
    elif solver_name == 'R-KFAC':
        parser = parse_basic_arguments(parser)
        parser = parse_KFAC_specific_arguments(parser)
        parser = parse_R_specific_arguments(parser)
        ### added to deal with more efficient work allocaiton
        parser.add_argument('--work_alloc_propto_RSVD_cost', type=int, default=1, help='Do we want to allocate work in proportion to FORECASTED (based on theoretical complexity) RSVD cost? set to any non-zero integer if yes. Uing integers as parsing bools with argparse is done wrongly' ) 
        parser.add_argument('--work_eff_alloc_with_time_measurement', type=int, default=0, help='Do we want to allocate work in proportion to MEASURED (somewhat noisy) RSVD cost? set to any non-zero integer if yes. Uing integers as parsing bools with argparse is done wrongly. Setting this to 1 (TRUE) has no effect if work_alloc_propto_RSVD_cost == False' ) 
        # the work_alloc args are slightly different in B and R (even though R also does B) because  
        # (1) in B (also BR and BRC) can only switch eff work alloc for both B and R parts and 
        # (2) B we did not implement time-measurement based eff wokr alloc as it turned out to be weak
        
    elif solver_name == 'B-KFAC':
        parser = parse_basic_arguments(parser)
        parser = parse_KFAC_specific_arguments(parser)
        # the R-specific ones are required because B-kfac only does B for FC layers, for Conv layer it does R !
        parser = parse_R_specific_arguments(parser)
        parser = parse_B_specific_arguments(parser)
        
    elif solver_name == 'BR-KFAC':
        parser = parse_basic_arguments(parser)
        parser = parse_KFAC_specific_arguments(parser)
        parser = parse_R_specific_arguments(parser)
        parser = parse_B_specific_arguments(parser)
        parser = parse_BR_specific_arguments(parser)
        
    elif solver_name == 'BRC-KFAC':
        parser = parse_basic_arguments(parser)
        parser = parse_KFAC_specific_arguments(parser)
        parser = parse_R_specific_arguments(parser)
        parser = parse_B_specific_arguments(parser)
        parser = parse_BR_specific_arguments(parser)
        parser = parse_BRC_specific_arguments(parser)
    else:
        raise ValueError('Solver: args.solver_name = {} not implemented'.format(solver_name))
    
    return parser

def parse_basic_arguments(parser): # Adding arguments to ALL solvers
    parser.add_argument('--world_size', type=int, required=True)
    
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum' )
    parser.add_argument('--WD', type=float, default=7e-4, help='Weight decay' )
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Batch size for 1 GPU (total BS for grad is n_gpu x *this).')
    
    parser.add_argument('--n_epochs', type=int, default = 10, help = 'Number_of_epochs' )
    
    ### for selecting net type
    parser.add_argument('--net_type', type=str, default = 'VGG16_bn_lmxp', help = 'Possible Choices: VGG16_bn_lmxp, FC_CIFAR10 (gives an adhoc FC net for CIFAR10), resnet##, resnet##_corrected. Simple_net_for_MNIST is also possible and works only for MNIST: changed to VGG16_bn_lmxp if dataset is other than MNIST and the -for_MNIST net is selected' )
    
    ### for dealing with data path (where the dlded dataset is stored) and dataset itself
    parser.add_argument('--data_root_path', type=str, default = '/data/math-opt-ml/', help = 'fill with path to download data at that root path. Note that you do not need to change this based on the dataset, it will change automatically: each dataset will have its sepparate folder witin the root_data_path directory!' )
    parser.add_argument('--dataset', type=str, default = 'cifar10', help = 'Possible Choices: MNIST, SVHN, cifar10, cifar100, imagenet, imagenette_fs_v2. Case sensitive! Anything else will throw an error. Using imagenet with resnet##_corrected net will force the net to turn to resnet##.' )
    
    ### for slecting when / if to do the test() function
    parser.add_argument('--test_at_end', type=int, default = 0, help='Set to 1 to perform a test at the end of the training' ) 
    parser.add_argument("--test_every_X_epochs", type = int, default = 10000, help = 'If you want to perform a test more frequently than just at the end. Default set to very high value to avoid. Set to 1 if you wish to test after each epoch.')
    
    ### for deciding whether to store metrics in lists or not
    parser.add_argument('--store_and_save_metrics', type=int, default = 0, help='Set to 1 to store metrics acquired during train and test' ) 
    parser.add_argument('--metrics_save_path', type=str, default = '/data/math-opt-ml/saved_metrics/', help = 'fill with path to download data at that root path. Note that you do not need to change this based on the dataset, it will change automatically: each dataset will have its sepparate folder witin the root_data_path directory!' )
    
    ### for seeding rng
    parser.add_argument('--seed', type=int, default = -1 , help='Set to -1 to avoid seeding. Otherwise seed at given number' ) 
    
    ### for choosing whether we have tqdm or not - might chage it later and integrate with some "verbose"-"nonverbose" choise and include the prints too
    parser.add_argument('--print_tqdm_progress_bar', type=int, default = 0, help='Set to 0 NOT to print TQDM progress bars. Anything other than 0 will print progress bars' ) 
    
    ### for choosing to stop early
    parser.add_argument('--stop_at_test_acc', type=int, default = 0, help='Set to 1 to stop immediately once test accuracy reaches --stopping_test_acc / args.stopping_test_acc\
                        Note that tests are only performed once in args.test_every_X_epochs, and that frequency is relevant here' ) 
    parser.add_argument('--stopping_test_acc', type=float, default = 99.75, help='if ==True, stop immediately once the test accuracy reaches *this threshold' ) 
    # 
    
    #### for lr schedules ############################################################
    parser.add_argument('--lr_schedule_type', type = str, default = 'exp', help='possible values `constant`, `cos`, `exp`, `stair`, `from_file`.\
                        From file lets you code the lr from file, different for each dataset, only the --dataset parameter is relevant to "from_file". \
                        The dataset param is not relevant to any other lr_schedule_type values' )
    parser.add_argument('--base_lr', type = float, default = 0.3, help='The Lr we begin with. Relevant to all lr_schedule_type in [`constant`,  `cos`, `exp`, `stair`]' )
    parser.add_argument('--lr_decay_rate', type = float, default = 9, help='Controls how strong the decay is if lr_schedule_type in [str(exp), str(stair)]\
                        decay rate (lr_decay_rate) needs to be > 1 to actually get a decay for all settings to which it is relevant' )
    parser.add_argument('--lr_decay_period', type = int, default = 80, help='for `exp` : the epoch at which lr becomes zero, exp-decaying towards zero at that epoch\
                        for `stair` how many epochs with the same lr to perform before dropping lr by a factor of lr_decay_rate. \
                        Set to larger than the number of training epochs.\
                        For Cos it is the Cosine Period factor. Not relevant to `constant`' )
    parser.add_argument('--auto_scale_forGPUs_and_BS', type = int, default = 0, help = 'Switch on to have lr schedule autmoatically scaled with GPUs and total batch-size.\
                        If on, scales lr by sqrt(total_batchsize) as typical and the periods by total batchsize - to ensure fixed\
                        schedule in number of steps (rather than epochs). Switching on will result in the lr and lr schedule\
                        be different from what is set: the set values are for 1 GPU at 256, and scaled appropriately. \
                        Switch off for more control, switch on for easier deal with increasing GPU numbers once good lr is found.')
    
    return parser

def parse_SGD_specific_arguments(parser): # Adding arguments to ALL solvers
    parser.add_argument('--use_nesterov', type = int, default = 0, help = '0-1 True-False scheme: set to True for momentum type to be nesterov.')
    parser.add_argument('--momentum_dampening', type = float, default = 0.0, help = 'dampening for momentum')
    
    ############# (non-lr) SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--momentum_dampening_schedule_flag', type=int, default = 0, 
                        help='Set to any non-zero integer if we want to use the \
                            momentum_dampening_schedule (schedule dict for TInv) \
                            from solver/schedules/SGD_schedules.py' ) 
    ############# END: (non-lr) SCHEDULE FLAGS #################################################
    
    return parser

def parse_KFAC_specific_arguments(parser): # Adding K-FAC specific arguments
    # the args here are added to all K-FAC variants, including R, B, BR BRC
    parser.add_argument('--kfac_clip', type=float, default=7e-2, help='clip factor for Kfac step' )
    parser.add_argument('--stat_decay', type=float, default=0.95, help='the rho' )
    
    ### Specific to inversion
    parser.add_argument('--TCov_period', type=int, default = 20, help = 'Period of reupdating Kfactors (not inverses)' )
    parser.add_argument('--TInv_period', type=int, default = 100, help = 'Period of reupdating K-factor INVERSE REPREZENTATIONS' )
    
    ############# (non-lr) SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--TInv_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the TInv_schedule (schedule dict for TInv) from solver/schedules/KFAC_schedules.py' ) 
    parser.add_argument('--TCov_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the TCov_schedule (schedule dict for TCov) from solver/schedules/KFAC_schedules.py' ) 
    ###for dealing with other optimizer schedules
    parser.add_argument('--KFAC_damping_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the KFAC_damping_schedule (schedule dict for KFAC_damping) from solver/schedules/KFAC_schedules.py . If set to 0, a default schedule is used within the main file. Constant values can be easily achieved by altering the schedule to say {0: 0.1} for instance' ) 
    
    ############# END: (non-lr) SCHEDULE FLAGS #################################################

    return parser

def parse_R_specific_arguments(parser):
    # the args here are added to R, B, BR, BRC
    
    ### RSVD specific params or stuff that 1st appeared in rsvd(clip and damping type)
    parser.add_argument('--rsvd_rank', type=int, default = 220, help = 'The target rank of RSVD' )
    parser.add_argument('--rsvd_oversampling_parameter', type=int, default = 10, help = 'the oversampling parameter of RSVD' )
    parser.add_argument('--rsvd_niter', type=int, default = 3, help = '# of power(like) iterations in getting projection subspace for RSVD' )
    parser.add_argument('--damping_type', type=str, default= 'adaptive', help = 'type of damping' )
    parser.add_argument('--clip_type', type=str, default = 'non_standard', help = 'Weight decay' )
    #### TO DO: CODE the RSVD to be adaptive, and adaptivity specific to each K-factor
    #### TO DO cond't: preserve the current functionality and add a rsvd_rank_type switch with "adaptive" vs standard 
    
    #### added to dal with RSVD adaptable rank
    parser.add_argument('--adaptable_rsvd_rank', type=int, default = 0, help='Set to any non-zero integer if we want adaptable rank. Uing integers as parsing bools with argparse is done wrongly' ) 
    parser.add_argument('--rsvd_target_truncation_rel_err', type=float, default=0.033, help='target truncation error in rsvd: the ran will adapt to be around this error (but rsvd rank has to be strictly below maximum_ever_admissible_rsvd_rank)' ) 
    parser.add_argument('--maximum_ever_admissible_rsvd_rank', type=int, default=700, help='Rsvd rank has to be strictly below maximum_ever_admissible_rsvd_rank' ) 
    parser.add_argument('--rsvd_rank_adaptation_TInv_multiplier', type = int, default = 1, help = 'After rsvd_rank_adaptation_TInv_multiplier * TInv steps we reconsider ranks')
    parser.add_argument('--rsvd_adaptive_max_history', type = int, default = 30, help = 'Limits the number of previous used ranks and their errors stored to cap memory, cap computation, and have only recent info')
    
    return parser

def parse_B_specific_arguments(parser):
    # the args here are added to B, BR, BRC
    
    ######### BRAND K-fac (also BRSKFAC) specific parameters
    #parser.add_argument('--brand_period', type=int, default=5, help='The factor by which (for Linear layers) the RSVDperiod is larger (lower freuency for higher brand_period)' )
    # this argument above is not used by B-pure-KFAC!
    parser.add_argument('--brand_r_target_excess', type=int, default=0, help='How many more modes to keep in the B-(.) than in the R-(.) reprezentation' )
    parser.add_argument('--brand_update_multiplier_to_TCov', type=int, default=1, help='The factor by which the B-update frequency is LOWER than the frequency at which we reiceve new K-factor information' )
    # ====================================================
      
    ### added to deal with more efficient work allocaiton
    #
    parser.add_argument('--work_alloc_propto_RSVD_and_B_cost', type=int, default=1, help='Do we want to allocate work in proportion to actual RSVD cost, and actual B-update Cost? set to any nonzero if yes. we use int rather than bool as argparse works badly with bool!' ) 
    
    #### added to allow for B-truncating just before inversion as well
    parser.add_argument('--B_truncate_before_inversion', type=int, default=0, help='Do we want to B-truncate just before inversion (more speed less accuracy) If so set to 1 (or anything other than 0). Standard way to deal with bools wiht buggy argparser that only work correctly wiht numbers!' ) 
    
    #### added to deal with B- adaptable rank
    parser.add_argument('--adaptable_B_rank', type=int, default = 0, help='Set to any non-zero integer if we want B- adaptable rank. Uing integers as parsing bools with argparse is done wrongly' ) 
    parser.add_argument('--B_target_truncation_rel_err', type=float, default=0.033, help='target truncation error in B-update_truncation: the rank will adapt to be around this error (but B-truncation rank has to be strictly below maximum_ever_admissible_B_rank and above 70. Unlike rsvd it is not above 10. That is because using B with very small truncation rank effectively means we carry no information from before, in which case B is pointless. If you need smaller minimum admissible value than 70, edit the corresponding function in the file adaptive_rank_utils.py)' ) 
    parser.add_argument('--maximum_ever_admissible_B_rank', type=int, default=500, help='B-truncation rank has to be strictly below maximum_ever_admissible_B_rank' ) 
    parser.add_argument('--B_rank_adaptation_T_brand_updt_multiplier', type = int, default = 5, help = 'After B_rank_adaptation_T_brand_updt_multiplier * TCov * brand_update_multiplier_TCov steps we reconsider ranks')
    parser.add_argument('--B_adaptive_max_history', type = int, default = 30, help = 'Limits the number of previous used ranks and their errors stored to cap memory, cap computation, and have only recent info')
    
    ############# (non-lr) SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--brand_update_multiplier_to_TCov_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the brand_update_multiplier_to_TCov (schedule dict for brand_update_multiplier_to_TCov) from solver/schedules/B_schedules.py' ) 
    ############# END: (non-lr) SCHEDULE FLAGS ################################################
    
    return parser

def parse_BR_specific_arguments(parser):
    # the args here are added to BR and BRC
    parser.add_argument('--B_R_period', type=int, default=5, help='The factor by which (for Linear layers) the RSVDperiod is larger (lower freuency for higher brand_period). (Multiplies TInv).' )
    parser.add_argument('--B_R_period_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the B_R_period_schedule (schedule dict for B_R_period) from solver/schedules/BR_schedules.py . Note: B_R_period multiplies TInv to get how many iterations between an R-update to B-Layers (ie LL layers)' ) 
    
    return parser

def parse_BRC_specific_arguments(parser):
    # the args here are added to BRC only
    ### for dealing with the correction (the C in B-R-C)
    ### strictly speaking the C can't be switch off, it's always on, if you want off, use B-R. Can set very large to have it off practically, in whihc case we're doing B-R
    parser.add_argument('--correction_multiplier_TCov', type=int, default=5, help='How often to correct (a partial RSVD) the LL B-update representation' )
    parser.add_argument('--brand_corection_dim_frac', type=float, default=0.2, help='what percentage of modes to refresh in the correction (avoid using close to 100% - at 100% the correction is as expensive an an RSVD and doing an RSVD is cheaper - in that case use B-R with higher "R" requency (for LLs)' )
    
    ############# (non-lr) SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--correction_multiplier_TCov_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the correction_multiplier_TCov_schedule (schedule dict for correction_multiplier_TCov) from solver/schedules/BRC_schedules.py . Note: correction_multiplier_TCov multiplies TInv to get how many iterations between an R-update to B-Layers (ie LL layers)' )
    ############# END: (non-lr) SCHEDULE FLAGS #################################################
    
    return parser

""" for testing"""
if __name__ == '__main__':
    solver_name = 'BRC-KFAC' # change this (cannot imput from keyboard as this would change main code functionality)
    args = parse_args(solver_name = solver_name)
    args = adjust_args_for_0_1_and_compatibility(args, rank = 0)
    if '_corrected' in args.net_type and 'resnet' in args.net_type:
            args.net_type = args.net_type.replace('_corrected', '')
        
    if solver_name in ['KFAC']:
    #### should throw an error unless --world_size is given! 
        print( 'args.net_type = {}, args.work_alloc_propto_EVD_cost = {}'.format( args.net_type , args.work_alloc_propto_EVD_cost) )
        
    elif solver_name in ['R-KFAC']:
    #### should throw an error unless --world_size is given! 
        print( 'args.net_type = {}, args.work_alloc_propto_RSVD_cost = {}, args.work_eff_alloc_with_time_measurement = {}, args.adaptable_rsvd_rank = {}'.format( args.net_type , args.work_alloc_propto_RSVD_cost, args.work_eff_alloc_with_time_measurement , args.adaptable_rsvd_rank) )
    
    elif solver_name in ['B-KFAC','BR-KFAC', 'BRC-KFAC']:
    #### should throw an error unless --world_size is given! 
        print( 'args.net_type = {}, args.adaptable_rsvd_rank = {}, args.adaptable_B_rank = {}, args.work_alloc_propto_RSVD_and_B_cost = {}, args.B_truncate_before_inversion = {}'.format( args.net_type , args.adaptable_rsvd_rank, args.adaptable_B_rank, args.work_alloc_propto_RSVD_and_B_cost , args.B_truncate_before_inversion) )