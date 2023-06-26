import argparse

def parse_args(solver_name):
    ## LL = Large Linear Layers
    ## CaSL = Conv and small linear layers
    parser = argparse.ArgumentParser()
    parser = arg_parser_add_arguments(parser, solver_name = solver_name)
    
    args = parser.parse_args()
    return args

def arg_parser_add_arguments(parser, solver_name):
    if solver_name == 'KFAC':
        parser = parse_KFAC_specific_arguments(parser)
        #### for efficient work allocaiton selection
        parser.add_argument('--work_alloc_propto_EVD_cost', type=bool, default = True, help = 'Set to True if allocation in proportion to EVD cost is desired. Else naive allocation of equal number of modules for each GPU is done!' )
        
    elif solver_name == 'R-KFAC':
        parser = parse_KFAC_specific_arguments(parser)
        parser = parse_R_specific_arguments(parser)
        ### added to deal with more efficient work allocaiton
        parser.add_argument('--work_alloc_propto_RSVD_cost', type=int, default=1, help='Do we want to allocate work in proportion to FORECASTED (based on theoretical complexity) RSVD cost? set to any non-zero integer if yes. Uing integers as parsing bools with argparse is done wrongly' ) 
        parser.add_argument('--work_eff_alloc_with_time_measurement', type=int, default=0, help='Do we want to allocate work in proportion to MEASURED (somewhat noisy) RSVD cost? set to any non-zero integer if yes. Uing integers as parsing bools with argparse is done wrongly. Setting this to 1 (TRUE) has no effect if work_alloc_propto_RSVD_cost == False' ) 
        # the work_alloc args are slightly different in B and R (even though R also does B) because  
        # (1) in B (also BR and BRC) can only switch eff work alloc for both B and R parts and 
        # (2) B we did not implement time-measurement based eff wokr alloc as it turned out to be weak
        
    elif solver_name == 'B-KFAC':
        parser = parse_KFAC_specific_arguments(parser)
        # the R-specific ones are required because B-kfac only does B for FC layers, for Conv layer it does R !
        parser = parse_R_specific_arguments(parser)
        parser = parse_B_specific_arguments(parser)
        
    elif solver_name == 'BR-KFAC':
        parser = parse_KFAC_specific_arguments(parser)
        parser = parse_R_specific_arguments(parser)
        parser = parse_B_specific_arguments(parser)
        parser = parse_BR_specific_arguments(parser)
        
    elif solver_name == 'BRC-KFAC':
        parser = parse_KFAC_specific_arguments(parser)
        parser = parse_R_specific_arguments(parser)
        parser = parse_B_specific_arguments(parser)
        parser = parse_BR_specific_arguments(parser)
        parser = parse_BRC_specific_arguments(parser)
    
    return parser

def parse_KFAC_specific_arguments(parser): # Adding K-FAC specific arguments
    parser.add_argument('--world_size', type=int, required=True)
    
    parser.add_argument('--kfac_clip', type=int, default=7e-2, help='clip factor for Kfac step' )
    parser.add_argument('--stat_decay', type=int, default=0.95, help='the rho' )
    parser.add_argument('--momentum', type=int, default=0.0, help='momentum' )
    parser.add_argument('--WD', type=int, default=7e-4, help='Weight decay' )
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Batch size for 1 GPU (total BS for grad is n_gpu x *this). Total BS for K-factors is just *this! (for lean-ness)')
    
    ### Others added only once moved to CIFAR10
    parser.add_argument('--n_epochs', type=int, default = 10, help = 'Number_of_epochs' )
    parser.add_argument('--TCov_period', type=int, default = 20, help = 'Period of reupdating Kfactors (not inverses)' )
    parser.add_argument('--TInv_period', type=int, default = 100, help = 'Period of reupdating K-factor INVERSE REPREZENTATIONS' )
    
    ### for selecting net type
    parser.add_argument('--net_type', type=str, default = 'VGG16_bn_lmxp', help = 'Possible Choices: VGG16_bn_lmxp, FC_CIFAR10 (gives an adhoc FC net for CIFAR10), resnet##, resnet##_corrected. Simple_net_for_MNIST is also possible and works only for MNIST: changed to VGG16_bn_lmxp if dataset is other than MNIST and the -for_MNIST net is selected' )
    
    ### for dealing with data path (where the dlded dataset is stored) and dataset itself
    parser.add_argument('--data_root_path', type=str, default = '/data/math-opt-ml/', help = 'fill with path to download data at that root path. Note that you do not need to change this based on the dataset, it will change automatically: each dataset will have its sepparate folder witin the root_data_path directory!' )
    parser.add_argument('--dataset', type=str, default = 'cifar10', help = 'Possible Choices: MNIST, cifar10, imagenet. Case sensitive! Anything else will throw an error. Using imagenet with resnet##_corrected net will force the net to turn to resnet##.' )
    
    ############# SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--TInv_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the TInv_schedule (schedule dict for TInv) from solver/schedules/KFAC_schedules.py' ) 
    parser.add_argument('--TCov_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the TCov_schedule (schedule dict for TCov) from solver/schedules/KFAC_schedules.py' ) 
    ###for dealing with other optimizer schedules
    parser.add_argument('--KFAC_damping_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the KFAC_damping_schedule (schedule dict for KFAC_damping) from solver/schedules/KFAC_schedules.py . If set to 0, a default schedule is used within the main file. Constant values can be easily achieved by altering the schedule to say {0: 0.1} for instance' ) 
    
    ############# END: SCHEDULE FLAGS #################################################

    return parser

def parse_R_specific_arguments(parser):
    
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
    parser.add_argument('--rsvd_rank_adaptation_TInv_multiplier', type = int, default = 5, help = 'After rsvd_rank_adaptation_TInv_multiplier * TInv steps we reconsider ranks')
    parser.add_argument('--rsvd_adaptive_max_history', type = int, default = 30, help = 'Limits the number of previous used ranks and their errors stored to cap memory, cap computation, and have only recent info')
    
    return parser

def parse_B_specific_arguments(parser):
    
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
    
    ############# SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--brand_update_multiplier_to_TCov_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the brand_update_multiplier_to_TCov (schedule dict for brand_update_multiplier_to_TCov) from solver/schedules/B_schedules.py' ) 
    ############# END: SCHEDULE FLAGS ################################################
    
    return parser

def parse_BR_specific_arguments(parser):
    
    parser.add_argument('--B_R_period', type=int, default=5, help='The factor by which (for Linear layers) the RSVDperiod is larger (lower freuency for higher brand_period). (Multiplies TInv).' )
    parser.add_argument('--B_R_period_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the B_R_period_schedule (schedule dict for B_R_period) from solver/schedules/BR_schedules.py . Note: B_R_period multiplies TInv to get how many iterations between an R-update to B-Layers (ie LL layers)' ) 
    
    return parser

def parse_BRC_specific_arguments(parser):
    ### for dealing with the correction (the C in B-R-C)
    ### strictly speaking the C can't be switch off, it's always on, if you want off, use B-R. Can set very large to have it off practically, in whihc case we're doing B-R
    parser.add_argument('--correction_multiplier_TCov', type=int, default=5, help='How often to correct (a partial RSVD) the LL B-update representation' )
    parser.add_argument('--brand_corection_dim_frac', type=float, default=0.2, help='what percentage of modes to refresh in the correction (avoid using close to 100% - at 100% the correction is as expensive an an RSVD and doing an RSVD is cheaper - in that case use B-R with higher "R" requency (for LLs)' )
    
    ############# SCHEDULE FLAGS #####################################################
    ### for dealing with PERIOD SCHEDULES
    parser.add_argument('--correction_multiplier_TCov_schedule_flag', type=int, default = 0, help='Set to any non-zero integer if we want to use the correction_multiplier_TCov_schedule (schedule dict for correction_multiplier_TCov) from solver/schedules/BRC_schedules.py . Note: correction_multiplier_TCov multiplies TInv to get how many iterations between an R-update to B-Layers (ie LL layers)' )
    ############# END: SCHEDULE FLAGS #################################################
    
    return parser