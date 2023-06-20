"""format of dictionaries is d = {epoch_index : value}"""
################## period schedules ############################
# Used by : R, B, B-R, B-R-C
TInv_schedule = {0: 100, 5: 50}

# Used by : R, B, B-R, B-R-C
TCov_schedule = {0: 20, 5: 10}

"""The 3 ones below are MULTIPLIERS to other periods, check the main files for documentation when the args are defined before # args = parser.parse_args() # line"""

# Used by : B, B-R, B-R-C
#brand_update_multiplier_to_TCov_schedule = {0: 5, 5: 1}

# Used by : B-R, B-R-C
#B_R_period_schedule = {0 : 5, 5: 2}

# Used by : B-R-C
#correction_multiplier_TCov_schedule = {0:5, 5: 2}
############# END: period schedules ############################

################## other optim hyperparam schedules ############################
KFAC_damping_schedule = {0: 1e-01, 7: 1e-01, 25: 5e-02, 35: 1e-02}
############# END: other optim hyperparam schedules ############################
