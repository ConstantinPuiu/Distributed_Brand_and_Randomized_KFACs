

def get_new_rsvd_rank(list_err, list_ranks, max_rank = 700, target_rel_err = 0.033, type_of_prediction = 'simplistic', TInv_multiplier = 1):
    # the lists correspond in index: the list_err[0] correspinds to the rank list_rank[0] etc
    # TInv_multiplier = self.rank_adaptation_TInv_multiplier  -- tells us how manypast items in the list (at max) are "samples" of the same thing
    if type_of_prediction == 'simplistic':
        # we use only the last point in a trivial fashion, no prediction is made...
        # steadily growing the rank until desired target_rel_err is achieved - be conservative on the increase side
        # we are aggressive on the decrease side in terms of the condition, but not so aggressive with the step
        prev_used_rank = list_ranks[-1]
        numel_to_consider = min(TInv_multiplier, len(list_err)) # adding the list_err part to ensure we don't ask for more "datapts" than length of list
        # but if we set TInv_multiplier = self.rank_adaptation_TInv_multiplier, the min should always return TInv_multiplier
        
        ###### Compute averate relative error for past "same chunk"
        avg_rel_err = sum(list_err[-numel_to_consider:])/numel_to_consider
        
        ## error margins - can take them as arguments later, but it's ok like this for now,a nd maybe forever###
        err_margin_to_incr = 0.02
        err_margin_to_decrease = 0.007
        rank_step = 10
        ## error margins - can take them as arguments later, but it's ok like this for now,a nd maybe forever###
        if target_rel_err - err_margin_to_decrease > avg_rel_err: # then we're too accurate, DECREASE target rank
            new_rsvd_rank = max(prev_used_rank - rank_step, 10)
            new_rsvd_rank = min(new_rsvd_rank, max_rank)
        elif target_rel_err +  err_margin_to_incr < avg_rel_err: # then we're not accurate enough, INCREASE target rank
            new_rsvd_rank = min(prev_used_rank + rank_step, max_rank)
    else:
        # here we will implement more ML - based methods using previous history to predict which rank we should use. Trivial one above is pretty sufficient...
        raise NotImplementedError('type of prediction #{}# for funciton get_new_rsvd_rank NOT implemented (adaptive rank allocation)')
    
    return new_rsvd_rank

if __name__ == '__main__': ### testing
    ###  test 1
    list_err = [0.21, 0.24, 0.23, 0.22, 0.25, 0.26, 0.17, 0.12, 0.11, 0.16, 0.14 ]
    list_ranks = [220, 220, 220, 220, 220, 220, 200, 200, 200, 200, 200]
    rsR1 = get_new_rsvd_rank(list_err, list_ranks, max_rank = 700, target_rel_err = 0.033, type_of_prediction = 'simplistic', TInv_multiplier = 5)
    print('test 1, rank = {}'.format(rsR1))
    
    ###  test 2
    list_err = [0.21, 0.24, 0.23, 0.22, 0.25, 0.26, 0.17, 0.12, 0.11, 0.16, 0.14 ]
    list_ranks = [220, 220, 220, 220, 220, 220, 200, 200, 200, 200, 200]
    rsR1 = get_new_rsvd_rank(list_err, list_ranks, max_rank = 205, target_rel_err = 0.033, type_of_prediction = 'simplistic', TInv_multiplier = 5)
    print('test 2, rank = {}'.format(rsR1))
    
    ###  test 3
    list_err = [0.21, 0.24, 0.23, 0.22, 0.25, 0.26, 0.02, 0.021, 0.011, 0.016, 0.014 ]
    list_ranks = [220, 220, 220, 220, 220, 220, 200, 200, 200, 200, 200]
    rsR1 = get_new_rsvd_rank(list_err, list_ranks, max_rank = 700, target_rel_err = 0.033, type_of_prediction = 'simplistic', TInv_multiplier = 5)
    print('test 2, rank = {}'.format(rsR1))
    
    ###  test 4
    list_err = [0.21, 0.24, 0.23, 0.22, 0.25, 0.26, 0.02, 0.021, 0.011, 0.016, 0.014 ]
    list_ranks = [220, 220, 220, 220, 220, 220, 21, 21, 21, 21, 21]
    rsR1 = get_new_rsvd_rank(list_err, list_ranks, max_rank = 700, target_rel_err = 0.033, type_of_prediction = 'simplistic', TInv_multiplier = 5)
    print('test 4, rank = {}'.format(rsR1))
    
    ###  test 5
    list_err = [0.21, 0.24, 0.23, 0.22, 0.25, 0.26, 0.02, 0.021, 0.011, 0.016, 0.014 ]
    list_ranks = [220, 220, 220, 220, 220, 220, 19, 19, 19, 19, 19]
    rsR1 = get_new_rsvd_rank(list_err, list_ranks, max_rank = 700, target_rel_err = 0.033, type_of_prediction = 'simplistic', TInv_multiplier = 5)
    print('test 5, rank = {}'.format(rsR1))