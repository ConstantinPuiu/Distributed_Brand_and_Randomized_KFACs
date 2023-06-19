import torch

def perform_C_correction(Q, d, m_aa_or_gg, brand_corection_dim_frac):
    sampled_indices = torch.multinomial(torch.ones(Q.shape[1]) * 1.0/Q.shape[1],
                                        num_samples = round(brand_corection_dim_frac * Q.shape[1] + 1), replacement=False)
    Q_subsampled = Q[:,sampled_indices]
    #reduced_space_AA_for_evd = Q_subsampled.T @ m_aa_or_gg @ Q_subsampled
    #ipdb.set_trace(context = 7)
    D_subsampled, U_subsampled = torch.linalg.eigh(Q_subsampled.T @ m_aa_or_gg @ Q_subsampled,
                                                                    UPLO='L',  out=None) # EIGENVALUE ARE IN ASCENDING ORDER!
    # how far is U form identity tells us how far we are from having "correct" basis direction
    Q[:, sampled_indices] = Q_subsampled @ U_subsampled
    d[sampled_indices] = D_subsampled
    # now the eigevalues order may have changed so sort ASCENDINGLY again
    d, sort_idx = torch.sort(d, descending=False )
    Q = Q[:, sort_idx]
    return Q, d
                        
""" run as

for A modules:
if self.steps % (self.TCov * self.correction_multiplier_TCov * self.brand_update_multiplier_to_TCov) == 0:
    self.Q_a[module], self.d_a[module] = perform_C_correction(Q = self.Q_a[module], d = self.d_a[module], m_aa_or_gg = self.m_aa[module], 
                                                              brand_corection_dim_frac = self.brand_corection_dim_frac)
    
for G modules:
if self.steps % (self.TCov * self.correction_multiplier_TCov * self.brand_update_multiplier_to_TCov) == 0:
    self.Q_g[module], self.d_g[module] = perform_C_correction(Q = self.Q_g[module], d = self.d_g[module], m_aa_or_gg = self.m_gg[module], 
                                                              brand_corection_dim_frac = self.brand_corection_dim_frac)
"""