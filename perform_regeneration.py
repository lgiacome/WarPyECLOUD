import numpy as np
from numpy.random import rand
from warp import echarge as charge
from warp import emass as mass
from warp import top
from warp_parallel import parallelsum


def perform_regeneration(target_n_mp, wsp, sec):
    # Old number of MPs
    n_mp = wsp.getn()
    # Compute old total charge and Ekin
    chrg = np.sum(wsp.getw())
    erg = np.sum(0.5 / np.abs(charge / mass) * wsp.getw() * (wsp.getvx() * wsp.getvx() +
                                                             wsp.getvy() * wsp.getvy() +
                                                             wsp.getvz() * wsp.getvz()))
    # Update the reference MP size
    new_nel_mp_ref = chrg / target_n_mp
    sec.set_nel_mp_ref(new_nel_mp_ref)

    print('New nel_mp_ref = %d' % sec.pyecloud_nel_mp_ref)
    print('Start SOFT regeneration. n_mp=%d Nel_tot=%1.2e, En_tot = %1.2e' % (n_mp, chrg, erg))
                
    # Compute the death probability         
    death_prob = float(n_mp - target_n_mp) / float(n_mp)
    # Decide which MPs to keep/discard
    n_mp_local = top.pgroup.nps[wsp.getjs()]
    flag_keep = (rand(n_mp_local) > death_prob)
    
    # Compute the indices of the particles to be kept/cleared       
    i_init = top.pgroup.ins[wsp.getjs()]
    inds_clear = np.where(not flag_keep)[0] + i_init - 1
    inds_keep = np.where(not flag_keep)[0] + i_init - 1
    # Compute the charge after the regeneration in the whole domain (also the other processes)
    if np.sum(inds_keep) > 0:
        chrg_after = parallelsum(np.sum(top.pgroup.pid[inds_keep, top.wpid-1]))
    else:
        chrg_after = parallelsum(0)
    # Resize the survivors
    correct_fact = chrg / chrg_after
    if np.sum(inds_keep) > 0:
        top.pgroup.pid[inds_keep, top.wpid-1] = top.pgroup.pid[inds_keep, top.wpid-1]*correct_fact
    # Flag the particles to be cleared
    if np.sum(inds_clear > 0):
        top.pgroup.gaminv[inds_clear] = 0.

    # Warning: top.clearpart is a Fortran routine so it has one-based ordering..
    top.clearpart(top.pgroup, wsp.getjs()+1, 1)        
    # Compute new number of MPs
    n_mp = wsp.getn()

    print('Applied correction factor = %e' % correct_fact)
    chrg = np.sum(wsp.getw())
    erg = np.sum(0.5 / np.abs(charge / mass) * wsp.getw() * (wsp.getvx() * wsp.getvx() +
                                                             wsp.getvy() * wsp.getvy() +
                                                             wsp.getvz() * wsp.getvz()))
    print('Done SOFT regeneration. n_mp=%d Nel_tot=%1.2e En_tot=%1.2e' % (n_mp, chrg, erg))
