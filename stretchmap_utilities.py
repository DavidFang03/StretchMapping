import numpy as np

def integrate_profile(rhoprofile,rmin,r,dim=1,ngrid=8192):
    dr = (r - rmin)/ngrid
    mass = 0.
    r_prev = rmin
    f_prev = rhoprofile(r_prev)*dS(r_prev, dim)
    for i in range(1,ngrid):
        r_curr = rmin + i*dr
        f_curr = rhoprofile(r_curr)*dS(r_curr, dim)
        mass += 0.5*(f_prev + f_curr)*dr
        f_prev = f_curr
    return mass

def dS(r,dim=1):
    match dim:
        case 1:
            return 1
        case 2:
            return 2*np.pi*r
        case 3:
            return 4*np.pi*r**2