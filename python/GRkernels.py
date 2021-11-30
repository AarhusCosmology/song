import numpy as np
from scipy.integrate import ode
from scipy.interpolate import UnivariateSpline
import scipy

Omega_m = 0.307
Omega_lambda = 1.0-Omega_m

a_ini = 1e-6
y0, t0 = [a_ini, 1, 3./7.*a_ini*a_ini, 6./7.*a_ini], a_ini

def g(a, y):
    D = y[0]
    D_prime = y[1]
    F = y[2]
    F_prime = y[3]

    aaHcHc = a*(Omega_lambda*a*a*a+Omega_m)
    return [D_prime, 
            1.0/aaHcHc*(3./2.*Omega_m*D/a-(3*Omega_lambda*a*a*a+3./2.*Omega_m)*D_prime),
           F_prime,
           1.0/aaHcHc*(3./2.*Omega_m*(D*D+F)/a-(3*Omega_lambda*a*a*a+3./2.*Omega_m)*F_prime)]

def solout(a,y):
    D = y[0]
    D_prime = y[1]
    F = y[2]
    F_prime = y[3]
    
    avec.append(a)
    fvec.append(a/D*D_prime)
    uvec.append(1.0/(1.+Omega_lambda*a*a*a/Omega_m))
    vvec.append(7./3.*F/D/D)
    wvec.append(7./6.*a*F_prime/D/D)
    return 0

r = ode(g)
r.set_integrator('dopri5',verbosity=7,first_step=1e-7,atol=1e-30,rtol=1e-8)
r.set_solout(solout)
r.set_initial_value(y0, t0)

avec = []
fvec = []
uvec = []
vvec = []
wvec = []

r.integrate(1.0)
fspl = UnivariateSpline(avec,fvec)
uspl = UnivariateSpline(avec,uvec)
vspl = UnivariateSpline(avec,vvec)
wspl = UnivariateSpline(avec,wvec)

def compute_cosk1k2(k1,k2,k3):
    cosk1k2 = -(k1*k1+k2*k2-k3*k3)/(2.*k1*k2)
    #mask = (k3 < 1e-3*k1)*(k3 < 1e-3*k2)
    #cosk1k2[mask] = -0.5*(k1[mask]/k2[mask]+k2[mask]/k1[mask])+k3[mask]*k3[mask]/(2.*k1[mask]*k2[mask])
    #cosk1k2 = -0.5*(k1/k2+k2/k1)+k3*k3/(2.*k1*k2)
    return cosk1k2

def kernel_FSZ(k1,k2,k3,Hc):
    #cosk1k2 = -(k1*k1+k2*k2-k3*k3)/(2.*k1*k2)
    cosk1k2 = compute_cosk1k2(k1,k2,k3)
    Hc_over_k3_sq = Hc*Hc/(k3*k3)
    Hc_over_k3_sqsq = Hc_over_k3_sq*Hc_over_k3_sq
    alpha = 2./7.+59./14.*Hc_over_k3_sq+45./2.*Hc_over_k3_sqsq
    beta = 1.-0.5*Hc_over_k3_sq+54*Hc_over_k3_sqsq
    gamma = -3./2.*Hc_over_k3_sq+9./2.*Hc_over_k3_sqsq
    
    source = beta-alpha+0.5*beta*cosk1k2*(k1/k2+k2/k1)+alpha*cosk1k2**2+gamma*(k1/k2-k2/k1)**2
    mask = (k1==k2)
    if mask.any():
        #The equation for source is numerically unstable for the case k1==k2.
        ratiosq = (k3[mask]/k1[mask])**2
        source[mask] = (0.5*beta[mask]-alpha[mask]+0.25*alpha[mask]*ratiosq)*ratiosq
    return (source/((1.+3*Hc**2/k1/k1)*(1.+3*Hc**2/k2/k2)))
    
def kernel_FSZ_mod(k1,k2,k3,Hc,logder):
    cosk1k2 = -(k1*k1+k2*k2-k3*k3)/(2.*k1*k2)
    Hc_over_k3_sq = Hc*Hc/(k3*k3)
    Hc_over_k3_sqsq = Hc_over_k3_sq*Hc_over_k3_sq
    alpha = 2./7.+59./14.*Hc_over_k3_sq+45./2.*Hc_over_k3_sqsq
    beta = 1.-0.5*Hc_over_k3_sq+54*Hc_over_k3_sqsq
    gamma = -3./2.*Hc_over_k3_sq+9./2.*Hc_over_k3_sqsq
    
    #Assuming logder is the logarithmic derivative of the newtonian delta_cdm:
    #gamma = gamma - 5.0/4.0*(3.*Hc_over_k3_sqsq+Hc_over_k3_sq)*(-2.0/(1.+3*Hc_over_k3_sq)+logder)
    gamma = gamma - 5.0/4.0*(3.*Hc_over_k3_sqsq+Hc_over_k3_sq)*logder
    
    source = beta-alpha+0.5*beta*cosk1k2*(k1/k2+k2/k1)+alpha*cosk1k2**2+gamma*(k1/k2-k2/k1)**2
    mask = (k1==k2)
    if mask.any():
        #The equation for source is numerically unstable for the case k1==k2.
        ratiosq = (k3[mask]/k1[mask])**2
        source[mask] = (0.5*beta[mask]-alpha[mask]+0.25*alpha[mask]*ratiosq)*ratiosq
    return (source/((1.+3*Hc**2/k1/k1)*(1.+3*Hc**2/k2/k2)))

def kernel_RV(k1,k2,k3,Hc,a):
    #cosk1k2 = -(k1*k1+k2*k2-k3*k3)/(2.*k1*k2)
    cosk1k2 = compute_cosk1k2(k1,k2,k3)
    Hc_over_k3_sq = Hc*Hc/(k3*k3)
    Hc_over_k3_sqsq = Hc_over_k3_sq*Hc_over_k3_sq
    f = fspl(a)
    u = uspl(a)
    v = vspl(a)
    w = wspl(a)
    fsq = f*f
    
    alpha = 0.5*((1.-3./7.*v)+(8*f+3*u-18./7.*w)*Hc_over_k3_sq+(36*fsq+(18*fsq-9*f)*u)*Hc_over_k3_sqsq)
    beta = 1.+(-2*fsq+6*f-9./2.*u)*Hc_over_k3_sq+(36*fsq+18*fsq*u)*Hc_over_k3_sqsq
    gamma = 0.5*((-fsq+f-3*u)*Hc_over_k3_sq+(9*fsq+4.5*(fsq-f)*u)*Hc_over_k3_sqsq)
    
    source = beta-alpha+0.5*beta*cosk1k2*(k1/k2+k2/k1)+alpha*cosk1k2**2+gamma*(k1/k2-k2/k1)**2
    mask = (k1==k2)
    if mask.any():
        #The equation for source is numerically unstable for the case k1==k2.
        ratiosq = (k3[mask]/k1[mask])**2
        source[mask] = (0.5*beta[mask]-alpha[mask]+0.25*alpha[mask]*ratiosq)*ratiosq
    return (source/((1.+3*f*Hc**2/k1/k1)*(1.+3*f*Hc**2/k2/k2)))

def kernel_RV_mod(k1,k2,k3,Hc,a,logder):
    cosk1k2 = -(k1*k1+k2*k2-k3*k3)/(2.*k1*k2)
    Hc_over_k3_sq = Hc*Hc/(k3*k3)
    Hc_over_k3_sqsq = Hc_over_k3_sq*Hc_over_k3_sq
    f = fspl(a)
    u = uspl(a)
    v = vspl(a)
    w = wspl(a)
    fsq = f*f
    
    alpha = 0.5*((1.-3./7.*v)+(8*f+3*u-18./7.*w)*Hc_over_k3_sq+(36*fsq+(18*fsq-9*f)*u)*Hc_over_k3_sqsq)
    beta = 1.+(-2*fsq+6*f-9./2.*u)*Hc_over_k3_sq+(36*fsq+18*fsq*u)*Hc_over_k3_sqsq
    gamma = 0.5*((-fsq+f-3*u)*Hc_over_k3_sq+(9*fsq+4.5*(fsq-f)*u)*Hc_over_k3_sqsq)
    
    #Assuming logder is the logarithmic derivative of the newtonian delta_cdm:
    #gamma = gamma - 1.0/2.0*(1+3*u/(2*f))*(3.*f*Hc_over_k3_sqsq+Hc_over_k3_sq)*(-2.0/(1.+3*f*Hc_over_k3_sq)+logder)
    gamma = gamma - f*1.0/2.0*(1+3*u/(2*f))*(3.*f*Hc_over_k3_sqsq+Hc_over_k3_sq)*logder
    
    source = beta-alpha+0.5*beta*cosk1k2*(k1/k2+k2/k1)+alpha*cosk1k2**2+gamma*(k1/k2-k2/k1)**2
    if isinstance(k1, np.ndarray) or isinstance(k2, np.ndarray):
        mask = (k1==k2)
        if mask.any():
            #The equation for source is numerically unstable for the case k1==k2.
            ratiosq = (k3[mask]/k1[mask])**2
            source[mask] = (0.5*beta[mask]-alpha[mask]+0.25*alpha[mask]*ratiosq)*ratiosq
    else:
        if k1==k2:
            ratiosq = (k3/k1)**2
            source = (0.5*beta-alpha+0.25*alpha*ratiosq)*ratiosq
    return (source/((1.+3*f*Hc**2/k1/k1)*(1.+3*f*Hc**2/k2/k2)))

def kernel_N(k1,k2,k3,a):
    #cosk1k2 = -(k1*k1+k2*k2-k3*k3)/(2.*k1*k2)
    cosk1k2 = compute_cosk1k2(k1,k2,k3)
    v = vspl(a)
    
    alpha = 0.5*(1.-3./7.*v)
    beta = 1.0
    
    source = beta-alpha+0.5*beta*cosk1k2*(k1/k2+k2/k1)+alpha*cosk1k2**2

    
    mask = (k1==k2)
    if mask.any():
        #The equation for source is numerically unstable for the case k1==k2.
        ratiosq = (k3[mask]/k1[mask])**2
        source[mask] = (0.5*beta-alpha+0.25*alpha*ratiosq)*ratiosq

    mask = (k1==k3)
    if mask.any():
        ratiosq = (k2[mask]/k1[mask])**2
        source[mask] = 0.25*(3.0-4.0*alpha+(alpha-1.0)*ratiosq)

    return source

def kernel_fNL(k1,k2,k3,Hc,a):
    Hc_over_k3_sq = Hc*Hc/(k3*k3)
    Hc_over_k3_sqsq = Hc_over_k3_sq*Hc_over_k3_sq
    f = fspl(a)
    u = uspl(a)
    
    gamma = 0.3*(2*f+3*u)*(Hc_over_k3_sq+3*f*Hc_over_k3_sqsq)
    source = gamma*(k1/k2-k2/k1)**2
    return (source/((1.+3*f*Hc**2/k1/k1)*(1.+3*f*Hc**2/k2/k2)))
