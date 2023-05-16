import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
from scipy.optimize import minimize
from astropy import constants as const

'''
stellar_model.py
Yukei S. Murakami, Johns Hopkins University May 2023

Stellar structure calculation for chemically homogeneous ZAMS star in its equilibrium state.

Contents:
    - energy transport: read in table, calculate opacity
        - calc_opacity
        - calc_grad_ad
        - calc_grad_rad
    - energy generation: pp-chain, CNO cycle, and combined
        - calc_eps_pp
        - calc_eps_CNO
        - calc_eps_total
    - equation of state: ideal gas + radiation pressure
        - calc_P_rad
        - calc_density: from P, T
        - calc_pressure: from density (inverse of calc_density)
        - calc_beta: ratio between P and P_gas
    - integration
        - shootf: shoot to a fit point
    - differential equations
        - derivs
    - boundary conditions
        - get_bc_i: boundary conditions near the center
        - get_bc_f: boundary conditions at surface

'''

# constants
sigma_sb = const.sigma_sb.cgs.value  # Stefan-Boltzmann constant
c = const.c.cgs.value # speed of light
G = const.G.cgs.value # gravitational constant
k_B = const.k_B.cgs.value  # Boltzmann constant in CGS units (erg/K)
m_p = const.m_p.cgs.value  # proton mass in CGS units (g)
a = 7.56577e-15 # erg/cm3/K4

# select Table 75
# X=0.7000
# Y=0.2600
# Z=0.0400

X = 0.35
Y = 0.55
Z = 0.1
# TABLE_ID = 75
TABLE_ID = 52


###################
#  opacity 
###################

# identify table in the raw ASCII file
# file downloaded from: https://opalopacity.llnl.gov/existing.html
with open('GN93hz') as f:
    for i, line in enumerate(f):
        if f'TABLE #{TABLE_ID:3.0f}' in line:
            if i < 240:
                table_info = line
                continue
            else:
                table_start_line = i
            
# get data & interpolate
df = pd.read_csv('GN93hz', delim_whitespace = True,
            skiprows = table_start_line + 3,
            nrows = 70).set_index('logT').T
logT_table = df.columns.astype(float).values
logR_table = df.index.astype(float).values
kappa = df.to_numpy()
interp = RegularGridInterpolator((logT_table, logR_table), kappa.T,
                                 bounds_error=False,
                                 fill_value=None)#np.nan)

def calc_opacity(logT,rho):
    logR = np.log10(rho / (1e-6*(10**logT))**3)
    log_opacity = interp(np.array([logT,logR]).T)
    opacity = 10**log_opacity
    return opacity

def calc_grad_ad(P,T):
    b = calc_beta(P,T)
    grad_ad = (1 + (1-b)*(4+b)/b**2)/(5/2 + 4*(1-b)*(4+b)/b**2)
    return grad_ad
    
def calc_grad_rad(p,t,m,l,opacity):
    grad_rad = (3 * opacity * l * p) / (16 * np.pi * a * c * G * m * t**4)
    return grad_rad

###################
#  energy generation
###################
def calc_eps_pp(T,rho,X1):
    Z1, Z2 = 1, 1 # p-p chain!
    zeta = 1 # "of order of unity" -- page 189
    T7 = 1e-7 * T
    T9 = 1e-9 * T
    
    # prepare factors (Eq. 18.56, 18.57)
    g11 = np.polyval([-0.0114,0.144,1.51,3.82,1],T9)
    psi = 1 
    f11 = np.exp(5.92e-3 * Z1 * Z2 * np.sqrt(zeta * rho / T7**3)) #
    
    # calculate energy in cgs
    eps_pp = 2.57e4 * psi * f11 * g11 * rho * X1**2 * \
             T9**(-2/3) * np.exp(-3.381 * T9**(-1/3))
    return eps_pp

def calc_eps_CNO(T,rho,X1,XCNO=0.01):
    T7 = 1e-7 * T
    T9 = 1e-9 * T
    
    # prepare factors
    g141 = np.polyval([-2.43,3.41,-2.00,1],T9)
    
    # calculate energy in cgs
    eps_CNO = 8.24e25 * g141 * XCNO * X1 * rho * T9**(-2/3) * \
              np.exp(-15.231 * T9**(-1/3) - (T9/0.8)**2)
    return eps_CNO

def calc_eps_total(T,rho,X1,XCNO):
    eps_pp = calc_eps_pp(T,rho,X1)
    eps_CNO = calc_eps_CNO(T,rho,X1,XCNO)
    return eps_pp + eps_CNO
    

###################
#  equation of state
###################
def calc_P_rad(T):
    return 4/3 * sigma_sb * T**4 / c

def calc_density(P, T, mu):        
    # Radiation pressure
    P_rad =  calc_P_rad(T)
    P_gas = np.asarray(P - P_rad)
   
    # Equation of state
    rho = np.asarray(P_gas  * (mu * m_p) / (k_B * T) )
    rho[P_gas<=0] = 0
    return rho

def calc_pressure(rho, T, mu):        
    # Radiation pressure
    P_rad =  calc_P_rad(T)
   
    # Equation of state
    P_gas =  rho / (mu * m_p) * (k_B * T) 

    return P_gas + P_rad

def calc_beta(P,T):
    P_rad = calc_P_rad(T)
    return 1 - (P_rad / P)

###################
#  shoot to a fit point
###################
def gen_logspaced_grid(start_val,final_val,n_x,log_scale=1000):
    ''' generate gridpoints between start_val and final_val 
    with logarithmically increasing 
    (decreasing if start_val < final_val) separation. 
    Values near start_val gets exponentially more gridpoints.
    '''
    scale = (final_val - start_val)
    spacing = (np.geomspace(1,log_scale,n_x)-1) / log_scale * scale
    gridpoints = start_val + spacing
    return gridpoints

def shoot_out(f,x_i,x_f,bc_i_guess,bc_f_guess,
           n_left=1000,n_right=1000, x_fit=None,return_y=False,
           grid_log_scale=1000):
    '''shoot to the surface'''
    # sanity check
    assert x_i < x_f, "boundary coordinates are invalid"
    
    # define/check midpoint coordinate
    if x_fit is None:
        x_fit = (x_i + x_f)/2
    else:
        assert x_i < x_fit and x_fit < x_f, "specified fit point is invalid"
    
    # define left- and right-side axes for integration
    # x_fit is only used to define the coordinate of the lowest density
    x_left = gen_logspaced_grid(x_i,x_fit,n_left,grid_log_scale)
    x_right = gen_logspaced_grid(x_f,x_fit,n_right,grid_log_scale)
    x = np.concatenate([x_left,np.flip(x_right)])
    
    # integrated values
    y_left = odeint(f,bc_i_guess,t=x)
    res = y_left[-1] - bc_f_guess
    
    if return_y:
        x_sol = np.concatenate([x_left,np.flip(x_right)])
        y_sol = y_left.T
        return res, x_sol, y_sol
    
    # return residual at the midpoint
    # note: the returned value is not an absolute value
    return res

def shootf(f,x_i,x_f,bc_i_guess,bc_f_guess,
           n_left=1000,n_right=1000, x_fit=None, return_y=False,
           grid_log_scale=1000):
    # sanity check
    assert x_i < x_f, "boundary coordinates are invalid"
    
    # define/check midpoint coordinate
    if x_fit is None:
        x_fit = (x_i + x_f)/2
    else:
        assert x_i < x_fit and x_fit < x_f, "specified fit point is invalid"
    
    # define left- and right-side axes for integration
    x_left = gen_logspaced_grid(x_i,x_fit,n_left,grid_log_scale)
    x_right = gen_logspaced_grid(x_f,x_fit,n_right,grid_log_scale)

    # integrated values
    y_left = odeint(f,bc_i_guess,t=x_left)
    y_right = odeint(f,bc_f_guess,t=x_right)
    res = y_left[-1] - y_right[-1]
    
    if return_y:
        x_sol = np.concatenate([x_left,np.flip(x_right)])
        y_sol = np.concatenate([y_left,np.flip(y_right)]).T
        return res, x_sol, y_sol
    
    # return residual at the midpoint
    # note: the returned value is not an absolute value
    return res

###################
#  differential equations
###################
def derivs(y,x,fixed_params):
    ''' calculate radial gradient in the mass-space.
    NOTE: all values are in cgs units
    '''
    l,p,r,t = y
    m = x
    mu,X,Y,Z = fixed_params    

    # calculate density: equation of state
    rho = calc_density(p,t,mu)
    
    # luminosity gradient: energy generation
    X1 = X
    XCNO = 0.7 * Z
    eps_pp = calc_eps_pp(t,rho,X1)
    eps_CNO = calc_eps_CNO(t,rho,X1,XCNO)
    l_prime = eps_pp + eps_CNO
    
    # pressure gradient: hydrostatic equilibrium
    p_prime = - G * m / (4 * np.pi * r**4)
    
    # radius gradient: spherical shell geometry
    r_prime = 1 / (4 * np.pi * r**2 * rho)

    # temperature gradient: energy transport  
    opacity = np.squeeze(calc_opacity(np.log10(t),rho))
    del_ad = calc_grad_ad(p,t)
    del_rad = calc_grad_rad(p,t,m,l,opacity)
    Del = np.nanmin([del_ad,del_rad])
    t_prime = - G * m * t / (4 * np.pi * r**4 * p) * Del
    
    
    # sanity check: derivative is zero if value is zero
    s = (l<=0) | (p<=0) | (r<=0) | (t<=0)
    l_prime = np.array(l_prime)
    p_prime = np.array(p_prime)
    r_prime = np.array(r_prime)
    t_prime = np.array(t_prime)
    l_prime[s] = 0
    p_prime[s] = 0
    r_prime[s] = 0
    t_prime[s] = 0
    
    return [l_prime,p_prime,r_prime,t_prime]

###################
#  boundary conditions
###################
def get_bc_i(free_params,M_tot,M_i,mu,X,Z):
    ''' calculate boundary condition at the center
    
    inputs:
        free_params (list): free parameters to be fitted
        M_tot: total mass of a star. A fixed value.
    
    
    '''
    P_c, T_c, R, L = free_params
    
    # get radius (assume constant density core)
    rho_c = calc_density(P_c,T_c,mu)
    R_c = (3 * M_i / (4 * np.pi * rho_c))**(1/3)
    
    # pressure
    P = P_c - (3 * G / 8 / np.pi) * \
        (4 * np.pi * rho_c / 3)**(4/3) * M_i**(2/3)
    
    # get central luminosity
    X1 = X
    XCNO = 0.7 * Z
    eps_total = calc_eps_total(T_c,rho_c,X1,XCNO)
    L_c = M_i * eps_total 
    
    # temperature -- assume opacity is approx. ~ k_c
    opacity = np.squeeze(calc_opacity(np.log10(T_c),rho_c))
    del_ad = calc_grad_ad(P_c,T_c)
    del_rad = calc_grad_rad(P_c,T_c,M_i,L_c,opacity)    
    if del_ad >= del_rad:
        T4 = T_c**4 - 1 / (2 * a * c) * \
            (3 / 4 / np.pi)**(2/3) * opacity * \
            eps_total * rho_c**(4/3) * M_i**(2/3)
        T = T4**(1/4)
    else: # del_ad <= del_rad:
        lnT = np.log(T_c) - (np.pi / 6)**(1/3) * G * \
              del_ad * rho_c**(4/3) * M_i**(2/3) / P_c
        T = np.exp(lnT)
    return [L_c, P, R_c, T]


### Eddington gray atmosphere method
def Eddington_deriv(P,tau,fixed_params):
    ''' Eddington gray atmosphere differential equation'''
    L,R,M,mu = fixed_params
    T = (3/4 * L / (4 * np.pi * sigma_sb * R**2) * (tau + 2/3))**(1/4)
    rho = calc_density(P,T,mu)[0]

    # rho=0 is returned when P_gas == 0 (i.e., P==P_rad)
    if rho <= 0:
        return np.nan#np.ones_like(rho)*0

    opacity = calc_opacity(np.log10(T),rho)[0]
    P_prime = G * M / R**2 / opacity
    return P_prime
    
def calc_P_surf(L,R,M,mu):
    fixed_params = [L,R,M,mu]

    # boundary condition
    # gas pressure is zero at the top of the atmosphere
    T_eff = (L / (4 * np.pi * R**2 * sigma_sb))**(1/4)
    P_boundary = calc_P_rad(T_eff) 

    # integrate
    P_arr = odeint(lambda P,tau: Eddington_deriv(P,tau,fixed_params),
                   y0 = P_boundary,  
                   t = np.linspace(0,2/3)) 
    P_surf = P_arr.max()
    return P_surf
    
### fitting density method 
# def calc_photosphere_pressure(rho,L,T_eff,M_surf,sg):
#     ''' calculate pressure at photosphere
#     inputs:
#         rho: density at photosphere
#         L: luminosity
#         T_eff: effective temperature calculated from L
#         M_surf: mass shell at surface
#         sg: surface gravity
#     '''
#     opacity = calc_opacity(np.log10(T_eff),rho)
#     P_photo = (2 / 3) * (sg / opacity) * \
#               (1 + (k_B * L) / (4 * np.pi * c * G * M_surf))
#     return P_photo
    


def get_bc_f(free_params,M_tot,mu):
    ''' calculate boundary condition at the surface 
     - reference: Stellar Structure and Evolution (2nd ed.) 11.2
     - define the surface at the photosphere where optical depth = 2/3
     
    inputs:
        free_params (list): free parameters to be fitted
        M_tot: total mass of a star. A fixed value.
    
    '''
    P_c, T_c, R, L = free_params
    sg = G * M_tot / (R**2) #surface gravity
    T_eff = (L / (4 * np.pi * R**2 * sigma_sb))**(1/4)
    
    # calculate surface pressure -- method 1
    # integrate diff.eq. 
    P_surf = calc_P_surf(L,R,M_tot,mu)
    
    # calculate surface pressure -- method 2
    # fitting density 
#     def rho_surf_helper(rho):
#         P_photo = calc_photosphere_pressure(rho,L,T_eff,M_tot,sg)
#         P_gas_rad = calc_pressure(rho,T_eff,mu)    
#         return np.abs(P_photo-P_gas_rad)    
    
# #     # optimize density
#     fit = minimize(rho_surf_helper,1e-8,
#                   method='Nelder-Mead',bounds=[(1e-13,1e-5)])
#     rho = fit.x[0]
#     P_surf = calc_pressure(rho,T_eff,mu)

    return [L, P_surf, R, T_eff]