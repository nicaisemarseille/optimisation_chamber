import numpy as np
from scipy.optimize import minimize

# Parametres initiaux
A = 1  
B = 1  
L = 3 # Longueur
l = 2 # Longueur du mur source
c0 = 340
gammap=1.4
rho0=1.2
eta0 = 1.0  
xi0 = 1/c0**2
w = 100
k0=2*np.pi/l 

materiaux={'isorel': {'phi': 0.7, 'sigma': 143000, 'alphah': 1.15},
           'ITFH' : {'phi': 0.94, 'sigma': 9067, 'alphah': 1},
           'melamine' : {'phi': 0.99, 'sigma': 14000, 'alphah': 1.02},
           'B5' : {'phi': 0.2, 'sigma': 2124000, 'alphah': 1.22},
           'laine de verre' : {'phi': 0.95, 'sigma': 1e4, 'alphah': 3},
           'bois massif' : {'phi': 0.2, 'sigma': 1700, 'alphah': 1},
           'double vitrage' : {'phi': 0.01, 'sigma': 1e5, 'alphah': 1},
           'fibres de carbone' : {'phi': 0.70, 'sigma': 5000, 'alphah': 1.5},
           'mousse d\'aluminium' : {'phi': 0.9, 'sigma': 1e4, 'alphah': 2.5}
           }

def materiau(name):
    return materiaux[name]['phi'], materiaux[name]['sigma'], materiaux[name]['alphah']

phi,sigma,alphah=materiau('mousse d\'aluminium')
phi,sigma,alphah=materiau('laine de verre')
phi,sigma,alphah=materiau('fibres de carbone')
xi1 = phi*gammap/c0**2 
a=sigma*phi**2*gammap/(c0**2*rho0*alphah)
eta1 = phi/alphah

def c(n,w):
    k=n*np.pi/l
    return 1 if k==k0 else 0

def coeffs(w,N):
    return {n:c(n,w) for n in range(-N,N+1)}

def lambda_0(k,w):
    if k**2 >= (xi0 / eta0) * w**2:
        return np.sqrt(k**2 - (xi0 / eta0) * w**2)
    else:
        return 1j * np.sqrt((xi0 / eta0) * w**2 - k**2)

def lambda_1(k,w):
    term1 = np.sqrt(0.5 * (k**2 - (xi1 / eta1) * w**2) + np.sqrt((k**2 - (xi1 / eta1) * w**2)**2 + (a * w / eta1)**2))
    term2 = 1j * np.sqrt(0.5 * ((xi1 / eta1) * w**2 - k**2) + np.sqrt((k**2 - (xi1 / eta1) * w**2)**2 + (a * w / eta1)**2))
    return term1 - term2

def f(x,k,w):
    return (lambda_0(k,w) * eta0 - x) * np.exp(-lambda_0(k,w) * L) + (lambda_0(k,w) * eta0 + x) * np.exp(lambda_0(k,w) * L)

def chi(k, alpha, n, w):
    gk=G[n]
    ans=(lambda_0(k,w) * eta0 - lambda_1(k,w) * eta1) / f(lambda_1(k,w)*eta1,k,w) - (lambda_0(k,w) * eta0 - alpha) / f(alpha,k,w)
    return gk*ans

def gamma(k, alpha, n, w):
    gk=G[n]
    ans=(lambda_0(k,w) * eta0 + lambda_1(k,w) * eta1) / f(lambda_1(k,w)*eta1,k,w) - (lambda_0(k,w) * eta0 + alpha) / f(alpha,k,w)
    return gk*ans

# Fonction pour calculer e_n
def e_n(n, alpha, w):    
    k=n*np.pi/L
    chi_val = chi(k, alpha, n, w)
    gamma_val = gamma(k, alpha, n, w)
    if k**2 >= (xi0 / eta0) * w**2:
        term1 = (A + B * np.abs(k)**2) * (1/(2*lambda_0(k,w))) * (np.abs(chi_val)**2 * (1 - np.exp(-2*lambda_0(k,w)*L)) + np.abs(gamma_val)**2 * (np.exp(2*lambda_0(k,w)*L) - 1))
        term2 = 2 * L * np.real(chi_val * np.conj(gamma_val))
        term3 = (B * lambda_0(k,w) / 2) * (np.abs(chi_val)**2 * (1 - np.exp(-2*lambda_0(k,w)*L)) + np.abs(gamma_val)**2 * (np.exp(2*lambda_0(k,w)*L) - 1))
        term4 = -2 * B * lambda_0(k,w)**2 * L * np.real(chi_val * np.conj(gamma_val))
        return term1 + term2 + term3 + term4
    else:
        term1 = (A + B * np.abs(k)**2)*(L*(np.abs(chi_val)**2 + np.abs(gamma_val)**2) + 1j/lambda_0(k,w) * np.imag(chi_val * np.conj(gamma_val)*(1-np.exp(-2*lambda_0(k,w)*L))))
        term2 = B*L*np.abs(lambda_0(k,w))**2 * (np.abs(chi_val)**2 + np.abs(gamma_val)**2)
        term3 = 1j*B*lambda_0(k,w)*np.imag(chi_val*np.conj(gamma_val)*(1-np.exp(-2*lambda_0(k,w)*L)))
        return term1 + term2 + term3

# Fonction principale pour la somme des e_n
def e(alphar,alphai, w, nmax):
    alpha=alphar+1j*alphai
    n_values = np.arange(-nmax, nmax+1)  # Discrétisation des valeurs de n
    return np.real(np.sum([e_n(n, alpha, w) for n in n_values]))

# Valeur initiale de alpha
initial_alpha = [1.0, -1.0]  
nmax=100

def objective(alpha):
    alphar, alphai = alpha
    return e(alphar, alphai, w, nmax)

# Choix de la fréquence
w=340/2**0.5

# Choix du materiau
mat='mousse d\'aluminium'
#mat='laine de verre'
#mat='fibres de carbone'

# Minimisation de la fonction objective
phi,sigma,alphah=materiau(mat)
xi1 = phi*gammap/c0**2 
a=sigma*phi**2*gammap/(c0**2*rho0*alphah)
eta1 = phi/alphah
G=coeffs(w,nmax)

def alpha(w,nmax):
    result = minimize(objective, initial_alpha, method='Nelder-Mead')
    return result.x

result = alpha(w,nmax)

# Affichage d'un exempple de résultat
print("Optimal alpha:", result)
