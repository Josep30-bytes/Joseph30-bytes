#---Importation des modules---
import os
import struct
import math



#---Classiques---

def random():
    # 8 octets = 64 bits, on décale de 11 bits pour garder 53 bits
    r64 = struct.unpack(">Q", os.urandom(8))[0]
    r53 = r64 >> 11                     # entier uniformément dans [0, 2^53 - 1]
    return r53 * (1.0 / (1 << 53))      # réel dans [0,1[

def randrange(n, a=0, step=1):
    if type(n)!=int or type(a)!=int or type(step)!=int :
        raise ValueError("Attetion entrées non entières")
    if step==1:
        if a>n:
            return(n+int((random()*100000)%(a-n)))
        else:
            return(int(random()*100000)%n)
    else:
        if a>n:
            return(n+(int(random()*100000)%((a-n)//step))*step)
        else:
            return((int(random()*100000)%(n//step))*step)

def randint(a,b):
    if type(a)!=int or type(b)!=int or a>b:
        raise ValueError("Attention entré")
    elif a==b:
        return a
    else:
        return a+int(random()*(b-a+1))
    
def getrandbits(k):
    if not type(k)==int:
        raise ValueError("Attention entrée non entière")
    if k==0:
        return 0
    else:
        r64 = struct.unpack(">Q", os.urandom(8))[0]
        r53 = r64 >> 11                     # entier uniformément dans [0, 2^53 - 1]
        return int(r53 * (1.0 / (1 << 53-k)))



#---Agi sur des listes---

def choice(L):
    if type(L)!=list:
        raise ValueError("Pas une liste")
    if len(L)==0:
        raise ValueError("Attention liste vide") 
    return(L[randrange(len(L))])

def shuffle(L):   #Fisher-Yates
    if type(L)!=list:
        raise ValueError("Pas une liste")
    if len(L)==0:
        raise ValueError("Attention liste vide")
    liste=L.copy()
    for i in range(len(L)-1,0,-1):
        j=randint(0,i)
        liste[j],liste[i]=liste[i],liste[j]
    return liste

def sample(L,n):
    if type(L)!=list or type(n)!=int:
        raise ValueError("Attention entrée")
    if len(L)==0:
        raise ValueError("Attention liste vide")
    return shuffle(L)[:n]



#---Loi de distribution---

def uniform(a,b):
    if a==b:
        return a
    if a>b:
        return b+(a-b)*random()
    else :
        return a+(b-a)*random()
    
def binomialvariate(n=1, p=0.5):
    if not type(n)==int and 0<p<1:
        raise ValueError("Attention aux entrées")
    return(sum(random() < p for _ in range(n)))

def triangular(low, high, mode=None):
    if high<=low:
        raise ValueError("high doit être > low")
    if mode is None:
        mode=(low+high)/2
    if not (low <= mode <= high):
        raise ValueError("mode doit être entre low et high")

    u=random()
    c=(mode-low)/(high-low)

    if u<c:
        return low + (u * (high - low) * (mode - low)) ** 0.5
    else:
        return high - ((1-u) * (high-low) * (high-mode)) ** 0.5
    
def expovariate(lmbda=1.0):
    if lmbda==0:
        raise ZeroDivisionError("Attention entrée nulle")
    return -math.log(1.0 - random()) / lmbda

def gammavariate(alpha, beta=1.0):
    if alpha <= 0.0 or beta <= 0.0:
        raise ZeroDivisionError("Alpha et Beta doivent etre positifs")
    if alpha < 1.0:   # Transformation de Johnk
        return gammavariate(alpha+1.0, beta) * (random() ** (1.0/alpha))

    # Marsaglia & Tsang (2000)
    d = alpha - 1.0/3.0
    c = 1.0/(9*d)**0.5
    while True:
        # Normal standard (Box-Muller)
        z = (-2.0*math.log(random()))**0.5 *math.cos(2*math.pi*random())
        v = (1 + c*z) ** 3
        if v > 0:
            u = random()
            if u < 1 - 0.0331 * (z**4) or math.log(u) < 0.5*z*z + d*(1-v+math.log(v)):
                return d * v / beta
            
def betavariate(alpha, beta):
    if alpha <= 0.0 or beta <= 0.0:
        raise ZeroDivisionError("Alpha and Beta doivent etre positifs")
    x = gammavariate(alpha, 1.0)
    y = gammavariate(beta, 1.0)
    return x / (x + y)

_gauss_next=None
def gauss(mu=0.0, sigma=1.0):
    global _gauss_next

    # Si on a déjà un échantillon en réserve, on le renvoie
    if _gauss_next is not None:
        z = _gauss_next
        _gauss_next = None
        return mu + sigma * z

    # Sinon, on en génère deux d'un coup
    while True:
        u1, u2 = random(), random()
        if u1 > 1e-12:  # éviter log(0)
            break
    r = (-2.0 * math.log(u1))**0.5
    z0 = r * math.cos(2 * math.pi * u2)
    z1 = r * math.sin(2 * math.pi * u2)

    # On garde z1 pour l'appel suivant
    _gauss_next = z1
    return mu + sigma * z0

def lognormvariate(mu, sigma):
    return math.exp(gauss(mu, sigma))

def normalvariate(mu=0.0, sigma=1.0):
    while True:
        u1 = random()
        u2 = 1.0 - random()   # évite log(0)
        z = 1.7155277699214135 * (u1 - 0.5) / u2
        x = z * z / 4.0
        if x <= -math.log(u2):   # test d’acceptation
            return mu + sigma * z

def paretovariate(alpha):
    if alpha<=0:
        raise ZeroDivisionError("Alpha doit être strictement positif")
    u = 1.0 - random()   # évite log(0) si U=1 exactement
    return u ** (-1.0 / alpha)

def weibullvariate(alpha, beta):
    if not (alpha>0 and beta>0):
        raise ZeroDivisionError("Alpha et Beta doivent etre positifs")
    u = 1.0 - random()   # évite log(0) si u=1 exactement
    return beta * (-math.log(u)) ** (1.0 / alpha)
