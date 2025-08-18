#---Moyenne---

def mean(data):
    n=len(data)
    if n==0:
        raise ValueError("Attention entrée vide")
    s=0
    for i in data:
        s+=i
    return s/n

def fmean(data,weights=None):
    n=len(data)
    if n==0:
        raise ValueError("Attention entrée vide")
    if weights==None:
        s=0
        for i in data:
            s+=i
        return s/n
    m=len(weights)
    if m==0:
        raise ValueError("Attention entrée vide")
    if n!=m:
        raise ValueError("Entrées avec pas la même dimension")    
    s=0
    for i in range(n):
        s+=data[i]*weights[i]
    return s

def geometric_mean(data):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste vide")
    p=1
    for i in data:
        if i<=0:
            raise ValueError("Attention valeur négative ou nulle dans la liste")
        p*=i
    return round((p**(1/n)),1)
    
def harmonic_mean(data, weights=None):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste en entré vide")
    if weights==None:
        s=0
        for i in data:
            if i==0:
                return 0
            s+= 1/i
        return round(n/s,1)
    m=len(weights)
    if n!=m:
        raise ValueError("Attention aux dimensions des entrées")
    s=0
    s_weights=0
    for i in range(n):
        d=data[i]
        w=weights[i]
        if d==0:
            return 0
        if w<0:
            raise ValueError("Attention poids négatif")
        s+= (1/d)*w
        s_weights+=w
    return round(s_weights/s,1)



#---Médiane---

def median(data):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste en entré vide")
    L=data.copy()
    L=sorted(L)
    mid = n // 2
    if n%2==0:
        return (data[mid-1]+data[mid])/2
    return data[mid]

def median_low(data):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste en entré vide")
    L=data.copy()
    L=sorted(L)
    mid = n // 2
    if n%2==0:
        return data[mid-1]
    return data[mid]

def median_high(data):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste en entré vide")
    L=data.copy()
    L=sorted(L)
    mid = n // 2
    return data[mid]

def median_grouped(data, interval=1.0):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste en entré vide")
    DATA=data.copy()
    DATA=sorted(DATA)
    m = data[n // 2]
    L = m - interval / 2
    f_m = DATA.count(m)
    CF = sum(1 for v in DATA if v < m)
    return L + (n/2-CF)/f_m*interval



#---Mode---

def mode(data):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste en entré vide")
    D={}
    for i in data:
        if not i in D:
            D[i]=1
        else:
            D[i]+=1
    m=max(D.values())
    for a,b in D.items():
        if b==m:
            return a
        
def multimode(data):
    n=len(data)
    if n==0:
        return list()
    D={}
    for i in data:
        if not i in D:
            D[i]=1
        else:
            D[i]+=1
    m=max(D.values())
    L=list()
    for a,b in D.items():
        if b==m:
            L.append(a)
    return L



#---Variance et écart-type---

def pvariance(data, mu=None):
    n=len(data)
    if n==0:
        raise ValueError("Attention liste en entré vide")
    if mu is None:
        mu=mean(data)
    return sum((x-mu)**2 for x in data) / n

def variance(data, xbar=None):
    n=len(data)
    if n<2:
        raise ValueError("Attention pas assez de donné")
    if xbar is None:
        xbar=mean(data)
    return sum((x-xbar)**2 for x in data) / (n-1)

def pstdev(data, mu=None):
    return (pvariance(data, mu))**0.5

def stdev(data, xbar=None):
    return (variance(data, xbar))**0.5



#---Entre deux entrées---

def covariance(x,y):
    n=len(x)
    m=len(y)
    if n<2 or m<2:
        raise ValueError("Attention il manque des données")
    if not n==m:
        raise ValueError("Attention aux dimensions")
    xbar=mean(x)
    ybar=mean(y)
    return sum((x[i]-xbar)*(y[i]-ybar) for i in range(n)) / (n-1)

def rankdata(data):
    """Retourne les rangs"""
    n=len(data)
    sorted_data=sorted((val, i) for i, val in enumerate(data))
    ranks=[0]*n
    i=0
    while i<n:
        j=i
        # Regroupe les égalités
        while j < n - 1 and sorted_data[j][0] == sorted_data[j + 1][0]:
            j += 1
        # Rang moyen pour les valeurs égales
        avg_rank = (i + j + 2) / 2.0  # +1 car rangs commencent à 1
        for k in range(i, j + 1):
            ranks[sorted_data[k][1]] = avg_rank
        i = j + 1
    return ranks

def correlation(x,y,method='linear'):
    if method=='linear':
        std_x=stdev(x)
        std_y=stdev(y)
        if std_x==0 or std_y==0:
            raise ZeroDivisionError("Attention une variable est constante")
        return covariance(x,y)/(std_x*std_y)
    elif method=='ranked':
        def rankdata(data):
            n=len(data)
            sorted_data=sorted((val, i) for i, val in enumerate(data))
            ranks=[0]*n
            i=0
            while i<n:
                j=i
                # Regroupe les égalités
                while j < n - 1 and sorted_data[j][0] == sorted_data[j + 1][0]:
                    j += 1
                # Rang moyen pour les valeurs égales
                avg_rank = (i + j + 2) / 2.0  # +1 car rangs commencent à 1
                for k in range(i, j + 1):
                    ranks[sorted_data[k][1]] = avg_rank
                i = j + 1
            return ranks
        ranks_x = rankdata(x)
        ranks_y = rankdata(y)
        return round(correlation(ranks_x,ranks_y),10)

def linear_regression(x,y,proportional=False):
    n=len(x)
    if n != len(y):
        raise ValueError("Attention aux dimensions")
    if n < 2:
        raise ValueError("Attention il manque des données")
    if not proportional:
        var_x=variance(x)
        if var_x==0:
            raise ZeroDivisionError("Attention x est constant")
        slope=covariance(x,y)/var_x
        intercept=mean(y)-slope*mean(x)
        return slope, intercept
    denom=sum(xi*xi for xi in x)
    if denom==0:
        raise ZeroDivisionError("Attention tout les x sont égaux à zéro")
    slope=sum(xi*yi for xi, yi in zip(x, y))/denom
    intercept=0
    return slope, intercept



#---Autre---

def quantiles(data, n=4, method='exclusive'):
    N = len(data)
    if N<2:
        raise ValueError("quantiles() requires at least two data points")
    if n<1 or type(n)!=int:
        raise ValueError("n doit être un entier non nul positif")
    L=data.copy()
    L=sorted(L)
    result=list()
    for i in range(1,n):
        p = i/n
        if method=='inclusive':
            r=p*(N-1)+1
        elif method=='exclusive':
            r=p*(N+1)
        else:
            raise ValueError("method doit être 'inclusive' ou 'exclusive'")
        k=int(r)  # partie entière
        d=r-k     # partie décimale
        if k<=0:
            q=data[0]
        elif k>=N:
            q=data[-1]
        else:
            q=data[k-1]+d*(data[k]-data[k-1])
        result.append(q)
    return result
