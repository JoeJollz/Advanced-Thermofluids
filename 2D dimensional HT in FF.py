from numpy import *
# Examplesheetproblem.m
from matplotlib.pyplot import *
a=0.01
L=10
Nr=200
Nz=50
U=0.1
Tin=10
Tw=100
rho=1000
cp=4.18
cond=0.0006
#calculate the grid spacing
DZ=L / (Nz - 1)
DR=a / (Nr - 1)
#calculate the radial position of each grid point
r=linspace(0,a,Nr)

#declare a sparse matrix
A=zeros((Nr*Nz,Nr*Nz))

#the right hand side is zero (mostly)
b=zeros(Nz*Nr)

#loop over all the nodes and write out the coefficients for the linear
#equations in the the matrix A
for i in range(1,Nz - 1):
    for j in range(1,Nr - 1):
        k=i + Nz*j
        A[k,k]=-cond*2/ DZ ** 2 - cond / r[j] / DR*(r[j] + DR / 2) / DR - cond / r[j] / DR*(r[j] - DR / 2) / DR
        A[k,k + 1]=-rho*cp*U*(1 - (r[j] / a) ** 2) / 2 / DZ + cond / DZ ** 2
        A[k,k - 1]=+rho*cp*U*(1 - (r[j] / a) ** 2) / 2 / DZ + cond / DZ ** 2
        A[k,k + Nz]=cond / r[j] / DR*(r[j] + DR / 2) / DR
        A[k,k - Nz]=cond / r[j] / DR*(r[j] - DR / 2) / DR
   
#loop over the special boundary nodes
i=0
for j in range(0,Nr):
    k=i + Nz*j
    A[k,k]=1
    b[k]=Tin

j=Nr-1

for i in range(0,Nz):
    k=i + Nz*j
    A[k,k]=1
    b[k]=Tw

#set zero gradient at the centre line in radial direction
j=0
for i in range(0,Nz):
    k=i + Nz*j
    A[k,k]=1
    A[k,k + Nz]=- 1
    b[k]=0
   
i=Nz-1
for j in range(1,Nr):
    k=i + Nz*j
    A[k,k]=1
    A[k,k - 1]=- 1
    b[k]=0

    
 #solve
T=linalg.solve(A,b)

 #the reshape command below puts the matrix into a 2 D matrix
#the plot command plots the matrix as a contour plot
Ans=reshape(T,(Nr,Nz))

r=linspace(0,a,Nr)
T_=T.reshape((Nr,Nz))
z = linspace(0,L, Nz)
Z,R= meshgrid(z,r)
fig=contourf(Z,R,T_,100)
xlabel('Z')
ylabel('r')
colorbar()
show()

# calculate mean temperature - using a very pythonic loop
Tmean = zeros(Nz)
for i,columns in enumerate(T_.T):
    Tmean[i] = dot(columns[1:],r[1:]*pi*2*DR*U*(1-(r[1:]/a)**2))/(U/2*pi*a**2)
    
figure()
fig=plot(z,Tmean)
xlabel('Z')
ylabel('Mean Temperature difference from wall')
show()

#compute the heat transfer coefficeintTmean = zeros(Nz)
h = zeros(Nz)
q = zeros(Nz)
Nu = zeros(Nz)
for i,columns in enumerate(T_.T):
    q[i] = cond*(columns[Nr-1]-columns[Nr-2])/DR
    h[i] = q[i]/(Tw-Tmean[i])
    Nu[i] = 2*h[i]*a/cond 

figure()
fig=plot(z,Nu)
xlabel('Z')
ylabel('Nusselt Number')
ylim([0,10])
show()