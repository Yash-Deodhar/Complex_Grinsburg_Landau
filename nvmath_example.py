import cupy as cp
import nvmath as nvmath


## Calculate Laplacian ##

def Laplacian(A):
    spectral_A = nvmath.fft.fft(A, axes=[0, 1])
    spectral_A *= factor
    return nvmath.fft.ifft(spectral_A, axes=[0, 1])

## Define Constants ##

N = 128
N_2 = cp.array(N*N)
c1 = 1.5
c3 = 0.25
M = 5000 # Total time steps
total_time = 500 # Total time in seconds
dt = cp.array(total_time/M)
k = cp.array([(1/4096)*(1 + c1*1j), (1/4096)*(1 - c3*1j)])
dt1 = cp.array(dt/4)
dt2 = cp.array(dt/3)
dt3 = cp.array(dt/2)

## Calculate mutliplication factors for Derivatives ##

i = cp.arange(N).reshape(N, 1)
j = cp.arange(N).reshape(1, N)
factor = cp.where(i <= N // 2,
                  cp.where(j <= N // 2, -(i**2 + j**2), -(i**2 + (N - j)**2)),
                  cp.where(j <= N // 2, -((N - i)**2 + j**2), -((N - i)**2 + (N - j)**2)))

## Initialize arrays ##

A, B = 3*cp.random.rand(N,N) - 1.5, 3*cp.random.rand(N,N) - 1.5
A = cp.array(A + B*1.j)
A1 = cp.zeros_like(A)
A.tofile("output.bin")

## Start Runge Kutta Loop ##
for i in range(M):

  abs_A = cp.abs(A)

  ## RK Step 1 ##
  LapA = Laplacian(A)
  A1 = A + dt1*(A + k[0]*LapA/(N_2) - abs_A*k[1]*A)

  ## RK Step 2 ##
  LapA = Laplacian(A1)
  A1 = A1 + dt2*(A1 + k[0]*LapA/(N_2) - abs_A*k[1]*A1)

  ## RK Step 3 ##
  LapA = Laplacian(A1)

  A1 = A1 + dt3*(A1 + k[0]*LapA/(N_2) - abs_A*k[1]*A1)

  ## RK Step 4 ##
  LapA = Laplacian(A1)
  A = A1 + dt*(A1 + k[0]*LapA/(N_2) - abs_A*k[1]*A1)

  with open("output.bin", "ab") as f:
    A.tofile(f)

  if((i+1)%(M/10) == 0):
    print(str(100*(i+1)/M) + '% completed\n')