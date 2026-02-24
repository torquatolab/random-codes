#!/usr/bin/env python

""" Simulation of 2D hyperplane intersection processes"""
"""
    The script simulates samples of an isotropic hyperplane
    intersection process (HIP) with unit number density in
    two dimensions. The observation window is [-L/2,L/2]^2.
"""
""" 
    If you publish an academic paper using this software,
    please cite the following paper:
    Klatt et al. Nature Communications 10, 811 (2019)
"""

__author__ = "Michael A. Klatt"
__copyright__ = "Copyright 2023, Michael A. Klatt"
__license__ = "GNU GPLv2"
__version__ = "1.0"
__maintainer__ = "Michael A. Klatt"
__email__ = "mklatt@princeton.edu"
__status__ = "Production"

import numpy as np
import math
import argparse

# Parse parameters
descrpt = "Simulation of 2D hyperplane intersection processes"
parser = argparse.ArgumentParser(description=descrpt)
parser.add_argument('-n', '--nsamples', metavar='int', type=int, default=10,
                    help='Number of samples')
parser.add_argument('-L', '--system-size', metavar='float', type=float, default=100,
                    help='Linear system size')
args = parser.parse_args()
L = args.system_size

# Simulate Hyperplanes Intersection Process
# in observation window W=(-m,m)^2 # where a = 1./sqrt(2)
m = L/2.
area = (m*2)**2
R = m*math.sqrt(2)

# Choose mean number of points in W
mean_number_of_points = L**2
# Resuling intensity of point process
intensity_of_pp = mean_number_of_points/area
assert(abs(intensity_of_pp - 1.0) < 1e-6)

# Compute intensity "gamma" of hyperplane process
def kappa(d):
    return math.pi**(0.5*d)/math.gamma(1+0.5*d)
gamma = (intensity_of_pp / (kappa(2-1)**2/(2**2*kappa(2)**(2-1))) )**(1./2)
mean_number_of_lines = 2*gamma*R

# Sample a realization of the Hyperplanes Intersection Process
def hyperplanes_intersection_process_2d(mean_nmbr_of_lines):
    xy = []
    number_of_lines = np.random.poisson(mean_nmbr_of_lines)

    d = np.random.uniform(-R,R,number_of_lines)
    theta = np.random.uniform(0,math.pi,number_of_lines)
    n_x = np.cos(theta)
    n_y = np.sin(theta)

    number_of_points = 0

    # iterate over all pairs of lines
    for l1 in range(number_of_lines):
        for l2 in range(l1+1,number_of_lines):
            # the point of itersection (x_i,y_i)
            # is the solution of the following matrix equation A*(x_i,y_i)=b:
            # (  n_x[l1] n_y[l1]  )   ( x_i )   ( d[l1] )
            # (                   ) * (     ) = (       )
            # (  n_x[l2] n_y[l2]  )   ( y_i )   ( d[l2] )
            A = np.empty((2,2))
            A[0,0] = n_x[l1]
            A[0,1] = n_y[l1]
            A[1,0] = n_x[l2]
            A[1,1] = n_y[l2]
            b = np.empty((2,1))
            b[0,0] = d[l1]
            b[1,0] = d[l2]

            # Find solution
            [xi, yi] = np.linalg.solve(A, b)
            # Check whether the solution lies within W
            # (i.e. whether |x_i| < m and |y_i| < m
            if np.absolute(xi) < m and np.absolute(yi) < m:
                # if yes, then we can increase the number of points
                xy += [[xi, yi]]
    
    return np.array(xy).reshape(-1,2)

for i in range(args.nsamples):
    coords = hyperplanes_intersection_process_2d(mean_number_of_lines)
    #np.savetxt(f"hip_2D__L_{L}__{i}.dat", coords)
    
    name=f"hip_2D__L_{L}__{i}.txt"
    f = open(name, "w")
    f.write("2\n")
    f.write("{0:0.4f}\t0.0\t0.0\t0.0\n".format(L))
    f.write("0.0\t{0:0.4f}\t0.0\t0.0\n".format(L))

    for n in coords :
        f.write("{0:0.10e}\t{1:0.10e}\t0.0\t0.0\n".format(*n))
    f.close()

    print("{0},{1}".format(max(coords[:,0]), max(coords[:,1])))
