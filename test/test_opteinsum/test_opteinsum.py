
import torch
import numpy as np
import opt_einsum as oe


#dim = 30
#I = np.random.rand(dim, dim, dim, dim)
#C = np.random.rand(dim, dim)

#%timeit optimized(I, C)
#10 loops, best of 3: 65.8 ms per loop

#%timeit contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)


#path_info = oe.contract_path('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
#######




I = 20
J = 7
A = np.random.rand(I, J)
B = np.random.rand(I)
path, path_info = oe.contract_path('ij, i -> i', A, B, optimize='optimize')

print(path_info)
print(path)



#I = 20
#J = 300
#A = torch.rand(I, J)
#path, path_info = oe.contract_path('ij -> ij', A, optimize='optimize')
#
#print(path_info)


#I = 20
#A = torch.rand(I)
#path, path_info = oe.contract_path('i, i ->', A, A, optimize='optimize')
#
#print(path_info)
#


#
#
#
#einsum_string = 'bdik,acaj,ikab,ajac,ikbd->'
#
#unique_inds = set(einsum_string) - {',', '-', '>'}
#index_size = [10, 17, 9, 10, 13, 16, 15, 14, 12]
#sizes_dict = dict(zip(unique_inds, index_size))
#views = oe.helpers.build_views(einsum_string, sizes_dict)
#
#path, path_info = oe.contract_path(einsum_string, *views)
#
#




