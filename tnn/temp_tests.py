import torch_conv_einsumfunc

### padding test
# Using this try to experimentally verify what the proper paddings are
#stride = 1
#dilation = 3
#for i in range(1, 12):
#    A = torch.rand(i, dtype=torch.double)
#    for j in range(1, 12):
#        B = torch.rand(j, dtype=torch.double)
#
#        last_pad = 60
#        for pad in range(0, last_pad):
#            #print(pad)
#            try:
#                AB = conv_einsum("i, i -> i | i", A, B, padding_mode='circular', padding=pad, \
#                                                        stride=stride, dilation=dilation) 
#                print(str(i) + ", " + str(j) + ": " + str(pad) + ", stride = " + str(stride) \
#                      + ", dilation = " + str(dilation))
#                break;
#            except RuntimeError:
#                if pad == last_pad-1:
#                    print(str(i) + ", " + str(j) + ": no pad")





#torch_A = torch.ones(2,2,2,2)
#torch_B = torch.ones(2,2)
#print(conv_einsum_pair("ijkl,ik->il|ik", torch_A, torch_B))
#print("\n\n\n")
#conv_einsum_pair("lijk,ik->il|ik", torch_A, torch_B)

#torch_A = torch.ones(2,2,2,2,2,2,2)
#torch_B = torch.ones(2,2,2,2,2,2)
#einsum_str = "abcdefh, dafgbh -> gfdaec | fdh"
#print("einsum_str = " + einsum_str)
#conv_einsum_pair(einsum_str, torch_A, torch_B)


#torch_A = torch.ones(2,3,2)
#torch_B = torch.ones(2,2,3,4)
#einsum_str = "ijk, ijmh -> ihj | j"
#print(einsum_str + " = \n" + str(conv_einsum_pair(einsum_str, torch_A, torch_B)))






#torch_A = torch.ones(2,3,2,2,2)
#torch_B = torch.ones(2,2)
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#einsum_str = "ifjkl,ik->lif|ik"
#print("einsum_str = " + einsum_str)
#out = conv_einsum(einsum_str, torch_A, torch_B, stride={"i":1, "k":1})
#print(out)
#print(out.size())


#I = 5
#J = 7
#K = 3
#L = 4
#M = 3
#N = 2
#torch_A = torch.ones(I, J, L, M, N)
#torch_B = torch.ones(I, J, L, M, N)
#einsum_str = "ijlmn, ijlmn -> lmn | lmn"
#print(einsum_str)
##print(str(conv_einsum(einsum_str, torch_A, torch_B, padding_mode='zeros', padding={"l":1, "m":1, "n":1}, {"l":1, "m":1, "n":1})))
#print(str(conv_einsum(einsum_str, torch_A, torch_B, {"l":2, "m":1, "n":1})))




#torch_A = torch.ones(4)
#torch_B = torch.ones(5)
#torch_C = torch.ones(6)
#einsum_str = "i,i,i -> i | i"
#out = conv_einsum(einsum_str, torch_A, torch_B, torch_C, padding_mode='max_zeros')
#print(einsum_str + " = \n" + str(out))

#torch_A = torch.ones(4)
#torch_B = torch.ones(5)
#torch_C = torch.ones(6)
#einsum_str = "i, i -> i | i"
#
#padding1 = max_zeros_padding_1d(torch_A.size(0), torch_B.size(0))
#out1 = conv_einsum(einsum_str, torch_A, torch_B, padding_mode='zeros', padding=padding1)
#
#padding2 = max_zeros_padding_1d(out1.size(0), torch_C.size(0))
#out2 = conv_einsum(einsum_str, out1, torch_C, padding_mode='zeros', padding=padding2)
#
#print(out1)
#print(out2)


#padding = max_zeros_padding_1d(10,10)
#print("padding = " + str(padding))




#for i in range(1, 10):
#    for j in range(1, 20):
#        
#        torch_A = torch.ones(i)
#        torch_B = torch.ones(j)
#        einsum_str = "i, i -> i | i"
#        dilation = 5
#        stride = 8
#        padding = max_zeros_padding_1d(torch_A.size(0), torch_B.size(0), dilation=dilation, \
#                                       stride=stride)
#        out = conv_einsum(einsum_str, torch_A, torch_B, padding=padding, padding_mode='zeros', \
#                          dilation=dilation, stride=stride)
#        expected_size = max(i*dilation, j)/stride # not sure about dilation factor
#        if out.size(0) == math.floor(expected_size):
#            print(str(i) + ", " + str(j) + " " + str(out) + str(out.size()) + " " + str(expected_size))

#ker = torch.rand(5, dtype=torch.double) 
#img = torch.rand(100, dtype=torch.double)
#max_mode_size = 105
#max_padding = max_zeros_padding_1d(ker.size(0), img.size(0), max_mode_size=max_mode_size)
#print("max_zeros_padding_1d = " + str(max_padding))
#outAB = conv_einsum_pair("i, i -> i | i", ker, img, max_mode_size=max_mode_size, padding_mode='zeros', padding=max_padding)
##outBA = conv_einsum_pair("i, i -> i | i", img, ker, max_mode_size=max_mode_size, padding_mode='zeros', padding=max_zeros_padding_1d(img.size(0), ker.size(0), max_mode_size=max_mode_size))
##print(outAB - outBA)
#print(outAB.size())
##print(outBA.size())
        

#### associativity test
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(50, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
##padding = 4 # circular for 5 x 5 -> 5
#padding = 50
##padding = 25
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#padding = 4 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)

        

#op12.size() = torch.Size([4, 3, 224, 224])
#self.layer_nodes['core'].size() = torch.Size([7, 3, 7, 7])
#einsum_str = "imkl, jnkl -> ijkl | kl"
#einsum_str = "jnkl, imkl -> ijkl | kl"
#print(einsum_str)
#A = torch.ones(4,3,224,224)
#B = torch.ones(7,3,7,7)
#out = conv_einsum(einsum_str, B, A, stride={"k":2, "l":2}, padding_mode='zeros', padding={"k":3, "l":3}) 
#print("out.size = " + str(out.size()))


#torch_A = torch.ones(20)
#torch_B = torch.ones(30)
#einsum_str = "i, i -> i | i"
#print(einsum_str + " = \n" + str(conv_einsum(einsum_str, torch_A, torch_B, stride=3)))


#I = 50 
#J = 10
#K = 15
#L = 20
#M = 25
## [30, 15, 30, 10, 25]
## [J,   K,  J,  L,  M]
#torch_A = torch.ones(L,M,K,I,J)
#torch_B = torch.ones(M,K,J,I)
#print("I = " + str(I) + " : J = " + str(J) + " : K = " + str(K) + " : L = " + str(L) + " : M = " + str(M))
#print("torch_A.size() = " + str(torch_A.size()))
#print("torch_B.size() = " + str(torch_B.size()))
#einsum_str = "lmkij, mkji -> lijkm | mij"
#print("einsum_str = " + einsum_str)
#out = conv_einsum(einsum_str, torch_A, torch_B, stride={"i":10, "j":5})
##out = conv_einsum(einsum_str, torch_A, torch_B)
#print(out.size())

#torch_A = torch.ones(4)
#torch_B = torch.ones(4,5)
#torch_C = torch.ones(4,5,6)
#einsum_str = "i,ij,ijk -> ij | ij"
#print(einsum_str + " = \n" + str(conv_einsum(einsum_str, torch_A, torch_B, torch_C)))



#K = 5
#I = 5
#kernel = torch.rand(1,1,1,K)
#input_tens = torch.rand(1,1,1,I)
#
#stride = 1
#dilation = 1
#padding = 11
#padding_mode = 'circular'
#
#batch_size = input_tens.size(0)
#kernel_size = kernel.size(3)
#input_size = input_tens.size(3) # image_size is perhaps a better name
#conv_len = input_size
#groups = kernel.size(0)
#in_channels = groups*kernel.size(2)
#out_channels = groups*kernel.size(1)
#
#m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, \
#                    kernel_size=kernel_size, \
#                    stride=stride, padding=padding, padding_mode=padding_mode, \
#                    dilation=dilation, \
#                    groups=groups)
#output = m(input_tens.reshape(1, 1, input_size))
#print("m = " + str(m))
#print("output.size() = " + str(output.size()))
##print("max_circular = " + str(max_circular_padding_1d(kernel_size, input_size)))
#print("padding = " + str(padding))




#
#
#
#####################################################################
#A = torch.rand(50, dtype=torch.double)
#B = torch.rand(50, dtype=torch.double)
#C = torch.rand(60, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 59 # circular
##padding = 25
#AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
##AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#padding = 59 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 49 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 49 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
#
#####################################################################
# 









        

######################################################################
#### associativity test
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(5, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 4 # circular
##padding = 25
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#
#padding = 4 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
######################################################################
#
#A = torch.ones(5, dtype=torch.double)
#B = torch.ones(8, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0) )
##max_mode_sizes = {"i": max_mode_size}
#max_mode_sizes = {"i": 12}
#
#padding = 4 
#AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes=max_mode_sizes, \
##AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='zeros', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
#print("AB = " + str(AB))
#
#
######################################################################
#### associativity test with full + circular convolutinos
### The idea is for the A*B to do a full convolution, and then finish the (AB)*C with a padded
### circular
### For the B*C do a padded circular, and then finish the A*(BC) with a padded circular
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(50, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 4 # circular
##padding = 25
#AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes={"i": 9}, \
##AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='zeros', padding=padding)
#print("AB.size() = " + str(AB.size()) + "\n\n")
## this is a full convolution since padding = ker_size - 1 and padding_mode='zeros',
## The output size should be o = i + k - 1 = 9
#
#padding = 8 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
##
##
##print("AB = " + str(AB))
#
#
##print("AB_nopad = " + str(AB_nopad))
#print(AB_C - A_BC)
######################################################################




##########
########## This test shows that the naive max padding binary circular convolution is not associative 
##########
#
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(9, dtype=torch.double)
#max_mode_size = max(A.size(0), B.size(0), C.size(0))
#max_mode_sizes = {"i": max_mode_size}
#
#padding = 4 
##padding = 25
##AB = conv_einsum_pair("i, i -> i | i", A, B, max_mode_sizes={"i": 9}, \
#AB = conv_einsum_pair("i, i -> i | i", A, B, \
#                                       padding_mode='circular', padding=padding)
#print("AB = " + str(AB))
#print("AB.size() = " + str(AB.size()) + "\n\n")
## this is a full convolution since padding = ker_size - 1 and padding_mode='zeros',
## The output size should be o = i + k - 1 = 9
#
#padding = 4 # circular 
##padding = 25 # zeros
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("AB_C.size() = " + str(AB_C.size()) + "\n\n")
#
#padding = 4 # circular
##padding = 2 # zeros
#BC = conv_einsum_pair("i, i -> i | i", B, C, max_mode_sizes=max_mode_sizes, \
#                      padding_mode='circular', padding=padding)
#print("BC.size() = " + str(BC.size()) + "\n\n")
#
#
#padding = 4 # circular
##padding = 2 # zeros
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, max_mode_sizes=max_mode_sizes, \
#                        padding_mode='circular', padding=padding)
#print("A_BC.size() = " + str(A_BC.size()) + "\n\n")
#
#
#print("AB_C = " + str(AB_C))
#print("A_BC = " + str(A_BC))
#



#### test case(2), 1
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(6, dtype=torch.double)
#
## since 6 < 5 + 5 - 1, case (2), so we pad the image tensor on the left to 6 with 0s and then 
## do the circular max padding with pytorch
#
#B_zero_pad = torch.cat([torch.zeros(1, dtype=torch.double), B]) 
#AB = conv_einsum_pair("i, i -> i | i", A, B_zero_pad, padding_mode='circular', padding=4)
#
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, padding_mode='circular', padding=5)
#
#BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=5)
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=5)
#
#print("A.size() = " + str(A.size()))
#print("B.size() = " + str(B.size()))
#print("C.size() = " + str(C.size()))
#print("AB.size() = " + str(AB.size()))
#print("BC.size() = " + str(BC.size()))
#print("AB_C = " + str(AB_C))
#print("A_BC = " + str(A_BC))


#### test case(2), 1b: checking if it doesn't matter if kernel or image is padded
####                   It does appear to matter
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(6, dtype=torch.double)
#
## since 6 < 5 + 5 - 1, case (2), so we pad the image tensor on the left to 6 with 0s and then 
## do the circular max padding with pytorch
#
#A_zero_pad = torch.cat([torch.zeros(1, dtype=torch.double), A]) 
#AB = conv_einsum_pair("i, i -> i | i", A_zero_pad, B, padding_mode='circular', padding=6)
#
#AB_C = conv_einsum_pair("i, i -> i | i", AB, C, padding_mode='circular', padding=5)
#
#BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=5)
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=5)
#
#print("A.size() = " + str(A.size()))
#print("B.size() = " + str(B.size()))
#print("C.size() = " + str(C.size()))
#print("AB.size() = " + str(AB.size()))
#print("BC.size() = " + str(BC.size()))
#print("AB_C = " + str(AB_C))
#print("A_BC = " + str(A_BC))


#### test case(2), 1
#A = torch.rand(5, dtype=torch.double)
#B = torch.rand(5, dtype=torch.double)
#C = torch.rand(9, dtype=torch.double)
#
#B_zero_pad = torch.cat([torch.zeros(4, dtype=torch.double), B]) 
#AB_circular = conv_einsum_pair("i, i -> i | i", A, B_zero_pad, padding_mode='circular', padding=8)
#AB_C_circular = conv_einsum_pair("i, i -> i | i", AB_circular, C, padding_mode='circular', padding=8)
#
#
#AB_full = conv_einsum_pair("i, i -> i | i", A, B, padding_mode='zeros', padding=4, max_mode_sizes={"i":9}) 
#AB_C_full = conv_einsum_pair("i, i -> i | i", AB_full, C, padding_mode='circular', padding=8)
#
#BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=8)
#A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=8)
#
#print("A.size() = " + str(A.size()))
#print("B.size() = " + str(B.size()))
#print("C.size() = " + str(C.size()))
#print("AB_full.size() = " + str(AB_full.size()))
#print("AB_full = " + str(AB_full))
#print("AB_circular.size() = " + str(AB_circular.size()))
#print("AB_circular = " + str(AB_circular))
#print("BC.size() = " + str(BC.size()))
#print("AB_C_circular = " + str(AB_C_circular))
#print("AB_C_full = " + str(AB_C_full))
#print("A_BC = " + str(A_BC))

####
# I think the cases for deciding which kind of circular convolution to do can be
# decided in the pairwise reduction 
# Basically we have the max_mode_sizes dictionary...

# let's test case(3), where O >= I + K - 1
# so here 20 >= 5 + 5 - 1
A = torch.rand(5, dtype=torch.double)
B = torch.rand(5, dtype=torch.double)
C = torch.rand(20, dtype=torch.double)

B_zero_pad = torch.cat([torch.zeros(15, dtype=torch.double), B]) 
AB_circular = conv_einsum_pair("i, i -> i | i", A, B_zero_pad, padding_mode='circular', padding=4)
AB_C_circular = conv_einsum_pair("i, i -> i | i", AB_circular, C, padding_mode='circular', padding=19)

B_zero_pad_right = torch.cat([B, torch.zeros(15, dtype=torch.double)]) 
AB_circular_right = conv_einsum_pair("i, i -> i | i", A, B_zero_pad_right, padding_mode='circular', padding=4)
AB_C_circular_right = conv_einsum_pair("i, i -> i | i", AB_circular_right, C, padding_mode='circular', padding=19)

B_zero_pad_mid = torch.cat([torch.zeros(7, dtype=torch.double), B, torch.zeros(8, dtype=torch.double)]) 
AB_circular_mid = conv_einsum_pair("i, i -> i | i", A, B_zero_pad_mid, padding_mode='circular', padding=4)
AB_C_circular_mid = conv_einsum_pair("i, i -> i | i", AB_circular_mid, C, padding_mode='circular', padding=19)



AB_full = conv_einsum_pair("i, i -> i | i", A, B, padding_mode='zeros', padding=4, max_mode_sizes={"i":9}) 
print("AB_full.size() = " + str(AB_full.size()))
AB_C_full = conv_einsum_pair("i, i -> i | i", AB_full, C, padding_mode='circular', padding=8)

BC = conv_einsum_pair("i, i -> i | i", B, C, padding_mode='circular', padding=4)
A_BC = conv_einsum_pair("i, i -> i | i", A, BC, padding_mode='circular', padding=19)

print("A.size() = " + str(A.size()))
print("B.size() = " + str(B.size()))
print("C.size() = " + str(C.size()))
print("AB_full.size() = " + str(AB_full.size()))
print("AB_full = " + str(AB_full))
print("AB_circular.size() = " + str(AB_circular.size()))
print("AB_circular = " + str(AB_circular))
print("AB_circular_right.size() = " + str(AB_circular_right.size()))
print("AB_circular_right = " + str(AB_circular_right))
print("AB_circular_mid.size() = " + str(AB_circular_mid.size()))
print("AB_circular_mid = " + str(AB_circular_mid))

print("BC.size() = " + str(BC.size()))
print("AB_C_circular = " + str(AB_C_circular))
print("AB_C_circular_right = " + str(AB_C_circular_right))
print("AB_C_circular_mid = " + str(AB_C_circular_mid))
print("AB_C_full = " + str(AB_C_full))
print("A_BC = " + str(A_BC))


