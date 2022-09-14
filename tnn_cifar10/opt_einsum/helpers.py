"""
Contains helper functions for opt_einsum testing scripts
"""

from collections import OrderedDict

import numpy as np

from .parser import get_symbol

__all__ = ["build_views", "compute_size_by_dict", "find_contraction", "flop_count"]

_valid_chars = "abcdefghijklmopqABC"
_sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4])
_default_dim_dict = {c: s for c, s in zip(_valid_chars, _sizes)}


def build_views(string, dimension_dict=None):
    """
    Builds random numpy arrays for testing.

    Parameters
    ----------
    string : list of str
        List of tensor strings to build
    dimension_dict : dictionary
        Dictionary of index _sizes

    Returns
    -------
    ret : list of np.ndarry's
        The resulting views.

    Examples
    --------
    >>> view = build_views(['abbc'], {'a': 2, 'b':3, 'c':5})
    >>> view[0].shape
    (2, 3, 3, 5)

    """

    if dimension_dict is None:
        dimension_dict = _default_dim_dict

    views = []
    terms = string.split('->')[0].split(',')
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        views.append(np.random.rand(*dims))
    return views


def compute_size_by_dict(indices, idx_dict, conv_subscripts="",intprods={},intermediate="", new_conv=0):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index _sizes
    conv_subcripts: list of indices which will be convolved.
    output: whether is an output term
    intprods: dictionary
        Dictionary of tensors involved in inputs. 
    intermediate: iterable
        The list of tensors involved in this particular intermediate/product.
    new_conv: Boolean
        Is a new convolution occuring?
    
    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:
            if i in conv_subscripts:
                if len(indices)==1: #This means we are evaluating a conv idx in the output. 
                    ret *= idx_dict[i]["max"]
                    continue
                poslist=[convdim for convdim in idx_dict[i]]
                intfactors=intprods[intermediate] #The tensors involved in this intermediate.
                convfactors=[factor for factor in intfactors if factor in poslist] #Check intermediates with
                #if indices=={'i','j','k','m'}:
                #    import pdb; pdb.set_trace()
                #convolving leg
                if len(convfactors)==1: 
                    ret *=idx_dict[i][convfactors[0]] #Only one convolving leg, so use its true size.
                elif len(convfactors)>1: #Check if this is a brand new convolution.                 
                    #if (intprods[convfactors[0]]|intprods[convfactors[1]])==indices:
                    if new_conv:
                        ret *=idx_dict[i]["max"]**2
                    else:
                        ret *=idx_dict[i]["max"] #A padding has already occurred; use max.           
                #else:
                    #ret*= idx_dict[i]["max"] #A padding has already occurred, so just take the max. 
            else:
                ret *= idx_dict[i]
    return ret

def find_contraction(positions, input_sets, output_set,intprods={},conv_subscripts={}):
    """
    Finds the contraction and convolution for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    conv_subscripts : set
        The list of convolution subscripts.


    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction
    new_convolution: boolean
        Whether a new convolution is occurring. 

    Examples
    --------

    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set('ab'), set('bc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
    """

    remaining = list(input_sets)
    new_conv = 0 #Boolean flag indicating whether a new convolution is occuring. 
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    composition=[] #The tensors which make up an intermediate product.
    idx_contract = set.union(*inputs)
    for i in positions:
        intermediate=sorted("".join(set(input_sets[i]))) #Coerce to set and then alphabetical string.
        intermediate="".join(intermediate)
        composition+=intprods[intermediate]
    
    if conv_subscripts:
        for j in conv_subscripts:
            if j in (input_sets[positions[0]]) and j in (input_sets[positions[1]]):
                new_conv = 1 #A new convolution is about to occur.
    
    idx_remain = output_set.union(*remaining)

    new_result = idx_remain & idx_contract
    new_result_str = "".join(sorted(sorted("".join(new_result)))) #Annoying manipulation from set to sorted string.
    intprods[new_result_str]=list(set(composition))
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)

    return new_result, remaining, idx_removed, idx_contract, new_conv

"""
def flop_count(idx_contraction, inner, num_terms, size_dictionary):
    
    Computes the number of FLOPS in the contraction or convolution.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    90

    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    270

    

    overall_size = compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor
"""
#tr: Fixing the flop_count. 
def flop_count(idx_contraction, inner, num_terms, size_dictionary,conv_subscripts="",intprods={},intermediate="",input_sets=[],
               output_set=[], new_conv=0):
    """
    Computes the number of FLOPS in the contraction or convolution.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction
    conv_subscripts: iterable
        The indices which will be convolved.
    intprods: dictionary
        A dictionary of which intermediate product each sequence of indices corresponds to. 
    intermediate: iterable
        The list of tensors involved in this particular product/intermediate.
    input_sets: iterable
        The list of all input sets. Only used if we need to compute a left-to-right cost.
    out_set: iterable
        The intended output.
    new_conv: Boolean
        Is a new convolution occurring
    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    90

    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    270
    """
    #If the intermediate is empty, we need to calculate the full product from left to right. 
    if intermediate=="":
        cost=1
        cost_list=[]
        len_inputs = len(input_sets) #Original length of inputs. 
        
        #Complete the first left-to-right-product.
        contract_tuple = find_contraction((0,1), input_sets, output_set,intprods,conv_subscripts)
        out_inds, input_sets, idx_removed, idx_contract, new_sub_conv = contract_tuple
        out_inds_format=sorted("".join(out_inds))
        out_inds_format="".join(out_inds_format)
        
        cost = flop_count(idx_contract, idx_removed, 2, size_dictionary, conv_subscripts,intprods,out_inds_format, "","",new_sub_conv)
        cost_list.append(cost)
        
        for j in range(1,len_inputs):
            if len(input_sets)==1: break #Nothing left to contract. 
            #Contract from left-to-right. The next intermediate product is pushed to the end of input_sets.
            contract_tuple = find_contraction((0,len(input_sets)-1), input_sets, output_set,intprods,conv_subscripts)
            out_inds, input_sets, idx_removed, idx_contract, new_sub_conv = contract_tuple
            
            out_inds_format=sorted("".join(out_inds))
            out_inds_format="".join(out_inds_format)
            
            cost = flop_count(idx_contract, idx_removed, 2, size_dictionary, conv_subscripts,intprods,
                              out_inds_format, "","",new_sub_conv)
            cost_list.append(cost)
            
        return sum(cost_list)
                                                                       
    overall_size = compute_size_by_dict(idx_contraction, size_dictionary, conv_subscripts, intprods,intermediate,new_conv)
    op_factor = max(1, num_terms - 1)
    #if inner:
    #    op_factor += 1

    return overall_size * op_factor

def rand_equation(n, reg, n_out=0, d_min=2, d_max=9, seed=None, global_dim=False, return_size_dict=False):
    """Generate a random contraction and shapes.

    Parameters
    ----------
    n : int
        Number of array arguments.
    reg : int
        'Regularity' of the contraction graph. This essentially determines how
        many indices each tensor shares with others on average.
    n_out : int, optional
        Number of output indices (i.e. the number of non-contracted indices).
        Defaults to 0, i.e., a contraction resulting in a scalar.
    d_min : int, optional
        Minimum dimension size.
    d_max : int, optional
        Maximum dimension size.
    seed: int, optional
        If not None, seed numpy's random generator with this.
    global_dim : bool, optional
        Add a global, 'broadcast', dimension to every operand.
    return_size_dict : bool, optional
        Return the mapping of indices to sizes.

    Returns
    -------
    eq : str
        The equation string.
    shapes : list[tuple[int]]
        The array shapes.
    size_dict : dict[str, int]
        The dict of index sizes, only returned if ``return_size_dict=True``.

    Examples
    --------
    >>> eq, shapes = rand_equation(n=10, reg=4, n_out=5, seed=42)
    >>> eq
    'oyeqn,tmaq,skpo,vg,hxui,n,fwxmr,hitplcj,kudlgfv,rywjsb->cebda'

    >>> shapes
    [(9, 5, 4, 5, 4),
     (4, 4, 8, 5),
     (9, 4, 6, 9),
     (6, 6),
     (6, 9, 7, 8),
     (4,),
     (9, 3, 9, 4, 9),
     (6, 8, 4, 6, 8, 6, 3),
     (4, 7, 8, 8, 6, 9, 6),
     (9, 5, 3, 3, 9, 5)]
    """

    if seed is not None:
        np.random.seed(seed)

    # total number of indices
    num_inds = n * reg // 2 + n_out
    inputs = ["" for _ in range(n)]
    output = []

    size_dict = OrderedDict((get_symbol(i), np.random.randint(d_min, d_max + 1)) for i in range(num_inds))

    # generate a list of indices to place either once or twice
    def gen():
        for i, ix in enumerate(size_dict):
            # generate an outer index
            if i < n_out:
                output.append(ix)
                yield ix
            # generate a bond
            else:
                yield ix
                yield ix

    # add the indices randomly to the inputs
    for i, ix in enumerate(np.random.permutation(list(gen()))):
        # make sure all inputs have at least one index
        if i < n:
            inputs[i] += ix
        else:
            # don't add any traces on same op
            where = np.random.randint(0, n)
            while ix in inputs[where]:
                where = np.random.randint(0, n)

            inputs[where] += ix

    # possibly add the same global dim to every arg
    if global_dim:
        gdim = get_symbol(num_inds)
        size_dict[gdim] = np.random.randint(d_min, d_max + 1)
        for i in range(n):
            inputs[i] += gdim
        output += gdim

    # randomly transpose the output indices and form equation
    output = "".join(np.random.permutation(output))
    eq = "{}->{}".format(",".join(inputs), output)

    # make the shapes
    shapes = [tuple(size_dict[ix] for ix in op) for op in inputs]

    ret = (eq, shapes)

    if return_size_dict:
        ret += (size_dict, )

    return ret
