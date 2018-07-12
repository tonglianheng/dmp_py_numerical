# Table of Contents

1.  [Python is Slow](#orgbc21f4f)
2.  [ndarrays](#org4de47e6)
3.  [Multiple Dimensions](#org1ae0eff)
4.  [Data Type](#orgca19b81)
5.  [Slice and Dice](#org092aa58)
6.  [Operations on `ndarray`](#orgb7f25bb)
7.  [Universal Functions](#orgf2235bf)
8.  [Broadcasting Python Functions](#orge3a054c)
    1.  [`numpy.vectorize()` Function](#orgc3924c9)
    2.  [Performance Considerations](#org841b77f)
9.  [Check Point](#org773adfa)
10. [Reduce](#orgcadd57b)
11. [Multi-Dimensional Reduce](#org9138925)
12. [Accumulate](#org9172cce)
13. [Summary Statistics](#org104387e)



<a id="orgbc21f4f"></a>

# Python is Slow

Python lists are very flexible and versatile. A list can contain any
types of objects, items can be added into a list, and items can be
removed. This flexibility, however, comes at a price: the Python
lists are not particularly suitable for storing and computing large
amount of numbers.

The reason for this is mainly due to how computer processors
work. Each element in a python list just like any other python
variable, only refers to the object it is currently associated
with. In other words, the python list elements only store the memory
addresses of the objects they represent. This offers great
flexibility and memory efficiency, but when the processor tries to
load the data assocaited with the list into its caches for
computation it cannot get all the values they need in one go.

Suppose we are summing up to every memeber of a list (`mylist`):

    mylist = [1,2,3,4,5,6,7,8,9]
    result = sum(mylist)

The CPU will try to load mylist from the main memory to a very fast
internal temporary storage area inside the CPU itself called the
cache. It will try to perform compute the final result from the data
in cache, and if the data in cache is not sufficient, then it will
then dump the cache and get more from the main memory. The
computational efficiency largely depends on how less often the cache
is refreshed and data reloaded from the main memory.

Now, a python list contains only references to the actual elements
stored in memory. Those elements can be at any parts of the main
memory. There is no requirement for them to be next to each
other. When the list is loaded into the cache, CPU will try to load
all data of the list into the cache, but during computation it
quickly realises that the list data do not contain the actual
elements, only the references. It needs to load the elements from
other locations. So the cache is dumped and a block of data
containing the first element is loaded.  But because the first
element and second element do not have to be near each other in the
main memory, the cache may have to be dumped again to load the
second element, and third and so on. This leads to very inefficient
use of cache and causes the computation to be slow.

The solution to this problem is to store the actual values of all
the elements of the list in one continuous block of memory, so that
the cache can obtain as many elements as possible in one load
operation. This leads to less frequent cache refreshes and thus more
efficient computation. The requirement for a different way of
storing the list is the main reason why `numpy` was introduced.

`numpy` arrays store their elements (not the references) directly in
one continuous block of memory. This is how arrays in C are
stored. However, because the array directly store the values and
array elements must be of the same size, hence elements must be of
the same type. And as the data is one continuous memory block,
unlike python lists, one cannot insert, add or remove elements from
the a `numpy` array easily. Using `numpy`, we trade convinience for
computational performance.

Since `numpy` arrays map directly to C arrays, they can utilise the
existing fast math libraries that are writen in C or
Fortran&#x2014;examples include open-source OpenBLAS, Intel's MKL, AMD's
ACML, Apple's Accelerate Framework etc. These libraries contain the
standard linear algebraic functions that are highly optimised to
your computer's CPU architechure, and try to squeeze out every ounce
of your CPU performance.

As a result, computationally `numpy` out performs the standard
Python by a long mile.

![img](./mat_mult_times.png "Comparison of computational performance of matrix multiplication using standard Python vs. `numpy`. Here the amount of time taken to perform a matrix multiplication is recorded for two sqaure matrices of increasing size (rank)")

![img](./mat_dot_times.png "Comparison of computational performance of dot product using standard Python vs `numpy`. Here the amount of time taken to perform a dot product is recorded for two vectors of increasing size (dimension)")


<a id="org4de47e6"></a>

# ndarrays

Centre to numerical python's computation ability is the `ndarray`
object. `ndarrays` are arrays of same typed values that are stored
continuously in memory. `ndarray` is defined in the `numpy` module
but is used in many other packages, from machine learning package
`sklearn` (scikit-learn), data-flow package `pandas` to plotting
package `matplotlib`. Therefore anyone wishing to use Python for
data science and analysis must understand and know how to use
`ndarray`.

To create an `ndarray` the easiest way is to convert from the
standard Python collections using `numpy`'s `array()` function:

    import numpy as np

    mylist = [1,2,3,4,5]
    myarray = np.array(mylist)
    print(type(myarray))
    print(myarray)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-lt7TW1", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

The biggest difference between a `ndarray` and python's standard
`list` object is that `ndarray` is guaranteed to store its values in
a continuous block of memory. `list` on the other hand is an array
of references, with the actual objects stored in other parts of
memory. This makes the `ndarrays` much more efficient for performing
computations&#x2014;makes it easier for processors to "anticipate" and
pre-load data into the CPU caches thus reducing the number of
loop-ups from the memory&#x2014;but also leads to two restrictions:

-   `ndarray` must contain values of the same type
-   `ndarray` cannot cannot change size: resizing would involve a new
    `ndarray` being created

    myarray = np.array([1, 2, '3'])
    print(myarray)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-njsIwo", line 1, in <module>
        myarray = np.array([1, 2, '3'])
    NameError: name 'np' is not defined

In the above example all elements are converted into strings, as the
`ndarray` cannot accept a mixture of types.

There are often the case that we want to store the result of our
computations in a collection for later use.  For Python's normal
`list` objects we would create an empty list first, and then append
the results to it as we compute them. This habbit needs to change
for `ndarrays`. Because `ndarrays` are fixed in size, it is often
more efficient to allocate an array of a desirable size beforehand
before we compute the value of each element.

For this we want to create a `ndarray` with all zeros, you can do
this easily using the `numpy` function `zeros()`:

    # 1D array of zeros
    initial_vec = np.zeros(10)
    print(initial_vec)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-VvyhjP", line 2, in <module>
        initial_vec = np.zeros(10)
    NameError: name 'np' is not defined

If we want a sequence of numbers, `arange()` function behaves much
like the builtin `range()` function but produces a `ndarray`
equivalent:

    sequence = np.arange(10)
    print(sequence)
    sequence = np.arange(10,20,2)
    print(sequence)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-XRB23l", line 1, in <module>
        sequence = np.arange(10)
    NameError: name 'np' is not defined


<a id="org1ae0eff"></a>

# Multiple Dimensions

`ndarrays` are named "nd" because it is designed for n-dimensions.

We cam create an array of any dimensions by supplying it with a list
of multiple dimensions:

    import numpy as np
    twodim = np.array(
      [[1, 2],
       [3, 4]]
    )
    threedim = np.array(
      [
        [[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]]
      ]
    )
    print(twodim)
    print('\n------------\n')
    print(threedim)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-PgLi8S", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

Arrays like any other python objects have useful attributes
associated to them. For example, the `.shape` parameter gives the
shape of an array:

    print(twodim.shape)
    print(threedim.shape)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-dyit6s", line 1, in <module>
        print(twodim.shape)
    NameError: name 'twodim' is not defined

The `.shape` parameter (note it is not a function) gives the number
of elements per dimension in a tuple.

We can also change the shape of an array by redefining the `.shape`
parameter:

    twodim.shape = (4,1)
    print(twodim)
    twodim.shape = (1,4)
    print(twodim)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-nm4ad8", line 1, in <module>
        twodim.shape = (4,1)
    NameError: name 'twodim' is not defined

When reshaping, the number of elements of the reshaped array must
match exactly that of the original array.  Therefore, something like
the following is illegal and would lead to an error:

    # leads to an error
    twodim.shape = (1,2)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-LB4YHO", line 2, in <module>
        twodim.shape = (1,2)
    NameError: name 'twodim' is not defined


<a id="orgca19b81"></a>

# Data Type

As `ndarray` objects are homogeneous, we often need a way to find
out and to define the type of the elements contained within it.

We can use the `.dtype` paramter to find out the type of values
currently in the array:

    import numpy as np
    array = np.array([1, 2])
    print(array.dtype)
    array = np.array([1.0, 2.0])
    print(array.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-T24b2x", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

What if we want to create an array of floats from the list of
integers? We can specify `dtype` as an optional parameter when we
create the `ndarray`:

    array = np.array([1, 2], dtype='float')
    print(array, array.dtype)
    array = np.array(['1', '2'], dtype='float')
    print(array, array.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-PXdUGm", line 1, in <module>
        array = np.array([1, 2], dtype='float')
    NameError: name 'np' is not defined

We can also convert an existing array from one type to another,
provided that the transformation makes sense. To do this we need to
use `.astype()` method:

    array = np.array([1, 2, 3])
    new = array.astype('float')
    print(new, new.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-ZbpCIc", line 1, in <module>
        array = np.array([1, 2, 3])
    NameError: name 'np' is not defined

The full list of allowed data types and their corresponding names is
given in the numpy manual:
<https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.dtypes.html#arrays-dtypes-constructing>

Note that when we want to switch the type of elements in a list, we
should always use `.astype()` method, and not by changing the
`.dtype` parameter directly.  We are allowed to change the value of
the `.dtype` parameter, but this does not change the data stored in
the array and only changes how the data should be interpreted by the
computer:

    array = np.array([1,2], dtype='int64')
    print(array, array.dtype)
    array.dtype = 'int32'
    print(array, array.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-7Krkl5", line 1, in <module>
        array = np.array([1,2], dtype='int64')
    NameError: name 'np' is not defined

In this example, the data stored in `array` are 64-bit binaries for
integers 1 and 2, but when we change `.dtype` to a 32 bit integer,
the stride for the array is halved and this leads to the array being
treated as 4 elements of 32 bit integers each&#x2014;as demonstrated by
the following diagram:

![img](./dtype_differences.png)

This will not happen when we use `.astype()`, as the data is also
changed correctly for the type conversion:

    array = np.array([1,2], dtype='int64')
    new = array.astype('int32')
    print(new, new.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-77npm4", line 1, in <module>
        array = np.array([1,2], dtype='int64')
    NameError: name 'np' is not defined


<a id="org092aa58"></a>

# Slice and Dice

Values in the arrays can be retrieved in much the same way as python lists:

    import numpy as np
    heights = np.array([175, 177, 172, 180])
    print(heights[0], heights[2], heights[-1])

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-fLCkl6", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

To get the heights of first three samples, we may use a slice:

    first_three = heights[0:3]
    print(first_three)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-7x6OCd", line 1, in <module>
        first_three = heights[0:3]
    NameError: name 'heights' is not defined

Just like `list`, the slice does not include the upper limit of the
slice.

We can also take a different stride. If we omit the upper
limit, the slice goes all the way to the end. If we omit the lower
limit, then the slice starts from the beginning:

    second_fourth = heights[1::2]
    print(second_fourth)
    first_two = heights[:2]
    print(first_two)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-EIydkf", line 1, in <module>
        second_fourth = heights[1::2]
    NameError: name 'heights' is not defined

Negative strides reverses the slice:

    reverse = heights[-1::-1]
    print(reverse)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-r7fItj", line 1, in <module>
        reverse = heights[-1::-1]
    NameError: name 'heights' is not defined

Slicing in multiple dimensions is simply a matter of taking the
relevent slices in each dimension:

    # no. daily messages sent between people
    alice, bob, charlie = 0, 1, 2
    messages = np.array(
      [[ 0,  3, 17],
       [12,  0, 11],
       [ 9,  5,  0]]
    )

To get how many messages did Alice send to Bob, we can use indices
separated by comma:

    n_sent = messages[alice, bob]
    print(f'Alice sent {n_sent} msgs to Bob')

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-hPDMvC", line 2
        print(f'Alice sent {n_sent} msgs to Bob')
                                               ^
    SyntaxError: invalid syntax

The standard Python's multi-dimensional list indexing syntax can
also be used. So:

    n_sent = messages[alice][bob]

will give the same result. However, this way is less efficient due
to the internal implementation of numpy, and is generally not
recommended.

To zoom in on how many messages did the first two people send to
each-other, we can take the 2-by-2 top-left block of `messages`:

    comm_first_two = messages[0:2, 0:2]
    print(comm_first_two)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-YwiiBY", line 1, in <module>
        comm_first_two = messages[0:2, 0:2]
    NameError: name 'messages' is not defined

This will show the number of messages sent between Alice and Bob.

A full explaination of the array indexing can be found on
<https://docs.scipy.org/doc/numpy-1.14.1/user/basics.indexing.html>


<a id="orgb7f25bb"></a>

# Operations on `ndarray`

In general, functions and operations designed to work with `ndarray`
objects will return a new `ndarray` as the result of computation.

And by default the actions (with a few exceptions) are
element-wise. These are often referred to as "broadcast" actions,
because the compulations are broadcasted to each element of an input
array and the results are collected in an output array.

    import numpy as np

    myarray = np.array([1,2,3,4,5,6])

    print(2 * myarray)
    # add 10 to each element
    print(myarray + 10)
    # square each element
    print(myarray ** 2)
    # multiply two arrays element by element
    print(myarray * myarray)
    # is each element less than 4?
    print(myarray < 4)
    # is each element even?
    print(myarray % 2 == 0)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-vbsjFp", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

Not all builtin python functions can work with `ndarray`, it depends
on whether `ndarray` has defined the corresponding behaviour for the
builtin function. And they may not act on the array element by
element. For example, the `len()` function still works as expected:
it will return the total number of elements in an array:

    print(len(myarray))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    NameError: name 'myarray' is not defined

However, the `round()` function, which takes a number and rounds it
to the nearst significant digit does not work on arrays:

    # works for numbers
    print(round(3.1415926, 2))
    # does not work for array
    print(round(np.array([1.23456, 2.34567]), 2))

    3.14
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-IXgfWU", line 4, in <module>
        print(round(np.array([1.23456, 2.34567]), 2))
    NameError: name 'np' is not defined

To round all the elements in an array we should instead use the
`.round()` method:

    # correct way to do round
    print(np.array([1.23456, 2.34567]).round(2))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-dbxykw", line 2, in <module>
        print(np.array([1.23456, 2.34567]).round(2))
    NameError: name 'np' is not defined


<a id="orgf2235bf"></a>

# Universal Functions

All of the `numpy` defined functions that works with `ndarray`
objects are of the type `numpy.ufunc`. They are referred to as
universal functions.

These functions are broadcasts, which take take one or more arrays,
act on them element by element and returns a new array.

In a lot of cases, a numpy `ufunc` also has an array method
counter-part. The array method simply refers back to the parent
`ufunc`. For example the `.round()` method we have seen earlier:

    import numpy as np
    weights = np.array([71.04, 58.58, 71.17])
    print(weights.round(0))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-KFdDe8", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

This is equivalent to calling the `numpy.round()` function directly:

    print(np.round(weights, 0))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    NameError: name 'np' is not defined

In most cases, `array.func(parameters)` is equivalent to
`np.func(array, parameters)`.


<a id="orge3a054c"></a>

# Broadcasting Python Functions

While `numpy` and other numerical packages for data analysis has
implemented many useful functions that works with `ndarray`, many
Python's native functions do not, and there are no equivalent
numerical replacements.

For example the `hex()` function, which gives a string of the
hexadecimal representation of any integer.

    number = 42
    print(hex(number))

    0x2a

while it works for single numbers, it does not work for a `ndarray`:

    import numpy as np
    numbers = np.array([42, 43, 44, 45])
    # this will fail
    print(hex(numbers))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-eCqplD", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

Also, sometimes while a native function works with an array, its
default behaviour is defined in a way that may be unsuitable for our
problem.

For example, the `len()` function, we already know that `len()`
acting on an array will produce the number of elements in an array.

    people = np.array(
        ['Alice', 'Bob', 'Charlie', 'Dave']
    )
    print(len(people))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-ZZ7trw", line 1, in <module>
        people = np.array(
    NameError: name 'np' is not defined

So `len()` tells us that there are 4 elements in `people`. But what
if we actually want to get the number of characters in everyone's
name? `len()` on each name will give us the answer. But `len()` is
not broadcasted to each element by default, and there is not a
pre-defined broadcasting version of `len()` in `numpy`.

We could use a loop. The arrays loop exactly the same as lists.  But
there is a nicer way.

The developers of `numpy` have provided us with a function that
allows us to convert any standard Python function into a `numpy`
universal broadcasting function.

    vlen = np.frompyfunc(len, 1, 1)
    results = vlen(people)
    print(results, results.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-WfM23q", line 1, in <module>
        vlen = np.frompyfunc(len, 1, 1)
    NameError: name 'np' is not defined

The `np.frompyfunc()` function takes three inputs, from left to
right:

1.  The original Python function you want to convert: e.g. `len`
2.  Number of inputs the Python function takes: in this case `len`
    has one input
3.  Length of output the Python function returns: in thie case `len`
    returns one number.

`vlen` becomes a universal function that would broadcast `len()` to
each element of a given array.

Note that the resulting function produced by `np.frompyfunc()` will
always return an array of the generic reference type `object`. If
this is undesirable, we can always convert using `.astype()`:

    results = results.astype(int)
    print(results, results.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-1fagno", line 1, in <module>
        results = results.astype(int)
    NameError: name 'results' is not defined


<a id="orgc3924c9"></a>

## `numpy.vectorize()` Function

There is another `numpy` function called `numpy.vectorize()` that
on the surface does the same as `numpy.frompyfunc()`.

    vlen = np.vectorize(len)
    print(vlen(people), results.dtype)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-cvIRBr", line 1, in <module>
        vlen = np.vectorize(len)
    NameError: name 'np' is not defined

The `vectorize()` function creates a new function that would
broadcast the original Python function to every element of an
array&#x2014;just like `frompyfunc()`, and it even handles the types
correctly.  Notice that this time the output is an array of
integers instead of generic references.

However, there is a subtle difference.  The `vectorize()` function
actually produces another type of functions: `vectorized`, and they
are *not* numpy universal functions (`ufunc`).  `vectorized`
functions have more limited functionalities than `ufunc`, and
cannot perform [reduce](#orgcadd57b) or [accumulate](#org9172cce).

`frompyfunc()` on the other hand, produces true `ufunc` functions,
and therefore its results would support the same thing all other
builtin `numpy` functions support.

If you want to create a broadcasting function that integrates
better with other `numpy` functionalities, then `frompyfunc()` is
probably the better choice.


<a id="org841b77f"></a>

## Performance Considerations

If performance is a big issue&#x2014;for example when your data set is
large&#x2014;then, use both `frompyfunc()` and `vectorize()` sparingly.

The `ufunc` functions that come with `numpy` are all written in C
and are highly optimised. On the other hand, the `ufunc` functions
produced on the fly by `frompyfunc()` are not optimised. So for
large data sets, you will see a performance difference.

If there are built-in `numpy` functions that does the same, or if a
alternative solution can be found using a combination of built-in
`numpy` functions, then use these instead.


<a id="org773adfa"></a>

# Check Point

-   Convert `hex()` into a broadcasting function.


<a id="orgcadd57b"></a>

# Reduce

While the default behaviour of all the universal functions are to
broadcast, there are situations we also want to consecutively
perform an operation to each members and summarise into a single a
scalar result. Summation is a typical example:

    import numpy as np
    numbers = np.arange(1,11)
    print(numbers)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-3jI7Fv", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

Sum on `numbers` is obtained when we add 1, 2, 3, &#x2026; together. More
precisely, summation is a process of repeately applying operator `+`
to a previous result with the next element:

    result1 = 1 + 2
    result2 = result1 + 3
    result3 = result2 + 4
    ...
    sum     = result8 + 10

We can ask any numpy universal broadcasting function to undertake
the same process:

    total = np.add.reduce(numbers)
    print(total)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-6SQdqC", line 1, in <module>
        total = np.add.reduce(numbers)
    NameError: name 'np' is not defined

`numpy.add()` function is the full name of the `+` operator for
`ndarrays`. Like all python objects, functions can also have their
own methods. The `.reduce()` method modifies the behaviour of the
broadcasting `add()` sucn that the following is performed instead:

    result1 = np.add(numbers[0], numbers[1])
    result2 = np.add(result1, numbers[2])
    result3 = np.add(result2, numbers[3])
    # etc...
    total   = np.add(result8, numbers[9])

So in the end, `total` becomes the sum of all elements in `numbers`.

Similary we can find the minimum in the array using `.reduce()`:

    minimum = np.minimum.reduce(numbers)
    print(minimum)

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-cwFTwI", line 1, in <module>
        minimum = np.minimum.reduce(numbers)
    NameError: name 'np' is not defined

`np.minimum()` is a broadcasting universal function that takes two
arrays, compares each pair of elements, and keeps the smallest one:

    a1 = np.array([2,1,4,1,6])
    a2 = np.array([1,3,1,5,1])
    print(np.minimum(a1, a2))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-ytrwJP", line 1, in <module>
        a1 = np.array([2,1,4,1,6])
    NameError: name 'np' is not defined

For sums, Python already has a `sum()` function:

    # native python sum already exist
    print(sum(numbers))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-KVdeD5", line 2, in <module>
        print(sum(numbers))
    NameError: name 'numbers' is not defined

Similarly for minimum:

    print(min(numbers))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    NameError: name 'numbers' is not defined

So you may wonder when is `.reduce()` useful in practice?  The
answer is that `.reduce()` is mainly used for reducing a paricular
dimension in multi-dimensional arrays.

If we look at our matrix of sent text example again.

    # texts sent in a day
    alice, bob, charlie = 0, 1, 2
    messages = np.array(
      [[ 0,  3, 17],    # sent by alice
       [12,  0, 11],    # sent by bob
       [ 9,  5,  0]]    # sent by charlie
    )

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-aLEgjr", line 3, in <module>
        messages = np.array(
    NameError: name 'np' is not defined

To work out how many texts did Alice send


<a id="org9138925"></a>

# Multi-Dimensional Reduce

Sometimes we may want to compute the sum, or product of an array or
a row of a matrix. This can be achieved through `reduce()`:

    import numpy as np

    myarray = np.array([1,2,3,4,5,6])

    print(np.add.reduce(myarray))
    print(np.multiply.reduce(myarray))
    print(np.maximum.reduce(myarray))

    mymat = np.matrix([[1,2,3], [4,5,6]])

    print(mymat)
    print(np.add.reduce(mymat, 0))
    print(np.add.reduce(mymat, 1))

    # the default dimension is 0
    print(np.add.reduce(mymat))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-MIw8mP", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

`reduce` is only supported for binary functions defined by `numpy`.


<a id="org9172cce"></a>

# Accumulate

If you want to compute a running total instead of sum, you can use
`accumulate()`:

    import numpy as np

    myarray = np.array([1,2,3,4,5,6])

    print(myarray)
    print(np.add.accumulate(myarray))
    print(np.multiply.accumulate(myarray))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-CHHtGi", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.


<a id="org104387e"></a>

# Summary Statistics

`numpy` also supplies a set of very useful statistical functions:

    import numpy as np

    mylist = np.array([0.94, 0.60, 0.74, 0.21, 0.50])

    # min and max
    print('min =', np.amin(mylist))
    print('max =', np.amax(mylist))

    # averages, sum
    print('sum =', np.sum(mylist))
    print('mean =', np.mean(mylist))
    print('median =', np.median(mylist))

    # variance and standard deviation
    print('var =', np.var(mylist))
    print('std =', np.std(mylist))

    # correlation coeffients between two arrays
    otherlist = np.array([0.14, 0.45, 0.57, 0.87, 0.29])
    print('corr =\n', np.corrcoef(mylist, otherlist))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-8TSABN", line 1, in <module>
        import numpy as np
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
      File "/Users/tonglh/homebrew/lib/python2.7/site-packages/numpy/core/__init__.py", line 24, in <module>
    ImportError:
    Importing the multiarray numpy extension module failed.  Most
    likely you are trying to import a failed build of numpy.
    If you're working with a numpy git repo, try `git clean -xdf` (removes all
    files not under version control).  Otherwise reinstall numpy.

If you have multiple dimensions, you can also choose which axis you
wish to perform the statistics on

    # build a 5 by 2 matrix using mylist and otherlist
    mat = np.matrix([mylist, otherlist])

    print('mat =\n', mat)
    print('mean over rows =\n', np.mean(mat, 0))
    print('mean over cols =\n', np.mean(mat, 1))
    print('corr coeff between rows in mat =\n',
          np.corrcoef(mat, rowvar=True))
    print('corr coeff between cols in mat =\n',
          np.corrcoef(mat, rowvar=False))

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/var/folders/2c/_t0h7x_529x6lj0q7xsvppph0000gn/T/babel-Cup132/python-yxr8ll", line 2, in <module>
        mat = np.matrix([mylist, otherlist])
    NameError: name 'np' is not defined
