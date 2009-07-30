$LOAD_PATH.unshift File.join(File.dirname(__FILE__), %w[.. lib])
require 'incremental_svd'

# A test similar to simple_test.rb with m > n, to exercise manual QR unpacking.

m = GSL::Matrix[[1, 2],
                [4, 5],
                [7, 8],
                [10, 11],
                [13, 14]]

col = GSL::Vector::Col[3, 6, 9, 12, 15]

# Compute the SVD of the original matrix m
m_u, m_v, m_s = m.SV_decomp

# Add the column to the SVD; we don't need m anymore.
u, v, s = GSL::Linalg.svd_add_column(m_u, m_v, m_s, col)

# The SVD, when re-composed, should now recreate [m, col].
m2 = u * s.to_m_diagonal * v.transpose
puts m2.inspect

# GSL::Matrix
# [  1.000e+00  2.000e+00  3.000e+00 
#    4.000e+00  5.000e+00  6.000e+00 
#    7.000e+00  8.000e+00  9.000e+00 
#    1.000e+01  1.100e+01  1.200e+01 
#    1.300e+01  1.400e+01  1.500e+01 ]

