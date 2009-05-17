$LOAD_PATH.unshift File.join(File.dirname(__FILE__), %w[.. lib])
require 'incremental_svd'

m = GSL::Matrix[[1, 2],
                [4, 5]]

col = GSL::Vector::Col[3, 6]

# Compute the SVD of the original matrix m
m_u, m_v, m_s = m.SV_decomp

# Add the column to the SVD; we don't need m anymore.
u, v, s = GSL::Linalg.svd_add_column(m_u, m_v, m_s, col)

# The SVD, when re-composed, should now recreate [m, col].
m2 = u * s.to_m_diagonal * v.transpose
puts m2.inspect

# GSL::Matrix
# [  1.000e+00  2.000e+00  3.000e+00 
#    4.000e+00  5.000e+00  6.000e+00 ]

