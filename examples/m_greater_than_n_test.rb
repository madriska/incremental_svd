$LOAD_PATH.unshift File.join(File.dirname(__FILE__), %w[.. lib])
require 'incremental_svd'

# A test similar to simple_test.rb with m > n, to exercise manual QR unpacking.

m = GSL::Matrix[[1, 2],
                [4, 5],
                [7, 8],
                [10, 11],
                [13, 14]]

#col = GSL::Vector::Col[3, 6, 9, 12, 15]
cols = GSL::Matrix[[3, 1],
                   [6, 2],
                   [9, 3],
                   [12, 4],
                   [15, 5]]

# Compute the SVD of the original matrix m
u, v, s = m.SV_decomp

rank = 3

5.times do
  # Add the column to the SVD
  u, v, s = GSL::Linalg.svd_add_column(u, v, s, cols.column(0))

  u = u.submatrix(0, 0, u.size1, rank) if u.size2 > rank
  v = v.submatrix(0, 0, v.size1, rank) if v.size2 > rank
  s = s.subvector(0, rank) if s.size > rank

  # The SVD, when re-composed, should now recreate [m, col].
  m2 = u * s.to_m_diagonal * v.transpose
  puts m2.inspect
end

# GSL::Matrix
# [  1.000e+00  2.000e+00  3.000e+00 
#    4.000e+00  5.000e+00  6.000e+00 
#    7.000e+00  8.000e+00  9.000e+00 
#    1.000e+01  1.100e+01  1.200e+01 
#    1.300e+01  1.400e+01  1.500e+01 ]

