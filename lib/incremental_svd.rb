# Requires Ruby/GSL (http://rb-gsl.rubyforge.org/).
# NOTE: this is *not* ruby-gsl (http://ruby-gsl.sourceforge.net/).
require 'gsl'

class GSL::Matrix
  # Emulates MATLAB's qr(A,0)
  def economy_QR_decomp
    qr, tau = self.QR_decomp
    q, r = qr.unpack(tau)
    
    m, n = shape
    return [q, r] if m <= n
    [q.submatrix(0, 0, q.size1, n), r.submatrix(0, 0, n, n)]
  end

  # Emulates MATLAB's svd(A,0)
  def economy_SV_decomp
    u, v, s = self.SV_decomp

    m, n = shape
    return [u, v, s] if m <= n
    [u.submatrix(0, 0, u.size1, n), v.submatrix(0, 0, v.size1, n), s.submatrix(0, 0, n, n)]
  end
end

module GSL::Linalg
  
  # Matthew Brand's algorithm for incremental SVD update
  # Adapted from MATLAB algorithm svdUpdate.m (written by Nathan Faggian, 
  # nathanf@mail.csse.monash.edu.au)
  #
  # (uu,s,vv) form the old SVD: 
  #   uu * diag(s) ** vv'  == M
  # 
  # cc are the new column(s) to be appended. Returns new SVD (uuN,sN,vvN) s.t.
  #   uuN * diag(sN) * vvN' == [M cc]
  # 
  def self.svd_add_column(uu,vv,s,cc)
    c = cc.size2 # number of added columns
    r = s.size
    q = vv.size1 
    
    # compute the projection of C onto the orthogonal subspace U
    ll = uu.transpose * cc
    
    # compute the component of C orthogonal that is orthogonal to the subspace U
    hh = cc - (uu * ll)
    
    # compute an orthogonal basis of H and the projection of C onto the
    # subspace orthogonal to U
    jj, kk = hh.economy_QR_decomp
    
    # compute the center matrix Q
    # Q = [      diag(s),  L;
    #         zeros(c,r),  K   ];
    qq =     GSL::Matrix.diagonal(s).horzcat(ll).
      vertcat(GSL::Matrix.zeros(c,r).horzcat(kk))
    
    # compute the SVD of Q
    uuu, vvu, s = qq.economy_SV_decomp
    
    # compute the updated SVD of [M,C]
    orth = uu.col(0).transpose * uu.col(uu.size2-1)
    
    uu = uu.horzcat(jj) * uuu
    
    # V = [   V          , zeros(q, c); ...
    #             zeros(c, r), eye(c)     ] * Vu;
    vv =                           (vv.horzcat(GSL::Matrix.zeros(q,c)).
       vertcat(GSL::Matrix.zeros(c,r).horzcat(GSL::Matrix.eye(c)))) * vvu
    
    # compact the new SVD
    r = [uu.size1, vv.size2].min
    
    [uu.submatrix(0, 0, uu.size1, r),
      vv.submatrix(0, 0, vv.size1, r),
      s.subvector(r)]
  end

end

