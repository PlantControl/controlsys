package controlsys

import "plantcontrol.org/v1/gonum/mat"

// stiffChainERank is rank(E) as level 0 of deflateStiffChain decides it
// (descriptorRank, MATLAB rank, stiff values nonzero); 0 when the SVD
// fails.
func stiffChainERank(E *mat.Dense) int {
	n, _ := E.Dims()
	var svd mat.SVD
	if n == 0 || !svd.Factorize(E, mat.SVDNone) {
		return 0
	}
	sv := svd.Values(nil)
	r, _, _ := descriptorRank(sv, sv[0], n, true)
	return r
}
