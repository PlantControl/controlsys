package controlsys

// explicitTFVerified reports whether tf, the transfer function of the
// explicit model sys (route tfRouteExplicit) or of the E⁻¹ fold of the
// descriptor sys (tfRouteFold), is within terms of the stored data.
func explicitTFVerified(sys *System, route tfRoute, tf *TransferFunc, c *pencilCache) bool {
	if route == tfRouteFold {
		return descriptorTFVerifiedCached(sys, tf, c)
	}
	return descriptorTFVerifiedCached(identityPencil(sys), tf, nil)
}

// identityPencil is the explicit sys as the descriptor (A, B, C, D, I),
// delays dropped.
func identityPencil(sys *System) *System {
	n, _, _ := sys.Dims()
	return &System{A: sys.A, B: sys.B, C: sys.C, D: sys.D, E: eyeDense(n), Dt: sys.Dt}
}
