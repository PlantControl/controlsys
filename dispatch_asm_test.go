//go:build amd64 && !noasm && !gccgo && !safe

package controlsys

// lastDenseGrid is the longest grid that useDenseSweep keeps dense for a
// coupled model with n states and m inputs, by {n, m}.
var lastDenseGrid = map[[2]int]int{
	{16, 1}: 5,
	{24, 1}: 3,
	{44, 4}: 3,
	{80, 4}: 2,
}
