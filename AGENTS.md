# Controlsys

Read `CONTEXT.md` for domain terms and modeling assumptions before design, doc, or behavior changes.

## gopls MCP

When `gopls` MCP tools are available (`gopls mcp -instructions` is authoritative; it sees saved files only):
start with `go_workspace`, run `go_symbol_references` before changing a symbol definition, and
`go_diagnostics` on edited Go files before tests (omit `files` for the workspace; no Markdown).
Run `go_vulncheck` if `go.mod` changes. If the tools are not exposed, say so and use the CLI.

## Checks

```bash
go fix ./...
go vet ./...
go test -v -count=1 ./...
```

## Conventions

- gonum LAPACK is **row-major** (`a[row*n+col]`), unlike Fortran.
- Test with non-symmetric A; diagonal A hides transposition bugs.
- No inline comments unless logic is counter-intuitive.

## Performance

- Compare revisions with `go run ./scripts/benchcmp -base <rev> -candidate .`
  (alternating rounds + `benchstat`; flags in its doc comment).
- Use `m.RawMatrix().Data` with stride-aware indexing, not `At`/`Set`; `copy()` raw slices for submatrices.
- Pre-allocate buffers outside loops; reuse LAPACK work arrays across calls.
