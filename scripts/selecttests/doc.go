// Command selecttests maps a git diff of the controlsys root package to the
// tests and benchmarks that can observe it, and prints a tiered plan for
// go test. Any uncertainty yields tier full: the tool fails open.
//
// Tiers:
//
//	docs     comments or docs only: gofmt, vet, build
//	tests    test code only: those tests, gofmt, vet
//	library  library functions: affected tests native, amd64, race; affected benches
//	full     everything; see Plan.Reasons
//
// Call relation, computed separately on base and head, then unioned, in each of
// DefaultConfigs (every GOARCH asm/noasm by race/norace combination):
//
//   - Static calls.
//   - Interface calls: class hierarchy analysis over-approximated by method name
//     and signature, across every package including the standard library, gonum
//     and test types.
//   - Function values: a function is reached when code that mentions it as a
//     value is reached, called or not. Pure CHA (all functions of the signature)
//     connects every test to every test through testing.tRunner and selects 99%,
//     and RTA from main is wrong because tests are the roots. Mention-implies-reach
//     subsumes RTA run from each test separately, provided values cannot pass
//     between roots: a value can only be called after reached code created it,
//     or init did (init-reached changes are full). A package variable that can
//     hold a function or interface value and is written after init makes the
//     tree uncertain, since it could carry a value from one test to another.
//   - Package variables: a reader reaches every non-init library writer. A
//     write is a store or map update through the variable, a method call on
//     it, or handing a reference into it to any call. A changed test that may
//     write a package variable selects every root reaching a reader of it.
//   - reflect.Value.Call, Method, MethodByName, MakeFunc in the root package,
//     cgo, and go:linkname make the whole tree uncertain.
//
// Declarations without call edges:
//
//   - const: functions naming a constant whose type or value changed count as
//     changed; a constant named outside a function body is full.
//   - type: changed or removed existing type is full; new type is additive.
//   - methods: adding or removing a method on an existing type, or changing a
//     method's signature, is full, since it changes interface satisfaction
//     (type assertions, errors.As) without any call edge.
//   - var declared in a test file (a case table): selects every root reaching
//     a function that references it, when its initializer is self-contained
//     (no function literal, no method call, no name of a package function or
//     variable; calls into imported packages are allowed) and no
//     initialization code reads it. Otherwise full.
//   - other var, init, TestMain, blank var: any change is full.
//   - build-constrained, arch-suffixed, generated, asm files, go: directives,
//     blank or dot imports, go.mod, go.sum, testdata, CI, scripts, subdirectories,
//     unknown file kinds: full.
//   - test helpers reached from a root in another file: full. Helpers used only
//     by roots in their own file select those roots.
//
// Tests that read source (architecture_*_test.go, go/ast and friends, literals
// naming .go or doc files, directory walks, runtime.Caller, os/exec) are always
// selected, in every tier.
//
// Subtests run through their top-level name. Selection above MaxFraction of
// tests or a regexp above MaxRegexLen bytes (Linux caps one argument at 128 KiB)
// becomes full.
//
// Usage, from the repository root:
//
//	go -C scripts/selecttests run . -repo ../.. -base origin/main -head HEAD -format commands
//
// -head . reads the working tree. Exit status is 0 with a full plan on failure.
// The commands format starts with "# tier: <tier>" and "# reason:" comment
// lines. scripts/localci.sh runs the plan as the local CI gate and documents
// when a full run is still required.
package main
