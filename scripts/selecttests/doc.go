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
// DefaultConfigs:
//
//   - Static calls.
//   - Interface calls: class hierarchy analysis over-approximated by method name
//     and signature, across every package including the standard library, gonum
//     and test types.
//   - Function values: a function is reached when code that mentions it as a
//     value is reached, called or not. Pure CHA (all functions of the signature)
//     connects every test to every test through testing.tRunner and selects 99%,
//     and RTA from main is wrong because tests are the roots. Mention-implies-reach
//     subsumes RTA run from each test separately: a value can only be called
//     after reached code created it, or init did (init-reached changes are full),
//     or another test stored it (next item).
//   - Package variables: a reader reaches every non-init library writer. A test
//     that writes a package variable is full when it changes.
//   - reflect.Value.Call, Method, MethodByName, MakeFunc in the root package,
//     cgo, and go:linkname make the whole tree uncertain.
//
// Declarations without call edges:
//
//   - const: functions naming a constant whose type or value changed count as
//     changed; a constant named outside a function body is full.
//   - type: changed or removed existing type is full; new type is additive.
//   - methods: adding or removing a method on an existing type is full, since it
//     changes interface satisfaction without any call edge.
//   - var, init, TestMain, blank var: any change is full.
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
package main
