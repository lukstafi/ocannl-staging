(* gh-ocannl-686: a routine whose name collides with a reserved word or a builtin.

   Routine names come from user-facing surfaces — an [Assignments.Block_comment] label, the [~name]
   argument of [Context.compile] and of the autotune drop-ins (gh-ocannl-669) — and used to reach
   the emitted [void <name>(] verbatim. [Block_comment ("asm", ...)] — a natural abbreviation, and
   the one that found this — therefore emitted [void asm(], which every C front end rejects, and the
   rejection surfaced as [Invalid_argument "... This is a bug in OCANNL. Please file an issue with
   the generated .c file ..."]: the actual cause, the name, appeared nowhere in the message.

   [C_syntax.kernel_ident] now normalizes the name once, at each backend's [compile] entry, against
   the same [ident_blacklist] the tensor-node names already avoid. What is pinned here:

   - The colliding routines COMPILE AND EXECUTE, with their values checked. A structural check on
   the emitted header would pass on a kernel that computes nothing; the point of the fix is that the
   routine works, not merely that it parses.
   - The emitted symbol is the mangled one, and the artifact names it — the deterministic scheme is
   observable, not just its absence of failure.
   - The identity half: a name that is already a legal, non-reserved identifier is emitted
   unchanged. This is what keeps schedule-cache identities and every existing golden from churning,
   and it is the half a mangling scheme can silently break.

   The names probed are chosen to be reserved on EVERY backend: [asm] is a keyword for gcc, clang,
   nvrtc, hiprtc and the Metal front end alike (which is why it is in [C_syntax.c_keywords] even
   though C89/C99 reserve it only as an extension), and [arrayjit_threefry4x32] is a builtin all four
   backends' tables define. The C++-only keywords ([class], [new], ...) are deliberately NOT probed
   for mangling: they are legal identifiers in plain C, so the cc backend must and does leave them
   alone — asserting a uniform mangling for them would be asserting something false. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p = Verdict.p
let approx a b = Float.(abs (a - b) < 1e-4)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let n = 16
let av = Array.init n ~f:(fun i -> Float.of_int (i % 5) *. 0.5)
let bv = Array.init n ~f:(fun i -> Float.of_int (i % 3) -. 1.)
let expected = Array.init n ~f:(fun i -> av.(i) +. bv.(i))

(* One compile+run of [a + b] under [~name], with fresh leaves each time so that no two cases share
   a comp (a comp's forward code is consumable once). Answers the values the routine produced. *)
let run_named ~label ~name =
  let a = TDSL.ndarray av ~label:[ label ^ "_a" ] ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~label:[ label ^ "_b" ] ~output_dims:[ n ] () in
  let%op c = a + b in
  Train.set_materialized c.Tensor.value;
  let comp = Train.forward c in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ~name ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx c.Tensor.value, routine.Context.name)

(* [claim] for a name that must be mangled to [emitted]: the routine runs correctly, the artifact
   the compile wrote defines the mangled symbol, and the caller's own record still carries the name
   it asked for (mangling is a codegen concern, not a rename of the user's routine). *)
let colliding ~label ~name ~emitted =
  let values, routine_name = run_named ~label ~name in
  p (Printf.sprintf "routine named %S computes correct values" name)
    (Array.for_all2_exn values expected ~f:approx);
  p (Printf.sprintf "routine named %S keeps its given name in the routine record" name)
    (String.equal routine_name name);
  Generated.assert_emits ~routine:emitted
    ~contains:("void " ^ emitted ^ "(")
    (Printf.sprintf "routine named %S emits the mangled symbol %S" name emitted)

(* --- 1. The reported case: a C keyword. --- *)
let () = colliding ~label:"gh686_kw" ~name:"asm" ~emitted:"asm__"

(* --- 2. A name that is a builtin every backend's table defines. Before the fix this both shadowed
   the definition (duplicate C symbol) and, since [filter_and_prepend_builtins] selects entries by
   searching the kernel for their key, was special-cased there to stop the name from dragging the
   definition in. Mangled, the routine's token is no longer the builtin's, so neither happens. --- *)
let () =
  colliding ~label:"gh686_bi" ~name:"arrayjit_threefry4x32" ~emitted:"arrayjit_threefry4x32__"

(* --- 3. Characters C does not admit in an identifier, and a leading digit. [Context.compile]'s
   [~name] bypasses [Assignments.get_name_exn]'s punctuation pass entirely, so these reached the
   emitted header as-is too. --- *)
let () = colliding ~label:"gh686_ch" ~name:"gh686 bad/name!" ~emitted:"gh686_bad_name_"
let () = colliding ~label:"gh686_dg" ~name:"686_leading_digit" ~emitted:"k_686_leading_digit"

(* --- 4. The identity half: an ordinary name is emitted verbatim. Without this, a mangling scheme
   that renamed everything would pass every claim above while churning every cache identity and
   every generated-source golden in the repository. --- *)
let () =
  let name = "gh686_ordinary_name" in
  let values, routine_name = run_named ~label:"gh686_ok" ~name in
  p "an ordinary routine name computes correct values" (Array.for_all2_exn values expected ~f:approx);
  p "an ordinary routine name reaches the routine record unchanged" (String.equal routine_name name);
  Generated.assert_emits ~routine:name
    ~contains:("void " ^ name ^ "(")
    "an ordinary routine name is emitted unmangled"

(* --- 5. The Block_comment surface the issue was filed from: no [~name] at all, the name derived
   from the block comment. This is the exact shape that produced [void asm(]. --- *)
let () =
  let a = TDSL.ndarray av ~label:[ "gh686_bc_a" ] ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~label:[ "gh686_bc_b" ] ~output_dims:[ n ] () in
  let%op c = a + b in
  Train.set_materialized c.Tensor.value;
  let comp = Tensor.consume_forward_code c in
  let comp = { comp with Asgns.asgns = Asgns.Block_comment ("asm", comp.Asgns.asgns) } in
  p "the block comment derives the reserved name" (String.equal (Asgns.get_name_exn comp.Asgns.asgns) "asm");
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  p "a Block_comment-named reserved routine computes correct values"
    (Array.for_all2_exn (Context.get_values ctx c.Tensor.value) expected ~f:approx)
