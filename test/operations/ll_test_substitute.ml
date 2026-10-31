(** [Ll_test.substitute ~from ~to_]: one node replaced by another everywhere the IR names a node.

    The helper exists for end-to-end negative controls of identity-keyed recognizers
    (schedule_epilogue_fusion's masked fragment, online_softmax_block_mma's fold tiles): a
    look-alike planted in lowered IR has to replace the real node in EVERY position, or the control
    judges a program in which the recognizer can still find the real node somewhere and passes for
    the wrong reason. So each position the IR has for a node is a case of its own here, built twice
    by one function -- over the real node and over the look-alike -- and the substitution of the
    first build must equal the second. Each case must also NAME the node (its two builds differ), or
    the equality is vacuous; and a substitution of a node the program does not hold must change
    nothing, the other node of every case included. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ll = Ll_test
open Verdict.Claims

let node = Ll.node_factory ~first_id:17210 ~dims:[| 4 |] ()
let real = node "subject"
let other = node "other"
let absent = node "absent"
let lookalike = Ll.look_alike ~id:17219 real
let i = Ll.sym ()
let at = [| Ll.iter i |]
let scope tn n : LL.scope_id = { tn; scope_id = n }
let read tn = Ll.get tn at
let write tn v = Ll.set tn at v
let arg v : LL.scalar_arg = (v, Ll.single)

let local_scope id body : LL.scalar_t =
  LL.Local_scope { id; body; orig_indices = at; mint = LL.Inlined_computation }

let mma ~d ~a ~b fallback = Ll.tile_mma ~lane:i ~d:(d, at) ~a:(a, at) ~b:(b, at) fallback

(* Every position of a node in [LL.t] / [LL.scalar_t], each built over the node [x] with [other] in
   every other position. *)
let cases : (string * (Tn.t -> LL.t)) list =
  [
    ("Set target", fun x -> write x (read other));
    ("Get", fun x -> write other (read x));
    ("Zero_out", fun x -> Ll.zero x);
    ( "Set_dynamic target",
      fun x -> Ll.scatter ~tn:x ~idcs:at ~dyn_axis:0 ~dyn_value:(arg (read other)) (read other) );
    ( "Set_dynamic row value",
      fun x -> Ll.scatter ~tn:other ~idcs:at ~dyn_axis:0 ~dyn_value:(arg (read x)) (Ll.c 1.) );
    ( "Get_dynamic table",
      fun x -> write other (Ll.gather ~tn:x ~idcs:at ~dyn_axis:0 ~dyn_value:(arg (read other))) );
    ( "Get_dynamic row value",
      fun x -> write other (Ll.gather ~tn:other ~idcs:at ~dyn_axis:0 ~dyn_value:(arg (read x))) );
    ("Get_merge_buffer", fun x -> write other (LL.Get_merge_buffer (x, at)));
    ( "Set_from_vec target",
      fun x ->
        Ll.set_from_vec ~length:4 ~vec_unop:Ir.Ops.Uint4x32_to_prec_uniform x at (arg (read other))
    );
    ( "Set_from_vec argument",
      fun x ->
        Ll.set_from_vec ~length:4 ~vec_unop:Ir.Ops.Uint4x32_to_prec_uniform other at (arg (read x))
    );
    ("Declare_local", fun x -> Ll.declare_local (scope x 1));
    ("Set_local id", fun x -> LL.Set_local (scope x 1, Ll.c 1.));
    ("Set_local value", fun x -> LL.Set_local (scope other 1, read x));
    ("Get_local", fun x -> write other (LL.Get_local (scope x 1)));
    ( "Local_scope id",
      fun x -> write other (local_scope (scope x 2) (LL.Set_local (scope x 2, read other))) );
    ( "Local_scope body",
      fun x -> write other (local_scope (scope other 2) (LL.Set_local (scope other 2, read x))) );
    ( "Scan_loop carried pair",
      fun x ->
        let cr : LL.carried = { prev = scope x 3; next = scope x 4; init = Ll.c 0. } in
        Ll.scan ~upto:3 i ~carried:[ cr ] (Ll.set_next cr (Ll.prev cr)) );
    ( "Scan_loop init",
      fun x ->
        let cr : LL.carried = { prev = scope other 3; next = scope other 4; init = read x } in
        Ll.scan ~upto:3 i ~carried:[ cr ] (Ll.set_next cr (Ll.prev cr)) );
    ("If condition", fun x -> Ll.if_ (read x) (write other (Ll.c 1.)));
    ("If body", fun x -> Ll.if_ (read other) (write x (Ll.c 1.)));
    ("For_loop body", fun x -> Ll.loop_n i 4 (write x (read other)));
    ("Seq first", fun x -> Ll.seq (Ll.zero x) (Ll.zero other));
    ("Seq second", fun x -> Ll.seq (Ll.zero other) (Ll.zero x));
    ("Tile_mma d", fun x -> mma ~d:x ~a:other ~b:other LL.Noop);
    ("Tile_mma a", fun x -> mma ~d:other ~a:x ~b:other LL.Noop);
    ("Tile_mma b", fun x -> mma ~d:other ~a:other ~b:x LL.Noop);
    ("Tile_mma fallback", fun x -> mma ~d:other ~a:other ~b:other (Ll.zero x));
  ]

let whole x = List.fold_right cases ~init:LL.Noop ~f:(fun (_, build) acc -> Ll.seq (build x) acc)

let () =
  p "the look-alike shares the subject's namespace, label, precision and shape, not its identity"
    (String.equal lookalike.Tn.namespace real.Tn.namespace
    && List.equal String.equal lookalike.Tn.label real.Tn.label
    && Ir.Ops.equal_prec (Lazy.force lookalike.Tn.storage_prec) (Lazy.force real.Tn.storage_prec)
    && Array.equal Int.equal (Lazy.force lookalike.Tn.dims) (Lazy.force real.Tn.dims)
    && not (Tn.equal lookalike real));
  p_all "every case names the node: its builds over the subject and the look-alike differ" cases
    ~f:(fun (_, build) -> not (LL.equal (build real) (build lookalike)));
  p_all "substituting the look-alike for the subject gives the build over the look-alike" cases
    ~f:(fun (_, build) ->
      LL.equal (Ll.substitute ~from:real ~to_:lookalike (build real)) (build lookalike));
  p_all "substituting a node the case does not hold changes nothing" cases ~f:(fun (_, build) ->
      LL.equal (Ll.substitute ~from:absent ~to_:lookalike (build real)) (build real));
  p "over all the cases in one program, the substitution gives the program over the look-alike"
    (LL.equal (Ll.substitute ~from:real ~to_:lookalike (whole real)) (whole lookalike));
  let seen = ref [] in
  let note tn = seen := tn :: !seen in
  Ll.walk
    (Ll.substitute ~from:real ~to_:lookalike (whole real))
    ~on_set:note ~on_get:note
    ~on_scope:(fun id -> note id.LL.tn);
  p_exists "the substituted program's accesses and scopes include the look-alike" !seen
    ~f:(Tn.equal lookalike);
  p_none "the substituted program's accesses and scopes never name the subject" !seen
    ~f:(Tn.equal real)
