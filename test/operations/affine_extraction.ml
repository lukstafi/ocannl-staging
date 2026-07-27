(* gh-494 waypoint 1: [Ir.Low_level.affine_accesses] — extraction of a program's tensor-node
   accesses as explicit affine relations, the queryable artifact behind the affine legality queries.
   The dump covers the statement forms broadly: plain nests, whole-node [Zero_out],
   read-modify-write accumulations (the [rmw] reduction-dependence flag), guarded statements,
   dynamic gathers/scatters, and vectorized writes. The tail composes extraction with
   [Affine.pair_conflict]: which loops of the reduction nest admit parallelization — the
   determinism-as-constraint reading, where the accumulation confines conflicts over [i] to one
   thread while any parallelization over the reduced [k] is refuted. *)

open Base
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Aff = Ir.Affine
module Tn = Ir.Tnode
module Ops = Ir.Ops

let fresh_tn =
  let c = ref 960_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ops.single

let for_over ?(extent = 4) sym body =
  LL.For_loop { index = sym; from_ = 0; to_ = extent - 1; body; trace_it = false; axis = LL.Serial }

let get tn idcs = LL.Get (tn, idcs)
let it s = Idx.Iterator s

let show_idx = function
  | Idx.Fixed_idx k -> Int.to_string k
  | Idx.Iterator s -> Idx.symbol_ident s
  | Idx.Affine { symbols; offset } ->
      let terms =
        List.map symbols ~f:(fun (c, s) -> Printf.sprintf "%d*%s" c (Idx.symbol_ident s))
      in
      String.concat ~sep:"+" (terms @ if offset = 0 then [] else [ Int.to_string offset ])
  | Idx.Sub_axis -> "sub"
  | Idx.Concat _ -> "concat"

let show (a : Tn.t Aff.access) =
  let flags =
    String.concat ~sep:""
      (List.filter_map
         [
           (a.a_dynamic, "dyn ");
           (a.a_whole, "whole ");
           (a.a_vec_last, "vec ");
           (a.a_guarded, "guarded ");
           (a.a_rmw, "rmw ");
         ]
         ~f:(fun (b, s) -> Option.some_if b s))
  in
  Stdio.printf "%-2s %-3s %-14s loops=[%s] path=[%s] %s\n"
    (if a.a_write then "wr" else "rd")
    (Tn.debug_name a.a_tn)
    (Printf.sprintf "[%s]" (String.concat_array ~sep:";" (Array.map a.a_map ~f:show_idx)))
    (String.concat ~sep:","
       (List.map a.a_loops ~f:(fun (s, (lo, hi)) ->
            Printf.sprintf "%s:%d..%d" (Idx.symbol_ident s) lo hi)))
    (String.concat ~sep:"." (List.map a.a_path ~f:Int.to_string))
    flags

let () =
  let a = fresh_tn "A" [| 4; 5 |] in
  let b = fresh_tn "B" [| 3 |] in
  let c = fresh_tn "C" [| 4; 3 |] in
  let s = fresh_tn "S" [| 4 |] in
  let d = fresh_tn "D" [| 1 |] in
  let g = fresh_tn "G" [| 1 |] in
  let e = fresh_tn "E" [| 4 |] in
  let ids = fresh_tn "I" [| 4 |] in
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  let i2 = Idx.get_symbol () and i3 = Idx.get_symbol () in
  let pointwise =
    (* for i: for j: C[i][j] = A[i][j] + B[j] *)
    for_over i
      (for_over ~extent:3 j
         (LL.Set
            {
              tn = c;
              idcs = [| it i; it j |];
              llsc = LL.Binop (Ops.Add, (get a [| it i; it j |], sp), (get b [| it j |], sp));
              debug = "";
            }))
  in
  let reduction =
    (* for i2: for k: S[i2] = S[i2] + A[i2][k] — an accumulation: rmw *)
    for_over i2
      (for_over ~extent:5 k
         (LL.Set
            {
              tn = s;
              idcs = [| it i2 |];
              llsc = LL.Binop (Ops.Add, (get s [| it i2 |], sp), (get a [| it i2; it k |], sp));
              debug = "";
            }))
  in
  let guarded =
    (* if G[0] then D[0] = 1 — a conditional (never-definite) write *)
    LL.If
      {
        cond = (get g [| Idx.Fixed_idx 0 |], sp);
        body = LL.Set { tn = d; idcs = [| Idx.Fixed_idx 0 |]; llsc = LL.Constant 1.; debug = "" };
      }
  in
  let gather =
    (* for i3: E[i3] = A[I[i3]][0] — dynamic gather *)
    for_over i3
      (LL.Set
         {
           tn = e;
           idcs = [| it i3 |];
           llsc =
             LL.Get_dynamic
               {
                 tn = a;
                 idcs = [| Idx.Fixed_idx 0; Idx.Fixed_idx 0 |];
                 dyn_axis = 0;
                 dyn_value = (get ids [| it i3 |], sp);
               };
           debug = "";
         })
  in
  let program = LL.unflat_lines [ LL.Zero_out s; pointwise; reduction; guarded; gather ] in
  Stdio.printf "=== affine_accesses dump ===\n";
  let accesses = LL.affine_accesses program in
  List.iter accesses ~f:show;

  Stdio.printf "\n=== which loops of the reduction nest parallelize? ===\n";
  let nest_accs =
    List.filter accesses ~f:(fun ac ->
        List.exists ac.Aff.a_loops ~f:(fun (sym, _) -> Idx.equal_symbol sym i2))
  in
  let range sym =
    List.find_map nest_accs ~f:(fun ac ->
        List.Assoc.find ac.Aff.a_loops sym ~equal:Idx.equal_symbol)
  in
  let dup sym = Option.is_some (range sym) in
  let check_par name sym =
    (* Every pair over a common node with at least one write must confine its conflicts to one
       thread of [sym]. *)
    let verdicts =
      List.concat_map nest_accs ~f:(fun x ->
          List.filter_map nest_accs ~f:(fun y ->
              if (x.Aff.a_write || y.Aff.a_write) && x.Aff.a_tn.Tn.uid = y.Aff.a_tn.Tn.uid then
                Some
                  (Aff.pair_conflict ~range ~dup_left:dup ~dup_right:dup
                     ~pairs:[ (sym, sym) ]
                     ~left:x.Aff.a_map ~right:y.Aff.a_map)
              else None))
    in
    let safe = List.for_all verdicts ~f:(function Aff.Cross_thread _ -> false | _ -> true) in
    Stdio.printf "%s %s parallelizable: %b\n" (Idx.symbol_ident sym) name safe
  in
  check_par "(map axis)" i2;
  check_par "(reduced axis)" k
