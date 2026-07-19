(* gh-ocannl-491 task 1: [Ir.Cost_model] — footprint/FLOPs extraction and the roofline bound over
   hand-built programs where every number is checkable by hand (single precision, 4 bytes/cell):

   - elementwise map: bytes = the three operand footprints, one op per cell;
   - matmul: FLOPs = 2*M*N*K, bytes = (MK + KN + 2*MN) * 4 (the rmw accumulator charged once per
     direction);
   - strided/gapped access: the image cardinality counts the touched cells, not the node size;
   - rmw reduction: the accumulator's read/write/rmw split;
   - guarded write: guards-taken op count ([flops_approx]) and a never-definite write;
   - dynamic gather: the uninterpretable-component fallback to whole-node bytes;
   - overlapping writes: the union bound capped by the node's size.

   The tail asserts the roofline bound is monotone in the envelope constants. *)

open Base
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops
module CM = Ir.Cost_model

let fresh_tn =
  let c = ref 970_000_000 in
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

let show_summary name (s : CM.summary) =
  Stdio.printf "== %s ==\n" name;
  List.iter s.CM.per_node ~f:(fun (tn, fp) ->
      Stdio.printf "  %-3s rd=%-4d wr=%-4d rmw=%-4d %s\n" (Tn.debug_name tn) fp.CM.fp_read_bytes
        fp.CM.fp_write_bytes fp.CM.fp_rmw_bytes
        (if fp.CM.fp_approx then "approx" else "exact"));
  Stdio.printf "  flops=%d%s read=%d write=%d intensity=%.3f%s\n" s.CM.flops
    (if s.CM.flops_approx then "~" else "")
    s.CM.read_bytes s.CM.write_bytes (CM.arithmetic_intensity s)
    (if s.CM.opaque then " OPAQUE" else "")

let () =
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  (* Elementwise map, 4x5: for i: for j: C[i][j] = A[i][j] + B[j].
     A rd 20 cells = 80 B, B rd 5 cells = 20 B, C wr 20 cells = 80 B; 1 add x 20 iters. *)
  let a = fresh_tn "A" [| 4; 5 |] in
  let b = fresh_tn "B" [| 5 |] in
  let c = fresh_tn "C" [| 4; 5 |] in
  let pointwise =
    for_over i
      (for_over ~extent:5 j
         (LL.Set
            {
              tn = c;
              idcs = [| it i; it j |];
              llsc = LL.Binop (Ops.Add, (get a [| it i; it j |], sp), (get b [| it j |], sp));
              debug = "";
            }))
  in
  show_summary "elementwise map 4x5" (CM.analyze pointwise);

  (* Matmul M=4, N=3, K=5: for i: for j: for k: D[i][j] = D[i][j] + A[i][k] * B2[k][j].
     FLOPs = 2*M*N*K = 120; A rd MK=20 cells, B2 rd KN=15, D rd MN=12 + wr 12 (rmw). *)
  let a_mk = fresh_tn "Am" [| 4; 5 |] in
  let b_kn = fresh_tn "Bm" [| 5; 3 |] in
  let d_mn = fresh_tn "Dm" [| 4; 3 |] in
  let matmul =
    for_over i
      (for_over ~extent:3 j
         (for_over ~extent:5 k
            (LL.Set
               {
                 tn = d_mn;
                 idcs = [| it i; it j |];
                 llsc =
                   LL.Binop
                     ( Ops.Add,
                       (get d_mn [| it i; it j |], sp),
                       ( LL.Binop (Ops.Mul, (get a_mk [| it i; it k |], sp), (get b_kn [| it k; it j |], sp)),
                         sp ) );
                 debug = "";
               })))
  in
  let mm = CM.analyze matmul in
  show_summary "matmul 4x3x5" mm;

  (* The same matmul in FMA form: D[i][j] = FMA(A[i][k], B2[k][j], D[i][j]) — an FMA counts as two
     operations, so the score is identical to the mul+add form. *)
  let matmul_fma =
    for_over i
      (for_over ~extent:3 j
         (for_over ~extent:5 k
            (LL.Set
               {
                 tn = d_mn;
                 idcs = [| it i; it j |];
                 llsc =
                   LL.Ternop
                     ( Ops.FMA,
                       (get a_mk [| it i; it k |], sp),
                       (get b_kn [| it k; it j |], sp),
                       (get d_mn [| it i; it j |], sp) );
                 debug = "";
               })))
  in
  show_summary "matmul 4x3x5, FMA form" (CM.analyze matmul_fma);

  (* Strided/gapped, size-8 nodes: for i: R[2*i] = A8[2*i] * 2 — touches 4 of 8 cells each. *)
  let a8 = fresh_tn "A8" [| 8 |] in
  let r8 = fresh_tn "R8" [| 8 |] in
  let stride2 s = Idx.Affine { symbols = [ (2, s) ]; offset = 0 } in
  let strided =
    for_over i
      (LL.Set
         {
           tn = r8;
           idcs = [| stride2 i |];
           llsc = LL.Binop (Ops.Mul, (get a8 [| stride2 i |], sp), (LL.Constant 2., sp));
           debug = "";
         })
  in
  show_summary "strided copy-scale over half the cells" (CM.analyze strided);

  (* Rmw reduction: for i: for k: S[i] = S[i] + A[i][k] — S rd 16 B, wr 16 B all rmw. *)
  let s = fresh_tn "S" [| 4 |] in
  let reduction =
    for_over i
      (for_over ~extent:5 k
         (LL.Set
            {
              tn = s;
              idcs = [| it i |];
              llsc = LL.Binop (Ops.Add, (get s [| it i |], sp), (get a [| it i; it k |], sp));
              debug = "";
            }))
  in
  show_summary "rmw reduction 4x5" (CM.analyze reduction);

  (* Guarded write: if G[0] then D1[0] = G[0] * 3 — guards-taken op count, approximate write. *)
  let g = fresh_tn "G" [| 1 |] in
  let d1 = fresh_tn "D1" [| 1 |] in
  let guarded =
    LL.If
      {
        cond = (get g [| Idx.Fixed_idx 0 |], sp);
        body =
          LL.Set
            {
              tn = d1;
              idcs = [| Idx.Fixed_idx 0 |];
              llsc = LL.Binop (Ops.Mul, (get g [| Idx.Fixed_idx 0 |], sp), (LL.Constant 3., sp));
              debug = "";
            };
      }
  in
  show_summary "guarded write" (CM.analyze guarded);

  (* Dynamic gather: for i: E[i] = A[I[i]][0] — the table read falls back to whole-node bytes. *)
  let e = fresh_tn "E" [| 4 |] in
  let ids = fresh_tn "I" [| 4 |] in
  let gather =
    for_over i
      (LL.Set
         {
           tn = e;
           idcs = [| it i |];
           llsc =
             LL.Get_dynamic
               {
                 tn = a;
                 idcs = [| Idx.Fixed_idx 0; Idx.Fixed_idx 0 |];
                 dyn_axis = 0;
                 dyn_value = (get ids [| it i |], sp);
               };
           debug = "";
         })
  in
  show_summary "dynamic gather (whole-node fallback)" (CM.analyze gather);

  (* Overlapping writes: Zero_out S2 then a covering pointwise write — the per-direction sum
     (16 + 16 B) is a union bound, capped by the node's 16 bytes and flagged approximate. *)
  let s2 = fresh_tn "S2" [| 4 |] in
  let overlap =
    LL.unflat_lines
      [
        LL.Zero_out s2;
        for_over i
          (LL.Set { tn = s2; idcs = [| it i |]; llsc = LL.Constant 0.5; debug = "" });
      ]
  in
  show_summary "zero-out then overwrite (union bound capped)" (CM.analyze overlap);

  (* Roofline: monotone in the envelope constants, bandwidth- vs. compute-bound flips. *)
  Stdio.printf "\n== roofline over the matmul (flops=%d, bytes=%d) ==\n" mm.CM.flops
    (CM.total_bytes mm);
  let bound ?peak_flops ?peak_memory_bandwidth () =
    CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:mm.CM.flops
      ~bytes:(CM.total_bytes mm) ()
  in
  let show_bound name t =
    Stdio.printf "  %-28s %s\n" name
      (match t with None -> "no bound" | Some t -> Printf.sprintf "%.1f ns" (t *. 1e9))
  in
  show_bound "no envelope" (bound ());
  show_bound "1 GFLOP/s only" (bound ~peak_flops:1e9 ());
  show_bound "1 GB/s only" (bound ~peak_memory_bandwidth:1e9 ());
  show_bound "1 GFLOP/s, 1 GB/s" (bound ~peak_flops:1e9 ~peak_memory_bandwidth:1e9 ());
  show_bound "2 GFLOP/s, 1 GB/s" (bound ~peak_flops:2e9 ~peak_memory_bandwidth:1e9 ());
  show_bound "1 GFLOP/s, 2 GB/s" (bound ~peak_flops:1e9 ~peak_memory_bandwidth:2e9 ());
  show_bound "2 GFLOP/s, 2 GB/s" (bound ~peak_flops:2e9 ~peak_memory_bandwidth:2e9 ());
  let peaks = [ 1e9; 2e9; 4e9 ] in
  let monotone =
    List.for_all peaks ~f:(fun pf ->
        List.for_all peaks ~f:(fun bw ->
            let t0 = Option.value_exn (bound ~peak_flops:pf ~peak_memory_bandwidth:bw ()) in
            List.for_all
              [
                bound ~peak_flops:(2. *. pf) ~peak_memory_bandwidth:bw ();
                bound ~peak_flops:pf ~peak_memory_bandwidth:(2. *. bw) ();
              ]
              ~f:(fun t -> Float.(Option.value_exn t <= t0))))
  in
  Stdio.printf "  monotone in the envelope constants: %b\n" monotone
