(* Autotune (docs/proposals/schedule-ir-optops.md, the search-harness half of the OptOps port):
   canonical schedule identities and the empirical schedule search.

   Covered here, backend-independent (all printed booleans hold on every backend):

   - Canonical digests are process-stable across sibling compiles of the same computation (each
   backend [compile] re-lowers with fresh symbols), and distinguish different computations. - A
   schedule serialized against one compile ([Schedule_cache.to_saved]) replays against another
   ([of_saved]) with structural fidelity: the replayed transform produces code with the same
   canonical digest as the original transform, and the executed values match the unscheduled twin. -
   [Autotune.tune] returns a working routine whose values match the unscheduled twin, reports a
   cache miss on the first call and a cache hit on the second (same computation, fresh context), and
   the cached winner replays to correct values. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let rec first_loop (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> ( match first_loop a with Some r -> Some r | None -> first_loop b)
  | LL.For_loop { index; body; _ } -> Some (index, body)
  | _ -> None

let first_loop_exn llc = Option.value_exn ~here:[%here] (first_loop llc)

let () =
  let av = Array.init 32 ~f:(fun i -> Float.of_int i *. 0.5) in
  let bv = Array.init 32 ~f:(fun i -> Float.of_int (i % 7) -. 3.) in
  let expected_c = Array.init 32 ~f:(fun i -> av.(i) +. bv.(i)) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ 4; 8 ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ 4; 8 ] () in
  let%op c = a + b in
  let comp = named "canon_ab" (Train.forward c) in

  (* --- Digest stability across sibling compiles (fresh lowering, fresh symbols) --- *)
  let capture () =
    let captured = ref None in
    let ctx = Context.auto () in
    let _ctx, _routine =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some opt;
          opt)
        ctx comp Ir.Indexing.Empty
    in
    Option.value_exn ~here:[%here] !captured
  in
  let opt1 = capture () in
  let opt2 = capture () in
  let canon1 = SC.canonicalize opt1 in
  let canon2 = SC.canonicalize opt2 in
  p "canonical digest stable across sibling compiles"
    (String.equal (SC.digest canon1) (SC.digest canon2));
  p "canonical form complete" (SC.complete canon1);
  let mav = Array.init 20 ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init 30 ~f:(fun i -> Float.of_int (i % 11) -. 4.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ 5 ] ~output_dims:[ 4 ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ 6 ] ~output_dims:[ 5 ] () in
  let%op mc = ma * mb in
  let mm_comp = named "canon_mm" (Train.forward mc) in
  let capture_mm () =
    let captured = ref None in
    let ctx = Context.auto () in
    let _ctx, _routine =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some opt;
          opt)
        ctx mm_comp Ir.Indexing.Empty
    in
    Option.value_exn ~here:[%here] !captured
  in
  let mm_opt = capture_mm () in
  p "different computations have different digests"
    (not (String.equal (SC.digest canon1) (SC.digest (SC.canonicalize mm_opt))));

  (* --- Local scope ids are alpha-numbered in the digest (Codex P2 on PR #103): [get_scope] draws
     from a process-global counter, so the same schedule applied in two compiles consumes different
     raw ids. [Privatize] introduces [Declare_local]/[Set_local]/[Get_local]: without
     alpha-numbering, the digest of a fresh compile's replay could never match the digest of the
     direct application. --- *)
  let mm_expected = Array.create ~len:24 0. in
  Array.iteri mm_expected ~f:(fun idx _ ->
      let i = idx / 6 and j = idx % 6 in
      let acc = ref 0. in
      for k = 0 to 4 do
        acc := !acc +. (mav.((i * 5) + k) *. mbv.((k * 6) + j))
      done;
      mm_expected.(idx) <- !acc);
  let mm_canon = SC.canonicalize mm_opt in
  let _i, mm_b1 = first_loop_exn mm_opt.LL.llc in
  let _j, mm_b2 = first_loop_exn mm_b1 in
  let mm_k, _ = first_loop_exn mm_b2 in
  let priv_sched = [ Sched.Privatize { target = mc.Tensor.value; over = mm_k } ] in
  let priv_saved, _reg = SC.to_saved (SC.base_registry mm_canon) priv_sched in
  let priv_direct = Sched.apply priv_sched mm_opt in
  let priv_direct_digest = SC.digest (SC.canonicalize priv_direct) in
  (* Two structurally identical codes whose scope ids differ (as two lowering / transform runs
     produce, [LL.get_scope] being a process-global counter) must digest equally. *)
  let scope_tn =
    Ir.Tnode.create (Ir.Tnode.Specified Ir.Ops.single) ~id:9001 ~label:[ "scope_probe" ]
      ~unpadded_dims:(lazy [| 4 |])
      ~padding:(lazy None)
      ()
  in
  let build_scoped () =
    let s = Ir.Indexing.get_symbol () in
    let scope = LL.get_scope scope_tn in
    let fake llc : LL.optimized =
      {
        LL.traced_store = Hashtbl.create (module Ir.Tnode);
        optimize_ctx = LL.empty_optimize_ctx ();
        llc;
        merge_node = None;
        workgroup_shared = Set.empty (module Ir.Tnode);
        simdgroup_fragments = Set.empty (module Ir.Tnode);
        swizzled = Set.empty (module Ir.Tnode);
        zero_fringe = Set.empty (module Ir.Tnode);
      }
    in
    fake
      (LL.For_loop
         {
           index = s;
           from_ = 0;
           to_ = 3;
           trace_it = false;
           axis = LL.Serial;
           body =
             LL.Seq
               ( LL.Declare_local { id = scope; needs_init = false },
                 LL.Seq
                   ( LL.Set_local (scope, LL.Constant 1.),
                     LL.Set
                       {
                         tn = scope_tn;
                         idcs = [| Ir.Indexing.Iterator s |];
                         llsc = LL.Get_local scope;
                         debug = "";
                       } ) );
         })
  in
  p "scope ids alpha-numbered in the digest"
    (String.equal
       (SC.digest (SC.canonicalize (build_scoped ())))
       (SC.digest (SC.canonicalize (build_scoped ()))));
  let priv_replay_digest = ref "" in
  let pctx = Context.auto () in
  let pctx, proutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let canon = SC.canonicalize opt in
        assert (String.equal (SC.digest canon) (SC.digest mm_canon));
        let sched, _reg = SC.of_saved canon priv_saved in
        let opt' = Sched.apply sched opt in
        priv_replay_digest := SC.digest (SC.canonicalize opt');
        opt')
      pctx mm_comp Ir.Indexing.Empty
  in
  let pctx = Context.run pctx proutine in
  let priv_got = Context.get_values pctx mc.Tensor.value in
  p "privatized replay values correct" (Array.for_all2_exn priv_got mm_expected ~f:approx);
  p "privatized replay structurally equals the direct application"
    (String.equal !priv_replay_digest priv_direct_digest);

  (* --- to_saved / of_saved roundtrip: serialize a split+swap schedule built against compile #1,
     replay it inside a fresh compile, check executed values and structural (digest) parity --- *)
  let sym1, _ = first_loop_exn opt1.LL.llc in
  let split_op, _outer, inner =
    Sched.split ~axis:sym1 ~factor:2 ~outer:LL.Serial ~inner:LL.Serial
  in
  let sched1 = [ split_op; Sched.Unroll { axis = inner; materialize = true } ] in
  let saved, _reg = SC.to_saved (SC.base_registry canon1) sched1 in
  let direct = Sched.apply sched1 opt1 in
  let direct_digest = SC.digest (SC.canonicalize direct) in
  let replay_digest = ref "" in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let canon = SC.canonicalize opt in
        assert (String.equal (SC.digest canon) (SC.digest canon1));
        let sched, _reg = SC.of_saved canon saved in
        let opt' = Sched.apply sched opt in
        replay_digest := SC.digest (SC.canonicalize opt');
        opt')
      ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx c.Tensor.value in
  p "replayed schedule values correct" (Array.for_all2_exn got expected_c ~f:approx);
  p "replayed schedule structurally equals the direct application"
    (String.equal !replay_digest direct_digest);
  p "saved schedule roundtrips through sexp"
    (SC.equal_saved_schedule saved (SC.saved_schedule_of_sexp (SC.sexp_of_saved_schedule saved)));

  (* --- End-to-end tune on a combo computation (an elementwise nest and a matmul nest): first call
     searches (cache miss), second call (same computation, fresh context) hits the cache; both
     return routines computing correct values --- *)
  let%op tc1 = a + b in
  let%op tc2 = ma * mb in
  let tune_comp = named "tune_combo" (Asgns.sequence [ Train.forward tc1; Train.forward tc2 ]) in
  let cache_dir = "autotune_cache_test" in
  (* The dune sandbox persists across runs: a stale entry written by an older binary (digest
     ingredients and the saved-schedule format evolve) breaks the miss-then-hit assertions below, so
     start from a clean cache. *)
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let reports = ref [] in
  let tune_once () =
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir
        ~report:(fun r -> reports := r :: !reports)
        ctx tune_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    (Context.get_values ctx tc1.Tensor.value, Context.get_values ctx tc2.Tensor.value)
  in
  let got1, got_mm1 = tune_once () in
  let got2, got_mm2 = tune_once () in
  let r2, r1 =
    match !reports with [ r2; r1 ] -> (r2, r1) | _ -> failwith "expected two reports"
  in
  p "tuned routine values correct"
    (Array.for_all2_exn got1 expected_c ~f:approx
    && Array.for_all2_exn got_mm1 mm_expected ~f:approx);
  p "first tune call searches (cache miss)" (not r1.Autotune.cache_hit);
  p "first tune call timed at least the baseline" (r1.Autotune.candidates_timed >= 1);
  p "completed search report is not partial"
    ((not r1.Autotune.partial) && Option.is_none r1.Autotune.terminal_failure);
  p "decline census sums to candidates_failed"
    (r1.Autotune.candidates_failed
    = List.sum (module Int) r1.Autotune.declines ~f:(fun d -> d.Autotune.count));
  p "second tune call hits the schedule cache" r2.Autotune.cache_hit;
  p "cache-hit report has no declines"
    (List.is_empty r2.Autotune.declines && r2.Autotune.candidates_failed = 0);
  p "cached schedule replays to correct values"
    (Array.for_all2_exn got2 expected_c ~f:approx
    && Array.for_all2_exn got_mm2 mm_expected ~f:approx)
