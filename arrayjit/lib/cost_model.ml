open Base
module Idx = Indexing
module Tn = Tnode

(* Analytic cost model, extraction half (gh-ocannl-491 task 1). See cost_model.mli for the
   approximation contract — every count here is an upper bound (exactness tracked per node) except
   under [opaque], the flagged under-counting escape hatch. *)

type node_footprint = {
  fp_read_bytes : int;
  fp_write_bytes : int;
  fp_rmw_bytes : int;
  fp_approx : bool;
}
[@@deriving sexp_of]

type summary = {
  per_node : (Tn.t * node_footprint) list;
  read_bytes : int;
  write_bytes : int;
  flops : int;
  flops_approx : bool;
  opaque : bool;
}
[@@deriving sexp_of]

(* Distinct cells one access can touch, with exactness. Interpretable maps: image cardinality =
   loop-box size / fiber size ({!Affine.fiber_cardinality}); an [`At_least] fiber (non-injective
   map) makes that an upper bound on the image. Uninterpretable components fall back to the whole
   node. Guarded accesses are counted guards-taken. All biases over-count. *)
let access_cells (a : Tn.t Affine.access) : int * bool =
  let node_cells = Tn.num_elems a.a_tn in
  let uninterpretable =
    a.a_dynamic
    || Array.exists a.a_map ~f:(function Idx.Sub_axis | Idx.Concat _ -> true | _ -> false)
  in
  if a.a_whole then (node_cells, a.a_guarded)
  else if uninterpretable then (node_cells, true)
  else
    let domain = List.map a.a_loops ~f:(fun (s, (lo, hi)) -> (s, hi - lo + 1)) in
    let box = List.fold domain ~init:1 ~f:(fun acc (_, w) -> acc * w) in
    let image, exact_image =
      match Affine.fiber_cardinality ~domain a.a_map with
      | `Exact f -> (box / max 1 f, true)
      | `At_least f -> (box / max 1 f, false)
    in
    if a.a_vec_last then
      (* Each map instance is the base of a run along the minor axis: runs may overlap for strided
         bases, so the product is an upper bound. *)
      (min node_cells (image * max 1 a.a_vec_len), true)
    else (min node_cells image, (not exact_image) || a.a_guarded)

let footprints (accesses : Tn.t Affine.access list) : (Tn.t * node_footprint) list =
  (* Per node and direction: sum of per-access cell counts (a union upper bound, capped by the
     node's size); exact only for a single exact access in the direction. *)
  let tbl = Hashtbl.create (module Tn) in
  let order = ref [] in
  List.iter accesses ~f:(fun a ->
      let cells, approx = access_cells a in
      let cur =
        Hashtbl.find_or_add tbl a.a_tn ~default:(fun () ->
            order := a.a_tn :: !order;
            (0, 0, 0, 0, 0, false))
      in
      let reads, writes, rmws, nreads, nwrites, apx = cur in
      let next =
        if a.a_write then
          ( reads,
            writes + cells,
            (rmws + if a.a_rmw then cells else 0),
            nreads,
            nwrites + 1,
            apx || approx )
        else (reads + cells, writes, rmws, nreads + 1, nwrites, apx || approx)
      in
      Hashtbl.set tbl ~key:a.a_tn ~data:next);
  List.rev_map !order ~f:(fun tn ->
      let reads, writes, rmws, nreads, nwrites, apx = Hashtbl.find_exn tbl tn in
      let width = Ops.prec_in_bytes (Lazy.force tn.Tn.storage_prec) in
      let node_bytes = Tn.num_elems tn * width in
      let cap n = min node_bytes (n * width) in
      ( tn,
        {
          fp_read_bytes = cap reads;
          fp_write_bytes = cap writes;
          fp_rmw_bytes = cap rmws;
          fp_approx = apx || nreads > 1 || nwrites > 1;
        } ))

let analyze (code : Low_level.t) : summary =
  let flops_approx = ref false and opaque = ref false in
  (* [scale] is the product of enclosing loop extents, [env] their (symbol, extent) bindings —
     needed by [Tile_mma], whose 2*m*n*k multiply-adds are cooperative across its [lane] loop, not
     repeated per lane. *)
  let rec go ~scale ~env (c : Low_level.t) : int =
    match c with
    | Low_level.Noop | Comment _ | Zero_out _ | Declare_local _ | Workgroup_barrier -> 0
    | Staged_compilation _ ->
        opaque := true;
        0
    | Seq (c1, c2) -> go ~scale ~env c1 + go ~scale ~env c2
    | For_loop { index; from_; to_; body; _ } ->
        let extent = max 0 (to_ - from_ + 1) in
        go ~scale:(scale * extent) ~env:((index, extent) :: env) body
    | Set { llsc; _ } -> scale * sc_flops llsc
    | Set_dynamic { dyn_value = dv, _; llsc; _ } -> scale * (sc_flops dv + sc_flops llsc)
    | Set_from_vec { length; arg = a, _; _ } -> scale * (length + sc_flops a)
    | Set_local (_, llsc) -> scale * sc_flops llsc
    | If { cond = cnd, _; body } ->
        (* Guards-taken: the body is charged as if the guard always passes. *)
        flops_approx := true;
        (scale * sc_flops cnd) + go ~scale ~env body
    | Tile_mma { m; n; k; lane; _ } ->
        let lane_extent =
          List.Assoc.find env lane ~equal:Idx.equal_symbol |> Option.value ~default:1
        in
        scale / max 1 lane_extent * (2 * m * n * k)
  and sc_flops (sc : Low_level.scalar_t) : int =
    match sc with
    | Low_level.Local_scope { body; _ } -> go ~scale:1 ~env:[] body
    | Get_local _ | Get _ | Constant _ | Constant_bits _ | Embed_index _ -> 0
    | Get_merge_buffer _ ->
        (* Merge-buffer traffic is not represented by [affine_accesses] either: flag the
           under-count. *)
        opaque := true;
        0
    | Get_dynamic { dyn_value = dv, _; _ } -> sc_flops dv
    | Ternop ((Ops.FMA | Ops.Mul3), a1, a2, a3) ->
        (* Two arithmetic operations each — matching [peak_flops]' FMA-counted-as-two convention, so
           an FMA-form kernel scores the same compute leg as its mul+add form. *)
        2 + arg a1 + arg a2 + arg a3
    | Ternop (Ops.Where, a1, a2, a3) -> 1 + arg a1 + arg a2 + arg a3
    | Binop ((Ops.Arg1 | Ops.Arg2), a1, a2) -> arg a1 + arg a2
    | Binop (_, a1, a2) -> 1 + arg a1 + arg a2
    | Unop (Ops.Identity, a1) -> arg a1
    | Unop (_, a1) -> 1 + arg a1
  and arg (sc, _prec) = sc_flops sc in
  let flops = go ~scale:1 ~env:[] code in
  let per_node = footprints (Low_level.affine_accesses code) in
  {
    per_node;
    read_bytes = List.fold per_node ~init:0 ~f:(fun acc (_, fp) -> acc + fp.fp_read_bytes);
    write_bytes = List.fold per_node ~init:0 ~f:(fun acc (_, fp) -> acc + fp.fp_write_bytes);
    flops;
    flops_approx = !flops_approx;
    opaque = !opaque;
  }

let total_bytes s = s.read_bytes + s.write_bytes
let arithmetic_intensity s = Float.of_int s.flops /. Float.of_int (max 1 (total_bytes s))

let footprint_approximate s = List.exists s.per_node ~f:(fun (_, fp) -> fp.fp_approx)
let approximate s = s.flops_approx || footprint_approximate s

let roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops ~bytes () : float option =
  let legs =
    List.filter_opt
      [
        Option.map peak_flops ~f:(fun p -> Float.of_int flops /. p);
        Option.map peak_memory_bandwidth ~f:(fun p -> Float.of_int bytes /. p);
      ]
  in
  match legs with [] -> None | l -> Some (List.reduce_exn l ~f:Float.max)

module Calibration = struct
  (* The calibration TSV schema and the envelope fitter over it (gh-ocannl-514 phase 0). This
     module is the schema's single owner: rows are emitted through [to_line] (Autotune) and read
     back through [of_line] (tools/fit_envelope.exe), so writer and reader cannot drift apart. *)

  type row = {
    backend : string;
    digest : string;
    label : string;
    measured_ms : float;
    model_ms : float option;
    kernels : int;
    flops : int;
    bytes : int;
    flops_approx : bool;
    bytes_approx : bool;
    opaque : bool;
  }
  [@@deriving sexp_of]

  (* Milliseconds are serialized FLOORED at the 6th decimal (not rounded to nearest): a stored
     time never exceeds the true measurement, so constants fit from the file remain conservative
     with respect to the original in-process measurement — round-to-nearest could round a short
     kernel's time up by 0.5 ns, a 1e-4 relative error at 5 us, far above [report]'s 2e-6 bump.
     The floored decimal survives the round-trip: [d /. 1e6] for integral [d] is within ~1e-11 of
     the decimal, so ["%.6f"] prints [d] back exactly. *)
  let floor6 v = Float.round_down (v *. 1e6) /. 1e6

  let to_line r =
    Printf.sprintf "%s\t%s\t%s\t%.6f\t%s\t%d\t%d\t%d\t%b\t%b\t%b" r.backend r.digest r.label
      (floor6 r.measured_ms)
      (match r.model_ms with Some m -> Printf.sprintf "%.6f" (floor6 m) | None -> "")
      r.kernels r.flops r.bytes r.flops_approx r.bytes_approx r.opaque

  let of_line line =
    match String.split line ~on:'\t' with
    | [ backend; digest; label; measured; model; kernels; flops; bytes; flops_approx; bytes_approx; opaque ]
      -> (
        try
          Some
            {
              backend;
              digest;
              label;
              measured_ms = Float.of_string measured;
              model_ms =
                (if String.is_empty model then None else Some (Float.of_string model));
              kernels = Int.of_string kernels;
              flops = Int.of_string flops;
              bytes = Int.of_string bytes;
              flops_approx = Bool.of_string flops_approx;
              bytes_approx = Bool.of_string bytes_approx;
              opaque = Bool.of_string opaque;
            }
        with _ -> None)
    | _ -> None

  type fit = {
    fit_backend : string;
    fit_rows : int;
    fit_opaque : int;
    fit_flops_approx : int;
    fit_bytes_approx : int;
    fit_multi_kernel : int;
    fit_violations : int;
    fit_fission_slack : (float * string) option;
    fit_peak_flops : (float * string) option;
    fit_peak_memory_bandwidth : (float * string) option;
  }
  [@@deriving sexp_of]

  let fit rows =
    let by_backend = Hashtbl.create (module String) in
    let order = ref [] in
    List.iter rows ~f:(fun r ->
        let cell =
          Hashtbl.find_or_add by_backend r.backend ~default:(fun () ->
              order := r.backend :: !order;
              ref [])
        in
        cell := r :: !cell);
    List.rev_map !order ~f:(fun backend ->
        let rows = List.rev !(Hashtbl.find_exn by_backend backend) in
        (* Each leg fits from the rows where THAT leg's counts are exact (and non-opaque,
           positively timed): guards-taken / union over-counting can "achieve" a counts/time
           ratio far above any hardware peak, and letting it drive a maximum would inflate the
           envelope machine-wide — but exactness is per leg (a multi-read footprint makes bytes
           approximate without touching an exact op count), so a row feeds whichever legs it can
           prove. The fitted constant per leg is the tightest sound one for those rows —
           [bound <= measured] needs [peak >= counts/measured] on every row — so it is the
           maximum achieved [counts/time]. *)
        let timed = List.filter rows ~f:(fun r -> (not r.opaque) && Float.(r.measured_ms > 0.)) in
        let leg ~exact count =
          List.fold timed ~init:None ~f:(fun acc r ->
              let c = count r in
              if (not (exact r)) || c <= 0 then acc
              else
                let rate = Float.of_int c /. (r.measured_ms *. 1e-3) in
                match acc with
                | Some (best, _) when Float.(best >= rate) -> acc
                | _ -> Some (rate, Printf.sprintf "%s (%s)" r.label r.digest))
        in
        let peak_flops = leg ~exact:(fun r -> not r.flops_approx) (fun r -> r.flops) in
        let peak_bandwidth = leg ~exact:(fun r -> not r.bytes_approx) (fun r -> r.bytes) in
        (* Multi-kernel rows aggregate per-kernel counts, so the per-leg maxima above are
           necessary for them but not sufficient: [summaries_roofline] sums per-kernel
           max-of-legs, which can exceed the aggregate legs (a compute-bound + bandwidth-bound
           kernel mix approaches twice either). The aggregate SUFFICIENT condition is
           [flops/peak_flops + bytes/peak_bandwidth <= time] (max <= sum per kernel), so both
           legs are raised uniformly — preserving their ratio — by the smallest slack that
           enforces it on every multi-kernel row. Raising peaks is the safe direction: the bound
           stays a lower bound, only pruning weakens. With one leg absent the bound degenerates
           to the other leg's aggregate, which the necessary maximum already covers. *)
        let fission_slack =
          match (peak_flops, peak_bandwidth) with
          | Some (pf, _), Some (pb, _) ->
              List.fold timed ~init:None ~f:(fun acc r ->
                  (* Only fully-exact rows can force slack: an over-counted leg would inflate
                     the aggregate sum, and with it both fitted constants. *)
                  if r.kernels <= 1 || r.flops_approx || r.bytes_approx then acc
                  else
                    let t = r.measured_ms *. 1e-3 in
                    let sum = (Float.of_int r.flops /. pf) +. (Float.of_int r.bytes /. pb) in
                    let s = sum /. t in
                    match acc with
                    | Some (best, _) when Float.(best >= s) -> acc
                    | _ when Float.(s <= 1.) -> acc
                    | _ -> Some (s, Printf.sprintf "%s (%s)" r.label r.digest))
          | _ -> None
        in
        let apply_slack =
          match fission_slack with
          | None -> fun p -> p
          | Some (s, _) -> Option.map ~f:(fun (v, w) -> (v *. s, w))
        in
        {
          fit_backend = backend;
          fit_rows = List.length timed;
          fit_opaque = List.count rows ~f:(fun r -> r.opaque);
          fit_flops_approx = List.count timed ~f:(fun r -> r.flops_approx);
          fit_bytes_approx = List.count timed ~f:(fun r -> r.bytes_approx);
          fit_multi_kernel = List.count timed ~f:(fun r -> r.kernels > 1);
          fit_violations =
            (* Fully-exact rows only, matching the runtime warning: on a row with an approx leg
               the exceedance may reflect over-counting, not an understated envelope, and this
               counter prompts a refit. *)
            List.count rows ~f:(fun r ->
                match r.model_ms with
                | Some m ->
                    Float.(m > r.measured_ms)
                    && (not r.flops_approx) && (not r.bytes_approx) && not r.opaque
                | None -> false);
          fit_fission_slack = fission_slack;
          fit_peak_flops = apply_slack peak_flops;
          fit_peak_memory_bandwidth = apply_slack peak_bandwidth;
        })

  let report f =
    let buf = Buffer.create 256 in
    let addf fmt = Printf.ksprintf (Buffer.add_string buf) fmt in
    addf
      "# backend %s: %d timed row%s (%d opaque excluded, %d approx-flops / %d approx-bytes \
       excluded per leg, %d multi-kernel), %d recorded bound violation%s\n"
      f.fit_backend f.fit_rows
      (if f.fit_rows = 1 then "" else "s")
      f.fit_opaque f.fit_flops_approx f.fit_bytes_approx f.fit_multi_kernel f.fit_violations
      (if f.fit_violations = 1 then "" else "s");
    Option.iter f.fit_fission_slack ~f:(fun (s, witness) ->
        addf "# fission slack %s applied to both legs, forced by multi-kernel row: %s\n"
          (Ndarray.concise_float ~prec:6 s) witness);
    let constant ~key ~leg = function
      | None -> addf "# no scoreable row constrains %s\n" key
      | Some (implied, witness) ->
          addf "# %s-leg binding row: %s\n" leg witness;
          (* [concise_float] truncates to [prec + 1] significant digits (relative error below
             [10^-prec] toward zero); the bump keeps the printed constant at or above the implied
             minimum, so the recorded rows satisfy the bound under the printed value too. *)
          addf "%s=%s\n" key (Ndarray.concise_float ~prec:6 (implied *. (1. +. 2e-6)))
    in
    constant ~key:"model_peak_flops" ~leg:"compute" f.fit_peak_flops;
    constant ~key:"model_peak_memory_bandwidth" ~leg:"memory" f.fit_peak_memory_bandwidth;
    Buffer.contents buf
end
