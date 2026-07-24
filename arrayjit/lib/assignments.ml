open Base
(** The code for operating on n-dimensional arrays. *)

module Lazy = Utils.Lazy
module Tn = Tnode
module Nd = Ndarray

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_ASSIGNMENTS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_ASSIGNMENTS"]

type init_data =
  | Reshape of Ndarray.t
  | Keep_shape_no_padding of Ndarray.t
  | Padded of { data : Nd.t; padding : Ops.axis_padding array; padded_value : float }
[@@deriving sexp_of, equal]

type buffer = Node of Tn.t | Merge_buffer of Tn.t [@@deriving sexp_of, equal]

(** Resets a array by performing the specified computation or data fetching. *)
type fetch_op =
  | Constant of float
  | Constant_bits of int64  (** Direct bit representation, primarily for uint4x32 *)
  | Constant_fill of float array
      (** Fills in the numbers where the rightmost axis is contiguous. Primes shape inference to
          require the assigned tensor to have the same number of elements as the array, but in case
          of "leaky" shape inference, will loop over the values. This unrolls all assignments and
          should be used only for small arrays. Consider using {!Tnode.set_values} instead for
          larger arrays. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell, i.e. how many cells away it is from the
          beginning, in the logical representation of the tensor node. (The actual in-memory
          positions in a buffer instantiating the node can differ.) *)
  | Slice of { batch_idx : Indexing.static_symbol; sliced : Tn.t }
  | Embed_symbol of Indexing.static_symbol
  | Embed_self_id  (** Embeds the id of the [array] field of the [Fetch] constructor. *)
  | Embed_dim of Indexing.variable_ref
[@@deriving sexp_of, equal]

type accum_rhs =
  | Ternop of { op : Ops.ternop; rhs1 : buffer; rhs2 : buffer; rhs3 : buffer }
  | Binop of { op : Ops.binop; rhs1 : buffer; rhs2 : buffer }
  | Unop of { op : Ops.unop; rhs : buffer }
  | Block of { op : Ops.unop; rhses : buffer array }
      (** [Block] and [Rev_sides] are the only assignment types that allow [Concat] axes in
          projections.

          Similar to [Unop] except it's a projection of potentially multiple tensors, e.g.
          concatenation or block tensor; or with just a single RHS, it can be a slice taking part of
          an argument axis or producing part of a result axis. The corresponding [projections] must
          use [Concat] in such a way that every choice of [Concat] components uses at most one of
          the [rhses] (note: none is allowed). Note: it is also allowed that there is no LHS (i.e.
          no valid LHS projection) for some choices of the [Concat] components. *)
  | Rev_sides of { op : Ops.unop; lhses : buffer array }
      (** Causes the [Accum_op] to completely reverse its semantics: left-hand side and right-hand
          side are swapped. [lhs] becomes the read-from tensor and [rhs], i.e. [lhses] above, become
          the written-to tensors. This is needed in particular for gradients of concatenation. *)
[@@deriving sexp_of, equal]

type t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t  (** Same as the given code, with a comment. *)
  | Accum_op of {
      initialize_neutral : bool;
      accum : Ops.binop;
      lhs : Tn.t;
      rhs : accum_rhs;
      projections : Indexing.projections Lazy.t;
      projections_debug : string;
    }
  | Set_vec_unop of {
      op : Ops.vec_unop;
      lhs : Tn.t;
      rhs : buffer;
      projections : Indexing.projections Lazy.t;
      projections_debug : string;
    }
  | Fetch of { array : Tn.t; fetch_op : fetch_op; dims : int array Lazy.t }
[@@deriving sexp_of]

type comp = {
  asgns : t;
  embedded_nodes : Set.M(Tn).t;
      (** The nodes in {!field-asgns} that are not in [embedded_nodes] need to already be in
          contexts linked with the {!comp}. *)
}
[@@deriving sexp_of]
(** Computations based on assignments. Note: the [arrayjit] library makes use of, but does not
    produce nor verify the {!field-embedded_nodes} associated to some given {!field-asgns}. *)

let to_comp asgns = { asgns; embedded_nodes = Set.empty (module Tnode) }
let empty_comp = to_comp Noop

let is_total ~initialize_neutral ~projections =
  initialize_neutral && Affine.is_surjective projections

let can_skip_accumulation ~projections =
  (* We can skip accumulation (use = instead of +=) only if the projection is injective *)
  Affine.is_injective projections

(** Returns materialized nodes in the sense of {!Tnode.Placements.is_in_context_force}, resolved
    against the given compilation lineage's placements. NOTE: it must be called after compilation
    (when the placements of all involved nodes are settled); otherwise, it will disrupt memory mode
    inference. *)
let%debug3_sexp context_nodes ~(plc : Tn.Placements.t) (asgns : t) : Tn.t_set =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let one tn =
    if Tn.Placements.is_in_context_force plc tn 34 then Set.singleton (module Tn) tn else empty
  in
  let of_node = function Node rhs -> one rhs | Merge_buffer _ -> empty in
  let rec loop = function
    | Noop -> empty
    | Seq (t1, t2) -> loop t1 + loop t2
    | Block_comment (_, t) -> loop t
    | Accum_op { lhs; rhs; _ } ->
        let rhses =
          match rhs with
          | Unop { rhs; _ } -> [ of_node rhs ]
          | Binop { rhs1; rhs2; _ } -> [ of_node rhs1; of_node rhs2 ]
          | Ternop { rhs1; rhs2; rhs3; _ } -> [ of_node rhs1; of_node rhs2; of_node rhs3 ]
          | Block { rhses; _ } -> Array.to_list rhses |> List.map ~f:of_node
          | Rev_sides { lhses; _ } -> Array.to_list lhses |> List.map ~f:of_node
        in
        Set.union_list (module Tn) (one lhs :: rhses)
    | Set_vec_unop { lhs; rhs; _ } -> Set.union (one lhs) (of_node rhs)
    (* A slice-alias view's parent must be in context too (it backs the view); the alias itself is
       dropped by [one] via [is_in_context_force] (gh-ocannl-293 293a). *)
    | Fetch { array; fetch_op = Slice { sliced; _ }; _ } -> one array + one sliced
    | Fetch { array; _ } -> one array
  in
  loop asgns

(** In the second set, returns the nodes that are not read from after being written to. In the first
    set, returns the nodes that are ever read from. The second set is also used as the set of nodes
    to materialize; for a [Fetch.Slice] the parent is included there so it is materialized to back a
    potential zero-copy alias view (gh-ocannl-293 293a). *)
let%debug3_sexp collect_nodes_guess_output (asgns : t) : Tn.t_set * Tn.t_set =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let one = Set.singleton (module Tn) in
  let of_node = function Node rhs -> one rhs | Merge_buffer _ -> empty in
  let rec loop = function
    | Noop -> (empty, empty)
    | Seq (t1, t2) ->
        let i1, o1 = loop t1 in
        let i2, o2 = loop t2 in
        (i1 + i2, o1 + o2 - (i1 + i2))
    | Block_comment (_, t) -> loop t
    | Accum_op { lhs; rhs; _ } ->
        let inputs, outputs =
          match rhs with
          | Unop { rhs; _ } -> (of_node rhs, one lhs)
          | Binop { rhs1; rhs2; _ } -> (of_node rhs1 + of_node rhs2, one lhs)
          | Ternop { rhs1; rhs2; rhs3; _ } -> (of_node rhs1 + of_node rhs2 + of_node rhs3, one lhs)
          | Block { rhses; _ } ->
              (Array.fold rhses ~init:empty ~f:(fun acc buf -> acc + of_node buf), one lhs)
          | Rev_sides { lhses; _ } ->
              (one lhs, Array.fold lhses ~init:empty ~f:(fun acc buf -> acc + of_node buf))
        in
        (inputs, outputs)
    | Set_vec_unop { lhs; rhs; _ } -> (of_node rhs, one lhs)
    (* Materialize the slice parent too, so it can back a zero-copy alias view of [array]; harmless
       in the copy-fallback case where the parent is read by the copy loop (gh-ocannl-293 293a). *)
    | Fetch { array; fetch_op = Slice { sliced; _ }; _ } ->
        (empty, Set.of_list (module Tn) [ array; sliced ])
    | Fetch { array; _ } -> (empty, one array)
  in
  loop asgns

(** All nodes that any assignment writes to (unlike the second set of {!collect_nodes_guess_output},
    nodes also read within [asgns] are included). *)
let collect_written (asgns : t) : Tn.t_set =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let one = Set.singleton (module Tn) in
  let rec loop = function
    | Noop -> empty
    | Seq (t1, t2) -> loop t1 + loop t2
    | Block_comment (_, t) -> loop t
    | Accum_op { rhs = Rev_sides { lhses; _ }; _ } ->
        Array.fold lhses ~init:empty ~f:(fun acc buf ->
            match buf with Node rhs -> acc + one rhs | Merge_buffer _ -> acc)
    | Accum_op { lhs; _ } | Set_vec_unop { lhs; _ } -> one lhs
    | Fetch { array; _ } -> one array
  in
  loop asgns

let sequential l =
  Option.value ~default:Noop @@ List.reduce l ~f:(fun sts another_st -> Seq (sts, another_st))

let sequence l =
  Option.value ~default:{ asgns = Noop; embedded_nodes = Set.empty (module Tn) }
  @@ List.reduce l
       ~f:(fun
           { asgns = sts; embedded_nodes = embs } { asgns = another_st; embedded_nodes = emb } ->
         { asgns = Seq (sts, another_st); embedded_nodes = Set.union embs emb })

let collect_neutral_elem (asgns : t) : float option =
  let rec loop acc = function
    | Noop -> acc
    | Seq (t1, t2) -> loop (loop acc t1) t2
    | Block_comment (_, t) -> loop acc t
    | Accum_op { accum; _ } -> (
        let neutral = Ops.neutral_elem accum in
        match acc with
        | None -> Some (Some neutral)
        | Some (Some v) when Float.( = ) v neutral -> acc
        | Some (Some _) -> Some None
        | Some None -> acc)
    | Set_vec_unop _ | Fetch _ -> acc
  in
  match loop None asgns with None -> None | Some v -> v

let%track4_sexp to_low_level ?(static_indices = []) code =
  let open Indexing in
  (* gh-490 symbolic extents: wrap a loop body in [index < value] when the loop iterates a
     symbolic-extent axis AND the extent's symbol is among the routine's bindings (so the value is a
     kernel parameter). An unbound extent symbol keeps the maximum-extent semantics: the loop covers
     the whole (max-sized) buffer, exactly as if the extent were written concretely. *)
  let extent_guard ~(projections : Indexing.projections) ~index ~iter body =
    match List.Assoc.find projections.extent_syms ~equal:Indexing.equal_symbol iter with
    | Some sym when List.mem static_indices sym ~equal:Indexing.equal_static_symbol ->
        let iprec = Ops.index_prec () in
        let cond =
          Low_level.Binop
            ( Ops.Cmplt,
              (Low_level.Embed_index (Indexing.Iterator index), iprec),
              (Low_level.Embed_index (Indexing.Iterator sym.Indexing.static_symbol), iprec) )
        in
        Low_level.If { cond = (cond, iprec); body }
    | _ -> body
  in
  (* Apply left padding offsets to convert from semantic to buffer indices. Semantic indices can be
     negative (e.g., -1 for convolution padding), but buffer indices must be non-negative. Adding
     left_padding converts semantic to buffer space. *)
  let apply_padding_offset (tn : Tn.t) (idcs : Indexing.axis_index array) :
      Indexing.axis_index array =
    match Tn.get_padding tn with
    | None -> idcs
    | Some (padding_arr, _) ->
        Array.mapi idcs ~f:(fun i idx ->
            if i >= Array.length padding_arr then idx
            else
              let left_pad = padding_arr.(i).Ops.left in
              if left_pad = 0 then idx
              else
                match idx with
                | Fixed_idx n -> Fixed_idx (n + left_pad)
                | Iterator s -> Affine { symbols = [ (1, s) ]; offset = left_pad }
                | Affine { symbols; offset } -> Affine { symbols; offset = offset + left_pad }
                | Sub_axis -> Sub_axis
                | Concat _ -> assert false)
  in
  let is_padded tn = Option.is_some (Tn.get_padding tn) in
  (* gh-504 clamped windows: a padded ([=]-mode) max-family window spec registers no margin demand
     (see [Row.solve_proj_equations]'s [clamp_padded]), so its semantic indices can escape the
     operand's valid region; the clamp is rendered here as a range guard on the accumulation
     statement — an out-of-range position contributes the accumulation identity, which is the same
     as not visiting it. [index_interval] bounds a semantic index over the loop ranges of its
     symbols; [None] when a symbol's range is unknown (e.g. a static routine binding), leaving the
     access unguarded (conservative: such accesses are in-range by construction). *)
  let index_interval ~sizes (idx : Indexing.axis_index) : (int * int) option =
    match idx with
    | Indexing.Fixed_idx i -> Some (i, i)
    | Indexing.Iterator s -> Option.map (Map.find sizes s) ~f:(fun d -> (0, d - 1))
    | Indexing.Affine { symbols; offset } ->
        List.fold
          (Indexing.coalesce_affine_terms symbols)
          ~init:(Some (offset, offset))
          ~f:(fun acc (c, s) ->
            match (acc, Map.find sizes s) with
            | Some (lo, hi), Some d ->
                if c >= 0 then Some (lo, hi + (c * (d - 1))) else Some (lo + (c * (d - 1)), hi)
            | _ -> None)
    | Indexing.Sub_axis | Indexing.Concat _ -> None
  in
  let index_plus k (idx : Indexing.axis_index) : Indexing.axis_index =
    match idx with
    | Indexing.Fixed_idx i -> Indexing.Fixed_idx (i + k)
    | Indexing.Iterator s -> Indexing.Affine { symbols = [ (1, s) ]; offset = k }
    | Indexing.Affine { symbols; offset } -> Indexing.Affine { symbols; offset = offset + k }
    | Indexing.Sub_axis | Indexing.Concat _ -> assert false
  in
  (* Range-guard conditions for the axes of an access at (semantic, pre-padding-shift) [idcs] that
     can escape the operand's valid region [0, N) — unless the escape is covered by committed
     margins holding this accumulation's neutral element (the physical-halo mechanism: covered
     reads/writes of the margins are intentional). *)
  let clamp_conds ~accum ~sizes (tn : Tn.t) (idcs : Indexing.axis_index array) :
      Low_level.scalar_arg list =
    let padding = Tn.get_padding tn in
    let sem_dims = Tn.dims_without_padding tn in
    let iprec = Ops.index_prec () in
    let embed idx = (Low_level.Embed_index idx, iprec) in
    Array.to_list
    @@ Array.filter_mapi idcs ~f:(fun i idx ->
        if i >= Array.length sem_dims then None
        else
          match index_interval ~sizes idx with
          | None -> None
          | Some (lo, hi) ->
              let n = sem_dims.(i) in
              if lo >= 0 && hi < n then None
              else
                let covered =
                  match padding with
                  | Some (pads, v) ->
                      i < Array.length pads
                      && Float.(v = Ops.neutral_elem accum)
                      && lo >= -pads.(i).Ops.left
                      && hi < n + pads.(i).Ops.right
                  | None -> false
                in
                if covered then None
                else
                  (* [0 <= idx] as [0 < idx + 1] (cf. the virtualizer's range guards). *)
                  let lower =
                    if lo < 0 then
                      Some
                        (Low_level.Binop
                           (Ops.Cmplt, embed (Indexing.Fixed_idx 0), embed (index_plus 1 idx)))
                    else None
                  in
                  let upper =
                    if hi >= n then
                      Some (Low_level.Binop (Ops.Cmplt, embed idx, embed (Indexing.Fixed_idx n)))
                    else None
                  in
                  (match (lower, upper) with
                  | Some l, Some u ->
                      Some (Low_level.Binop (Ops.And, (l, iprec), (u, iprec)), iprec)
                  | Some c, None | None, Some c -> Some (c, iprec)
                  | None, None -> None))
  in
  let and_all (conds : Low_level.scalar_arg list) : Low_level.scalar_arg =
    List.reduce_exn conds ~f:(fun a b -> (Low_level.Binop (Ops.And, a, b), Ops.index_prec ()))
  in
  (* Redirect a slice-alias view to its parent: a read/write of the alias at [idcs] becomes a
     read/write of the parent at [batch_idx :: idcs] -- exactly the index the materializing copy
     loop used to build for its RHS. Recursive to cover (currently impossible) alias chains. The
     parent is unpadded by alias eligibility, so the downstream padding logic runs correctly against
     it (gh-ocannl-293 293a). *)
  let rec resolve_alias (tn : Tn.t) (idcs : Indexing.axis_index array) :
      Tn.t * Indexing.axis_index array =
    match Tn.alias_of tn with
    | Some (parent, { static_symbol; _ }) ->
        resolve_alias parent (Array.append [| Iterator static_symbol |] idcs)
    | None -> (tn, idcs)
  in
  (* [clamp] (gh-504): when [Some (accum, sizes, conds)], prepend to [conds] the clamp range
     guards of this access, computed on the semantic (pre-padding-shift) indices. *)
  let get ?clamp (buffer : buffer) (idcs : Indexing.axis_index array) : Low_level.scalar_t =
    (* Only [Node] buffers can be slice-alias views; [Merge_buffer] is never redirected. *)
    let buffer, idcs =
      match buffer with
      | Node tn ->
          let parent, idcs = resolve_alias tn idcs in
          (Node parent, idcs)
      | Merge_buffer _ -> (buffer, idcs)
    in
    let tn = match buffer with Node tn -> tn | Merge_buffer tn -> tn in
    let idcs =
      match (idcs, Lazy.force tn.Tn.dims) with
      | [||], [| 1 |] -> [| Fixed_idx 0 |]
      | [| Fixed_idx 0 |], [||] -> idcs
      | idcs, dims when Array.length idcs = Array.length dims -> idcs
      | _ ->
          let dims = Indexing.dims_to_string (Lazy.force tn.Tn.dims) in
          let idcs = Sexp.to_string_hum ([%sexp_of: Indexing.axis_index array] idcs) in
          invalid_arg
            [%string
              "Assignments.to_low_level: indexing mismatch for %{Tn.debug_name tn}: shape %{dims} \
               vs. %{idcs}"]
    in
    Option.iter clamp ~f:(fun (accum, sizes, conds) ->
        conds := clamp_conds ~accum ~sizes tn idcs @ !conds);
    (* The same projection can be used to access a padded or a non-padded tensor. *)
    let idcs = if is_padded tn then apply_padding_offset tn idcs else idcs in
    match buffer with
    | Node tn -> Low_level.Get (tn, idcs)
    | Merge_buffer tn -> Low_level.Get_merge_buffer (tn, idcs)
  in
  let set ?clamp (tn : Tn.t) (idcs : Indexing.axis_index array) (llsc : Low_level.scalar_t) :
      Low_level.t =
    (* Write-through a slice-alias view goes to the parent buffer (gh-ocannl-293 293a). *)
    let tn, idcs = resolve_alias tn idcs in
    let idcs =
      match (idcs, Lazy.force tn.Tn.dims) with
      | [||], [| 1 |] -> [| Fixed_idx 0 |]
      | [| Fixed_idx 0 |], [||] -> idcs
      | idcs, dims when Array.length idcs = Array.length dims -> idcs
      | _ ->
          let dims = Indexing.dims_to_string (Lazy.force tn.Tn.dims) in
          let idcs = Sexp.to_string_hum ([%sexp_of: Indexing.axis_index array] idcs) in
          invalid_arg
            [%string
              "Assignments.to_low_level: indexing mismatch for %{Tn.debug_name tn}: shape %{dims} \
               vs. %{idcs}"]
    in
    Option.iter clamp ~f:(fun (accum, sizes, conds) ->
        conds := clamp_conds ~accum ~sizes tn idcs @ !conds);
    let idcs = if is_padded tn then apply_padding_offset tn idcs else idcs in
    Low_level.Set { tn; idcs; llsc; debug = "" }
  in
  let reset_padding_regions tn neutral_value : Low_level.t list =
    match Tn.get_padding tn with
    | None -> []
    | Some (padding_arr, _) ->
        Low_level.Comment
          ("reset padding margins of " ^ Tnode.debug_name tn ^ " to "
         ^ Float.to_string neutral_value)
        :: [
             Low_level.loop_over_padding_region ~dims:(Lazy.force tn.dims) ~padding:padding_arr
               ~body:(fun idcs ->
                 Low_level.Set
                   {
                     tn;
                     idcs;
                     llsc = Low_level.Constant neutral_value;
                     debug = Tn.debug_name tn ^ " padding := " ^ Float.to_string neutral_value;
                   });
           ]
  in
  let default_padding_before array llc =
    let padding_loops =
      match Tn.get_padding array with Some (_, v) -> reset_padding_regions array v | None -> []
    in
    Low_level.unflat_lines @@ padding_loops @ [ llc ]
  in
  let is_allowed_by_concat ~concat_syms_opt ~block_iters i =
    match concat_syms_opt with
    | None -> true
    | Some syms -> Array.mem ~equal:Indexing.equal_symbol block_iters syms.(i)
  in
  let rec loop_accum ~initialize_neutral ~accum ~(op : Ops.op) ~lhs ~rhses projections : Low_level.t
      =
    let projections : Indexing.projections = Lazy.force projections in
    let all_prod_iters =
      Array.to_list projections.product_iterators
      |> List.concat
      |> Set.of_list (module Indexing.Symbol)
    in
    let iter_sizes =
      Array.fold2_exn projections.product_space projections.product_iterators
        ~init:(Map.empty (module Indexing.Symbol))
        ~f:(fun acc ds its ->
          List.fold2_exn ds its ~init:acc ~f:(fun acc d iter -> Map.set acc ~key:iter ~data:d))
    in
    let concat_offset_for syms active =
      let _, offset =
        List.fold syms ~init:(0, None) ~f:(fun (cumul, found) s ->
            let size =
              match Map.find iter_sizes s with
              | Some v -> v
              | None ->
                  raise
                  @@ Utils.User_error
                       ("concat_offset_for: iterator symbol " ^ Indexing.symbol_ident s
                      ^ " absent from projection iter_sizes; a projection component was dropped")
            in
            if Indexing.equal_symbol s active then (cumul + size, Some cumul)
            else (cumul + size, found))
      in
      Option.value ~default:0 offset
    in
    let basecase block_iters rev_iters =
      (* Create a substitution from product iterators to loop iterators. Fresh loop symbols are
         needed because product_iterators may be shared across different operations/tensors, but
         each lowered operation needs private loop symbols to avoid conflicts in low_level.ml's
         symbol-to-tensor tracking. Concat offsets are computed per Concat index using symbol
         order. *)
      let exception Empty_block in
      let block_iters = Array.of_list_rev block_iters in
      let concat_syms_opt =
        match
          Array.filter_map projections.project_lhs ~f:(function
            | Indexing.Concat syms -> Some syms
            | _ -> None)
        with
        | [| syms |] when List.length syms = Array.length rhses -> Some (Array.of_list syms)
        | _ -> None
      in
      let subst_map =
        let loop_iters = Array.of_list_rev rev_iters in
        Array.map2_exn block_iters loop_iters ~f:(fun block_iter loop_iter ->
            (block_iter, Indexing.Iterator loop_iter))
        |> Array.to_list
        |> Map.of_alist_exn (module Indexing.Symbol)
      in
      (* Substitute in projections - including inside Affine indices *)
      let subst_index = function
        | (Indexing.Fixed_idx _ | Indexing.Sub_axis) as idx -> idx
        | Iterator s
          when Set.mem all_prod_iters s
               && not (Array.mem ~equal:Indexing.equal_symbol block_iters s) ->
            raise Empty_block
        | Indexing.Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
        | Indexing.Affine { symbols; offset } ->
            let symbols =
              List.map symbols ~f:(fun (coeff, s) ->
                  match Map.find subst_map s with
                  | Some (Indexing.Iterator s') -> (coeff, s')
                  | Some (Indexing.Affine _) ->
                      failwith "Affine substitution in Affine index not supported"
                  | Some (Indexing.Concat _) ->
                      failwith "Concat substitution in Affine index not supported"
                  | Some (Indexing.Fixed_idx _) | Some Indexing.Sub_axis | None -> (coeff, s))
            in
            Indexing.Affine { symbols; offset }
        | Indexing.Concat syms -> (
            (* For Block lowering: find the active component (in block_iters) and resolve to it with
               the appropriate offset based on Concat symbol order. *)
            let active =
              List.find_mapi syms ~f:(fun _i s ->
                  if Array.mem ~equal:Indexing.equal_symbol block_iters s then
                    match Map.find subst_map s with
                    | Some (Indexing.Iterator s') ->
                        let offset = concat_offset_for syms s in
                        Some (s', offset)
                    | _ -> None
                  else None)
            in
            match active with
            | Some (s', 0) -> Indexing.Iterator s'
            | Some (s', offset) -> Indexing.Affine { symbols = [ (1, s') ]; offset }
            | None ->
                raise
                @@ Utils.User_error
                     "Concat index could not be resolved to an active component during Block \
                      lowering")
      in
      try
        let lhs_idcs : Indexing.axis_index array =
          Array.map projections.project_lhs ~f:subst_index
        in
        let open Low_level in
        (* gh-504 clamped windows: ranges of the fresh loop symbols, for bounding the semantic
           indices of each access. *)
        let fresh_sizes =
          Map.fold subst_map
            ~init:(Map.empty (module Indexing.Symbol))
            ~f:(fun ~key ~data acc ->
              match (data, Map.find iter_sizes key) with
              | Indexing.Iterator s', Some d -> Map.set acc ~key:s' ~data:d
              | _ -> acc)
        in
        let lhs_conds = ref [] and rhs_conds = ref [] in
        let lhs_ll = get (Node lhs) lhs_idcs in
        let rhses_ll =
          Array.filter_mapi projections.project_rhs ~f:(fun i rhs_idcs ->
              try
                if not (is_allowed_by_concat ~concat_syms_opt ~block_iters i) then None
                else
                  let rhs_idcs = Array.map ~f:subst_index rhs_idcs in
                  Some (get ~clamp:(accum, fresh_sizes, rhs_conds) rhses.(i) rhs_idcs)
              with Empty_block -> None)
        in
        if Array.is_empty rhses_ll then raise Empty_block;
        let rhs2 =
          try apply_op op rhses_ll
          with Invalid_argument _ ->
            raise
            @@ Utils.User_error
                 "Ambiguous indices in concatenation: multiple blocks viable for same position"
        in
        (* Out-of-range reads contribute the accumulation identity: exactly the semantics of a
           padded window spec, so clamping is semantically exact (gh-504). *)
        let rhs2 =
          match !rhs_conds with
          | [] -> rhs2
          | conds ->
              let cond, _ = and_all conds in
              apply_op (Ops.Ternop Ops.Where) [| cond; rhs2; Constant (Ops.neutral_elem accum) |]
        in
        let clamp = (accum, fresh_sizes, lhs_conds) in
        let stmt =
          if initialize_neutral && can_skip_accumulation ~projections then
            set ~clamp lhs lhs_idcs rhs2
          else set ~clamp lhs lhs_idcs @@ apply_op (Ops.Binop accum) [| lhs_ll; rhs2 |]
        in
        (* An out-of-range write target (the transposed clamp of a backward scatter) skips the
           whole statement. *)
        match !lhs_conds with
        | [] -> stmt
        | conds -> Low_level.If { cond = and_all conds; body = stmt }
      with Empty_block -> Low_level.Noop
    in
    let rec for_loop block_iters rev_iters = function
      | [] -> basecase block_iters rev_iters
      | (ds, its) :: product ->
          let index = Indexing.get_symbol () in
          Low_level.unflat_lines
          @@ List.map2_exn ds its ~f:(fun d iter ->
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body =
                    extent_guard ~projections ~index ~iter
                      (for_loop (iter :: block_iters) (index :: rev_iters) product);
                  trace_it = true;
                  axis = Serial;
                })
    in
    let for_loops =
      for_loop [] []
        (Array.to_list @@ Array.zip_exn projections.product_space projections.product_iterators)
    in
    (* Need initialization if: initialize_neutral is true AND (not surjective OR not injective)

       Not surjective: some positions never written (need init to avoid garbage)

       Not injective: accumulation needed (need init for first += operation) *)
    let needs_init =
      initialize_neutral && not (Affine.is_surjective projections && Affine.is_injective projections)
    in
    (* The padding neutral element is part of a padded tensor's identity: margins permanently hold
       the committed value (conflicting margin-touching demands are rejected at shape-inference
       time, and valid-window readers never see the margins), so operands need no resets here.
       Establish the committed neutral element in the lhs margins: only hosted / host-initialized
       buffers are creation-filled, device buffers are allocated raw — so the (idempotent) fill
       accompanies every writer of a padded node. *)
    let neutral_value = Ops.neutral_elem accum in
    let padding_resets =
      match Lazy.force lhs.padding with Some (_, v) -> reset_padding_regions lhs v | None -> []
    in
    let for_loops_with_resets =
      if List.is_empty padding_resets then for_loops
      else Low_level.unflat_lines (padding_resets @ [ for_loops ])
    in
    if needs_init then
      let dims = lazy projections.lhs_dims in
      let fetch_op = Constant neutral_value in
      Low_level.Seq (loop (Fetch { array = lhs; fetch_op; dims }), for_loops_with_resets)
    else for_loops_with_resets
  and loop_accum_rev ~initialize_neutral ~accum ~(op : Ops.op) ~lhs ~lhses projections : Low_level.t
      =
    let projections : Indexing.projections = Lazy.force projections in
    let all_prod_iters =
      Array.to_list projections.product_iterators
      |> List.concat
      |> Set.of_list (module Indexing.Symbol)
    in
    let target_projections =
      Array.mapi projections.project_rhs ~f:(fun i project_lhs ->
          { projections with lhs_dims = projections.rhs_dims.(i); project_lhs })
    in
    let target_can_skip =
      Array.map target_projections ~f:(fun proj -> can_skip_accumulation ~projections:proj)
    in
    let target_needs_init =
      Array.map target_projections ~f:(fun proj ->
          initialize_neutral && not (Affine.is_surjective proj && Affine.is_injective proj))
    in
    let iter_sizes =
      Array.fold2_exn projections.product_space projections.product_iterators
        ~init:(Map.empty (module Indexing.Symbol))
        ~f:(fun acc ds its ->
          List.fold2_exn ds its ~init:acc ~f:(fun acc d iter -> Map.set acc ~key:iter ~data:d))
    in
    let concat_offset_for syms active =
      let _, offset =
        List.fold syms ~init:(0, None) ~f:(fun (cumul, found) s ->
            let size =
              match Map.find iter_sizes s with
              | Some v -> v
              | None ->
                  raise
                  @@ Utils.User_error
                       ("concat_offset_for: iterator symbol " ^ Indexing.symbol_ident s
                      ^ " absent from projection iter_sizes; a projection component was dropped")
            in
            if Indexing.equal_symbol s active then (cumul + size, Some cumul)
            else (cumul + size, found))
      in
      Option.value ~default:0 offset
    in
    let basecase block_iters rev_iters =
      let exception Empty_block in
      let block_iters = Array.of_list_rev block_iters in
      let concat_syms_opt =
        match
          Array.filter_map projections.project_lhs ~f:(function
            | Indexing.Concat syms -> Some syms
            | _ -> None)
        with
        | [| syms |] when List.length syms = Array.length lhses -> Some (Array.of_list syms)
        | _ -> None
      in
      let subst_map =
        let loop_iters = Array.of_list_rev rev_iters in
        Array.map2_exn block_iters loop_iters ~f:(fun block_iter loop_iter ->
            (block_iter, Indexing.Iterator loop_iter))
        |> Array.to_list
        |> Map.of_alist_exn (module Indexing.Symbol)
      in
      let subst_index = function
        | (Indexing.Fixed_idx _ | Indexing.Sub_axis) as idx -> idx
        | Iterator s
          when Set.mem all_prod_iters s
               && not (Array.mem ~equal:Indexing.equal_symbol block_iters s) ->
            raise Empty_block
        | Indexing.Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
        | Indexing.Affine { symbols; offset } ->
            let symbols =
              List.map symbols ~f:(fun (coeff, s) ->
                  match Map.find subst_map s with
                  | Some (Indexing.Iterator s') -> (coeff, s')
                  | Some (Indexing.Affine _) ->
                      failwith "Affine substitution in Affine index not supported"
                  | Some (Indexing.Concat _) ->
                      failwith "Concat substitution in Affine index not supported"
                  | Some (Indexing.Fixed_idx _) | Some Indexing.Sub_axis | None -> (coeff, s))
            in
            Indexing.Affine { symbols; offset }
        | Indexing.Concat syms -> (
            (* For Rev_sides lowering: find the active component and resolve with offset *)
            let active =
              List.find_mapi syms ~f:(fun _i s ->
                  if Array.mem ~equal:Indexing.equal_symbol block_iters s then
                    match Map.find subst_map s with
                    | Some (Indexing.Iterator s') ->
                        let offset = concat_offset_for syms s in
                        Some (s', offset)
                    | _ -> None
                  else None)
            in
            match active with
            | Some (s', 0) -> Indexing.Iterator s'
            | Some (s', offset) -> Indexing.Affine { symbols = [ (1, s') ]; offset }
            | None ->
                raise
                @@ Utils.User_error
                     "Concat index could not be resolved to an active component during Rev_sides \
                      lowering")
      in
      let target_tn_exn = function
        | Node tn -> tn
        | Merge_buffer _ -> raise @@ Utils.User_error "Rev_sides cannot write to merge buffers"
      in
      try
        let rhs_idcs : Indexing.axis_index array =
          Array.map projections.project_lhs ~f:subst_index
        in
        let open Low_level in
        (* gh-504 clamped windows: see [loop_accum]'s basecase. *)
        let fresh_sizes =
          Map.fold subst_map
            ~init:(Map.empty (module Indexing.Symbol))
            ~f:(fun ~key ~data acc ->
              match (data, Map.find iter_sizes key) with
              | Indexing.Iterator s', Some d -> Map.set acc ~key:s' ~data:d
              | _ -> acc)
        in
        let lhs_conds = ref [] and rhs_conds = ref [] in
        let rhs_ll = get ~clamp:(accum, fresh_sizes, rhs_conds) (Node lhs) rhs_idcs in
        let targets =
          Array.filter_mapi projections.project_rhs ~f:(fun i lhs_idcs ->
              try
                if not (is_allowed_by_concat ~concat_syms_opt ~block_iters i) then None
                else
                  let lhs_idcs = Array.map ~f:subst_index lhs_idcs in
                  Some (i, lhses.(i), lhs_idcs)
              with Empty_block -> None)
        in
        if Array.is_empty targets then raise Empty_block;
        if Array.length targets > 1 then
          raise
          @@ Utils.User_error
               "Ambiguous indices in concatenation: multiple blocks viable for same position";
        let i, target_buf, lhs_idcs = targets.(0) in
        let rhs2 = apply_op op [| rhs_ll |] in
        let rhs2 =
          match !rhs_conds with
          | [] -> rhs2
          | conds ->
              let cond, _ = and_all conds in
              apply_op (Ops.Ternop Ops.Where) [| cond; rhs2; Constant (Ops.neutral_elem accum) |]
        in
        let target_tn = target_tn_exn target_buf in
        let clamp = (accum, fresh_sizes, lhs_conds) in
        let stmt =
          if initialize_neutral && target_can_skip.(i) then set ~clamp target_tn lhs_idcs rhs2
          else
            set ~clamp target_tn lhs_idcs
            @@ apply_op (Ops.Binop accum) [| get target_buf lhs_idcs; rhs2 |]
        in
        match !lhs_conds with
        | [] -> stmt
        | conds -> Low_level.If { cond = and_all conds; body = stmt }
      with Empty_block -> Low_level.Noop
    in
    let rec for_loop block_iters rev_iters = function
      | [] -> basecase block_iters rev_iters
      | (ds, its) :: product ->
          let index = Indexing.get_symbol () in
          Low_level.unflat_lines
          @@ List.map2_exn ds its ~f:(fun d iter ->
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body =
                    extent_guard ~projections ~index ~iter
                      (for_loop (iter :: block_iters) (index :: rev_iters) product);
                  trace_it = true;
                  axis = Serial;
                })
    in
    let for_loops =
      for_loop [] []
        (Array.to_list @@ Array.zip_exn projections.product_space projections.product_iterators)
    in
    let neutral_value = Ops.neutral_elem accum in
    (* Establish the committed neutral element in the lhs margins (device buffers are allocated
       raw). *)
    let padding_resets =
      match Lazy.force lhs.padding with Some (_, v) -> reset_padding_regions lhs v | None -> []
    in
    let for_loops_with_resets =
      if List.is_empty padding_resets then for_loops
      else Low_level.unflat_lines (padding_resets @ [ for_loops ])
    in
    let init_ops =
      Array.filter_mapi lhses ~f:(fun i buf ->
          if not target_needs_init.(i) then None
          else
            let array =
              match buf with
              | Node tn -> tn
              | Merge_buffer _ ->
                  raise @@ Utils.User_error "Rev_sides cannot initialize merge buffers"
            in
            Some
              (Fetch
                 { array; fetch_op = Constant neutral_value; dims = lazy projections.rhs_dims.(i) }))
      |> Array.to_list
    in
    if List.is_empty init_ops then for_loops_with_resets
    else Low_level.unflat_lines (List.map init_ops ~f:loop @ [ for_loops_with_resets ])
  and loop (code : t) : Low_level.t =
    match code with
    | Accum_op { initialize_neutral; accum; lhs; rhs; projections; _ } -> (
        let op, rhses =
          match rhs with
          | Unop { op; rhs } -> (Ops.Unop op, [| rhs |])
          | Binop { op; rhs1; rhs2 } -> (Ops.Binop op, [| rhs1; rhs2 |])
          | Ternop { op; rhs1; rhs2; rhs3 } -> (Ops.Ternop op, [| rhs1; rhs2; rhs3 |])
          | Block { op; rhses } -> (Ops.Unop op, rhses)
          | Rev_sides { op; lhses } -> (Ops.Unop op, lhses)
        in
        match rhs with
        | Rev_sides _ -> loop_accum_rev ~initialize_neutral ~accum ~op ~lhs ~lhses:rhses projections
        | _ -> loop_accum ~initialize_neutral ~accum ~op ~lhs ~rhses projections)
    | Set_vec_unop { op; lhs; rhs; projections; _ } ->
        (* Handle vector unary operations *)
        let projections = Lazy.force projections in
        let full_length =
          match op with
          | Ops.Uint4x32_to_prec_uniform ->
              (* Prevent over-eager guard against forcing precision. *)
              ignore (Lazy.force lhs.dims);
              Ops.vec_unop_lanes (Lazy.force lhs.storage_prec)
        in
        (* [Set_from_vec] stores [length] lanes at flat consecutive offsets; a padded (halo) target
           breaks that assumption across rows, and would make the random stream layout-dependent.
           Reject explicitly with a remedy. *)
        (match Tn.get_padding lhs with
        | None -> ()
        | Some (pads, _) when Array.for_all pads ~f:(fun p -> p.Ops.left = 0 && p.Ops.right = 0) ->
            ()
        | Some _ ->
            raise
            @@ Utils.User_error
                 [%string
                   "Set_vec_unop (packed uniform): target %{Tn.debug_name lhs} is padded; \
                    materialize the random tensor into an unpadded node and copy it into the \
                    padded one instead"]);
        (* Tail peel: when the target's total element count is not a multiple of [full_length], the
           final counter iteration stores only the remaining lanes of its 128-bit block. *)
        let total_elems = Tn.num_elems lhs in
        let rem = total_elems % full_length in
        let basecase ~length rev_iters =
          let subst_map =
            let loop_iters = Array.of_list_rev rev_iters in
            Array.map2_exn loop_iters projections.product_iterators ~f:(fun loop_iter prod_iter ->
                let prod_iter =
                  match prod_iter with
                  | [ prod_iter ] -> prod_iter
                  | _ -> raise @@ Utils.User_error "Concat indexing not supported in Set_vec_unop"
                in
                (prod_iter, Indexing.Iterator loop_iter))
            |> Array.to_list
            |> Map.of_alist_exn (module Indexing.Symbol)
          in
          let subst_index = function
            | Indexing.Concat _ ->
                raise @@ Utils.User_error "Concat indexing not supported in Set_vec_unop"
            | (Fixed_idx _ | Sub_axis) as idx -> idx
            | Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
            | Affine { symbols; offset } ->
                (* Substitute symbols in affine index *)
                let subst_symbols =
                  List.map symbols ~f:(fun (coeff, s) ->
                      match Map.find subst_map s with
                      | Some (Indexing.Iterator new_s) -> (coeff, new_s)
                      | _ -> (coeff, s))
                in
                Indexing.Affine { symbols = subst_symbols; offset }
          in
          let lhs_idcs = Array.map projections.project_lhs ~f:subst_index in
          let rhs_idcs = Array.map projections.project_rhs.(0) ~f:subst_index in
          let open Low_level in
          let rhs_ll = get rhs rhs_idcs in
          (* Redirect a vector store through a slice-alias view to the parent, mirroring [set] for
             scalar stores (gh-ocannl-293 293a). Without this the alias [lhs] -- which owns no
             buffer and is excluded from [ctx_buffers] -- would be a write target the backend cannot
             link. The parent is unpadded by alias eligibility, and its precision matches the
             slice's, so [length] (computed from [lhs.storage_prec] above) stays correct. *)
          let lhs, lhs_idcs = resolve_alias lhs lhs_idcs in
          Set_from_vec
            {
              tn = lhs;
              idcs = lhs_idcs;
              length;
              vec_unop = op;
              arg = (rhs_ll, Low_level.scalar_precision rhs_ll);
              debug = "";
            }
        in
        let peel_axis =
          if rem = 0 then -1
          else if total_elems <= full_length then
            (* A single (partial) block: the counter axis is dim-1 and typically has no iterator in
               the projections; every store is the tail store (handled by starting in tail mode
               below). *)
            -1
          else begin
            (* The counter operand is a single axis: exactly one product axis drives the argument's
               projection, and it is the one to peel. *)
            let rhs_symbols =
              Array.to_list projections.project_rhs.(0)
              |> List.concat_map ~f:(function
                | Indexing.Iterator s -> [ s ]
                | Indexing.Affine { symbols; _ } -> List.map symbols ~f:snd
                | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> [])
            in
            let driving =
              Array.filter_mapi projections.product_iterators ~f:(fun i prod_iter ->
                  match prod_iter with
                  | [ s ] when List.mem rhs_symbols s ~equal:Indexing.Symbol.equal -> Some i
                  | _ -> None)
            in
            match Array.to_list driving with
            | [ i ] ->
                (* The peel below assumes the flat store offset is driven solely by the peeled
                   counter axis: every other product axis must be degenerate. *)
                let others_product =
                  Array.foldi projections.product_space ~init:1 ~f:(fun j acc d ->
                      if j = i then acc else List.fold d ~init:acc ~f:( * ))
                in
                if others_product <> 1 then
                  raise
                  @@ Utils.User_error
                       [%string
                         "Set_vec_unop (packed uniform): non-divisible target size \
                          %{total_elems#Int} (lanes per block: %{full_length#Int}) requires a 1-D \
                          counter iteration"]
                else i
            | _ ->
                raise
                @@ Utils.User_error
                     [%string
                       "Set_vec_unop (packed uniform): non-divisible target size \
                        %{total_elems#Int} (lanes per block: %{full_length#Int}) but could not \
                        identify the counter axis to peel"]
          end
        in
        let rec for_loop ~tail rev_iters = function
          | [] -> basecase ~length:(if tail then rem else full_length) rev_iters
          | (i, [ d ]) :: product when i = peel_axis ->
              (* Peel the final counter iteration: interior stores stay full-width and guard-free,
                 the last store writes the [rem] remaining lanes. *)
              let make ~tail ~from_ ~to_ =
                let index = Indexing.get_symbol () in
                Low_level.For_loop
                  {
                    index;
                    from_;
                    to_;
                    body = for_loop ~tail (index :: rev_iters) product;
                    trace_it = true;
                    axis = Serial;
                  }
              in
              let tail_loop = make ~tail:true ~from_:(d - 1) ~to_:(d - 1) in
              if d = 1 then tail_loop
              else Low_level.Seq (make ~tail:false ~from_:0 ~to_:(d - 2), tail_loop)
          | (_, [ d ]) :: product ->
              let index = Indexing.get_symbol () in
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body = for_loop ~tail (index :: rev_iters) product;
                  trace_it = true;
                  axis = Serial;
                }
          | _ -> raise @@ Utils.User_error "Concat indexing not supported in Set_vec_unop"
        in
        for_loop
          ~tail:(rem <> 0 && total_elems <= full_length)
          []
          (List.mapi ~f:(fun i d -> (i, d)) (Array.to_list projections.product_space))
    | Noop -> Low_level.Noop
    | Block_comment (s, c) -> Low_level.unflat_lines [ Comment s; loop c; Comment "end" ]
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Low_level.Seq (c1, c2)
    | Fetch { array; fetch_op = Constant 0.0; dims = _ } ->
        (* [Zero_out] covers the whole buffer including the margins, so a nonzero committed neutral
           element must be re-established after it (a zero neutral is already correct). *)
        let padding_after =
          match Tn.get_padding array with
          | Some (_, v) when Float.( <> ) v 0.0 -> reset_padding_regions array v
          | _ -> []
        in
        Low_level.unflat_lines (Low_level.Zero_out array :: padding_after)
    | Fetch { array; fetch_op = Constant c; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant c)
    | Fetch { array; fetch_op = Constant_bits i; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant_bits i)
    | Fetch { array; fetch_op = Slice { batch_idx = { static_symbol = idx; _ }; sliced }; dims } ->
        if Tn.is_alias array then
          (* Zero-copy alias view: no materialization. Reads/writes of [array] are redirected to
             [sliced] (with [idx] prepended) by [get]/[set] (gh-ocannl-293 293a). *)
          Low_level.Noop
        else
          default_padding_before array
          @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
              set array idcs @@ get (Node sliced) @@ Array.append [| Iterator idx |] idcs)
    | Fetch { array; fetch_op = Embed_symbol s; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Embed_index (Iterator s.static_symbol))
    | Fetch { array; fetch_op = Embed_self_id; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant_bits (Int64.of_int array.id))
    | Fetch { array; fetch_op = Embed_dim variable_ref; dims } ->
        (* Forcing [dims] first forces shape inference to complete, which is what fills in
           [variable_ref]: it must happen before reading the solved dimension (this fetch may be the
           first statement lowered in the routine). *)
        let dims = Lazy.force dims in
        let dim_value =
          match (variable_ref.Indexing.solved_dim, variable_ref.Indexing.solved_sym) with
          | Some d, _ -> d
          | None, Some { Indexing.static_range = Some range; _ } ->
              (* A symbolic extent (gh-490) materializes at its declared maximum. *)
              range
          | None, Some { Indexing.static_range = None; _ } | None, None ->
              raise
              @@ Utils.User_error
                   ("Embed_dim: variable reference " ^ variable_ref.Indexing.ref_label
                  ^ " has no solved dimension")
        in
        default_padding_before array
        @@ Low_level.loop_over_dims dims ~body:(fun idcs ->
            set array idcs @@ Constant (Float.of_int dim_value))
    | Fetch { array; fetch_op = Range_over_offsets; dims = (lazy dims) } ->
        default_padding_before array
        @@ Low_level.loop_over_dims dims ~body:(fun idcs ->
            let offset = Indexing.reflect_projection ~dims ~projection:idcs in
            set array idcs @@ Embed_index offset)
    | Fetch { array; fetch_op = Constant_fill values; dims = (lazy dims) } ->
        (* TODO: consider failing here and strengthening shape inference. *)
        let size = Array.length values in
        let limit_constant_fill_size =
          Int.of_string @@ Utils.get_global_arg ~default:"16" ~arg_name:"limit_constant_fill_size"
        in
        if size > limit_constant_fill_size then
          raise
          @@ Utils.User_error
               [%string
                 "Constant_fill size is too large to unroll for %{Tn.debug_name array} (size: \
                  %{size#Int}, limit: %{limit_constant_fill_size#Int}), either increase \
                  ocannl_limit_constant_fill_size or use Tnode.set_values instead"];
        default_padding_before array
        @@ Low_level.unroll_dims dims ~body:(fun idcs ~offset ->
            set array idcs @@ Constant values.(offset % size))
  in
  (* Pre-pass: mark alias-eligible [Fetch.Slice]s before lowering, so [get]/[set] redirect them and
     the [Slice] lowering emits no copy loop. Eligibility needs forced shapes, which are available
     here (shape inference is forced before [lower]/[to_low_level]). Idempotent across re-lowerings.
     A slice falls back to the materializing copy loop unless ALL hold: - leading-axis,
     rank-drop-by-one (the only shape [Slice] produces): parent rank = child rank + 1 and the
     trailing dims match elementwise; - parent and child share the same precision: the copy loop
     silently converts precision (e.g. a float buffer sliced then reinterpreted as uint4x32), which
     a shared-storage alias cannot do; - the parent has backing storage (not [Virtual] /
     [Effectively_constant]); - parent and child are both unpadded (aliasing would otherwise break
     the padding contract). (gh-ocannl-293 subtask 293a.) *)
  let slice_alias_eligible ~(array : Tn.t) ~(sliced : Tn.t) : bool =
    let pdims = Lazy.force sliced.Tn.dims and cdims = Lazy.force array.Tn.dims in
    Array.length pdims = Array.length cdims + 1
    && Array.equal Int.equal (Array.subo pdims ~pos:1) cdims
    && Ops.equal_prec (Lazy.force array.Tn.storage_prec) (Lazy.force sliced.Tn.storage_prec)
    (* Alias-ness is a semantic fact settled deterministically at assignments lowering, BEFORE
       per-lineage placement decisions diverge (context-scoped memory modes, category 1). So
       eligibility may consult only lineage-independent facts -- shapes, precision, padding, and the
       parent's DECLARED INTENT -- never a lineage's placements: the alias mark is cached globally
       on the tnode, and a lineage-dependent eligibility input would let one lineage's alias
       redirect accesses to a parent that another lineage virtualized (PR #93 review). Conversely,
       confirming an alias declares [On_device] intent on the parent (see [mark_aliases]), so no
       lineage can resolve the backing buffer away from under the view. *)
    && (not (Tn.known_virtual sliced))
    && (not (Tn.known_constant sliced))
    && Option.is_none (Tn.get_padding sliced)
    && Option.is_none (Tn.get_padding array)
  in
  let rec mark_aliases (c : t) : unit =
    match c with
    | Noop -> ()
    | Seq (c1, c2) ->
        mark_aliases c1;
        mark_aliases c2
    | Block_comment (_, c) -> mark_aliases c
    | Fetch { array; fetch_op = Slice { batch_idx; sliced }; dims = _ } ->
        if slice_alias_eligible ~array ~sliced then (
          (* The view's write semantics (a write through [array] is a write to [sliced]'s sub-range,
             potentially observed by a later routine) require the parent to own a persistent buffer
             in EVERY lineage that lowers this alias. Declare the intent globally, like the alias
             mark itself -- monotone and idempotent; mirrors [collect_nodes_guess_output]'s
             materialization of slice parents. Provenance 27. *)
          Tn.update_memory_mode sliced On_device 27;
          Tn.set_alias_of array ~parent:sliced ~batch_idx)
    | Fetch _ | Accum_op _ | Set_vec_unop _ -> ()
  in
  mark_aliases code;
  loop code

let flatten c =
  let rec loop = function
    | Noop -> []
    | Seq (c1, c2) -> loop c1 @ loop c2
    | Block_comment (s, c) -> Block_comment (s, Noop) :: loop c
    | (Accum_op _ | Set_vec_unop _ | Fetch _) as c -> [ c ]
  in
  loop c

let is_noop c =
  List.for_all ~f:(function Noop | Block_comment (_, Noop) -> true | _ -> false) @@ flatten c

let get_ident_within_code ?no_dots c =
  let ident_style = Tn.get_style ~arg_name:"cd_ident_style" ?no_dots () in
  let nograd_idents = Hashtbl.create (module String) in
  let grad_idents = Hashtbl.create (module String) in
  let visit tn =
    let is_grad, ident = Tn.no_grad_ident_label tn in
    let idents = if is_grad then grad_idents else nograd_idents in
    Option.iter ident
      ~f:
        (Hashtbl.update idents ~f:(fun old ->
             Set.add (Option.value ~default:Utils.no_ints old) tn.uid))
  in
  let tn = function Node tn -> tn | Merge_buffer tn -> tn in
  let rec loop (c : t) =
    match c with
    | Noop -> ()
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | Block_comment (_, c) -> loop c
    | Accum_op { lhs; rhs; _ } ->
        let rhses =
          match rhs with
          | Unop { rhs; _ } -> [ tn rhs ]
          | Binop { rhs1; rhs2; _ } -> [ tn rhs1; tn rhs2 ]
          | Ternop { rhs1; rhs2; rhs3; _ } -> [ tn rhs1; tn rhs2; tn rhs3 ]
          | Block { rhses; _ } -> Array.to_list rhses |> List.map ~f:tn
          | Rev_sides { lhses; _ } -> Array.to_list lhses |> List.map ~f:tn
        in
        List.iter ~f:visit (lhs :: rhses)
    | Set_vec_unop { op = _; lhs; rhs; projections = _; projections_debug = _ } ->
        List.iter ~f:visit [ lhs; tn rhs ]
    | Fetch { array; fetch_op = _; dims = _ } -> visit array
  in
  loop c;
  let repeating_nograd_idents =
    Hashtbl.filter nograd_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  let repeating_grad_idents =
    Hashtbl.filter grad_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  fun tn ->
    let ident = Tn.styled_ident ~repeating_nograd_idents ~repeating_grad_idents ident_style tn in
    Tn.update_code_name tn ident;
    ident

let to_doc ?name ?static_indices () c =
  let ident = get_ident_within_code c in
  let buffer_ident = function Node tn -> ident tn | Merge_buffer tn -> ident tn ^ ".merge" in

  let open PPrint in
  let doc_of_fetch_op (op : fetch_op) =
    match op with
    | Constant f -> string (Float.to_string f)
    | Constant_bits i -> string (Printf.sprintf "bits(%LdLL)" i)
    | Constant_fill values ->
        let values_str =
          String.concat ~sep:", " (Array.to_list (Array.map values ~f:Float.to_string))
        in
        string ("constant_fill([" ^ values_str ^ "])")
    | Range_over_offsets -> string "range_over_offsets()"
    | Slice { batch_idx; sliced } ->
        string (ident sliced ^ " @| " ^ Indexing.symbol_ident batch_idx.static_symbol)
    | Embed_symbol { static_symbol; static_range = _; used_as_extent = _; used_as_slice = _ } ->
        string ("!@" ^ Indexing.symbol_ident static_symbol)
    | Embed_self_id -> string "self_id()"
    | Embed_dim { ref_label; _ } -> string ("(dim " ^ ref_label ^ ")")
  in

  let rec doc_of_code = function
    | Noop -> empty
    | Seq (c1, c2) -> doc_of_code c1 ^^ doc_of_code c2
    | Block_comment (s, Noop) -> string ("# \"" ^ s ^ "\";") ^^ break 1
    | Block_comment (s, c) -> string ("# \"" ^ s ^ "\";") ^^ break 1 ^^ doc_of_code c
    | Accum_op { initialize_neutral; accum; lhs; rhs; projections_debug; _ } -> (
        let proj_spec = projections_debug in
        match rhs with
        | Ternop { op; rhs1; rhs2; rhs3 } ->
            (* Uncurried syntax for ternary operations. *)
            string (ident lhs)
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ string (Ops.ternop_cd_syntax op)
            ^^ string "("
            ^^ string (buffer_ident rhs1)
            ^^ string ", "
            ^^ string (buffer_ident rhs2)
            ^^ string ", "
            ^^ string (buffer_ident rhs3)
            ^^ string ")"
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Binop { op; rhs1; rhs2 } ->
            string (ident lhs)
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ string (buffer_ident rhs1)
            ^^ space
            ^^ string (Ops.binop_cd_syntax op)
            ^^ space
            ^^ string (buffer_ident rhs2)
            ^^ (if
                  (not (String.equal proj_spec "."))
                  || List.mem ~equal:Ops.equal_binop Ops.[ Mul; Div ] op
                then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Unop { op; rhs } ->
            string (ident lhs)
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ (if not @@ Ops.equal_unop op Ops.Identity then string (Ops.unop_cd_syntax op ^ " ")
                else empty)
            ^^ string (buffer_ident rhs)
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Block { op; rhses } ->
            (* TODO: Pretty-print Block operations *)
            string (ident lhs)
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ (if not @@ Ops.equal_unop op Ops.Identity then string (Ops.unop_cd_syntax op ^ " ")
                else empty)
            ^^ brackets
                 (separate (semi ^^ space)
                    (Array.to_list (Array.map rhses ~f:(Fn.compose string buffer_ident))))
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Rev_sides { op; lhses } ->
            brackets
              (separate (semi ^^ space)
                 (Array.to_list (Array.map lhses ~f:(Fn.compose string buffer_ident))))
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ (if not @@ Ops.equal_unop op Ops.Identity then string (Ops.unop_cd_syntax op ^ " ")
                else empty)
            ^^ string (ident lhs)
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1)
    | Set_vec_unop { op; lhs; rhs; projections = _; projections_debug } ->
        let proj_spec = projections_debug in
        string (ident lhs)
        ^^ space
        ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral:false Arg2)
        ^^ space
        ^^ string (Ops.vec_unop_cd_syntax op)
        ^^ space
        ^^ string (buffer_ident rhs)
        ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
            else empty)
        ^^ string ";" ^^ break 1
    | Fetch { array; fetch_op; dims = _ } ->
        string (ident array) ^^ string " =: " ^^ doc_of_fetch_op fetch_op ^^ string ";" ^^ break 1
  in

  (* Create the header document *)
  let header_doc =
    match (name, static_indices) with
    | Some n, Some si ->
        string (n ^ " (")
        ^^ separate (comma ^^ space) (List.map si ~f:Indexing.Doc_helpers.pp_static_symbol)
        ^^ string "):" ^^ space
    | Some n, None -> string (n ^ ":") ^^ space
    | _ -> empty
  in

  header_doc ^^ nest 2 (doc_of_code c)

let to_string c =
  let doc = to_doc () c in
  let b = Buffer.create 100 in
  PPrint.ToBuffer.pretty 0.7 100 b doc;
  Buffer.contents b

let get_name_exn asgns =
  let punct_or_sp = Str.regexp "[-@*/:.;, ]" in
  let punct_and_sp = Str.regexp {|[-@*/:.;,]\( |$\)|} in
  let rec loop = function
    | Block_comment (s, _) ->
        Str.global_replace punct_and_sp "" s |> Str.global_replace punct_or_sp "_"
    | Seq (t1, t2) ->
        let n1 = loop t1 and n2 = loop t2 in
        let prefix = String.common_prefix2_length n1 n2 in
        let suffix = String.common_suffix2_length n1 n2 in
        if String.is_empty n1 || String.is_empty n2 then n1 ^ n2
        else String.drop_suffix n1 suffix ^ "_then_" ^ String.drop_prefix n2 prefix
    | _ -> ""
  in
  let result = loop asgns in
  if String.is_empty result then
    invalid_arg ("Assignments.get_name_exn: no comments in code: " ^ to_string asgns)
  else result

let%track6_sexp lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name static_indices
    (proc : t) : Low_level.optimized =
  (match cd_source with
  | None -> ()
  | Some callback -> callback (to_doc ~name ~static_indices () proc));
  let llc : Low_level.t = to_low_level ~static_indices proc in
  Low_level.optimize optim_ctx ~unoptim_ll_source ~ll_source ~name static_indices llc
