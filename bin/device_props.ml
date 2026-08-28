(* gh-ocannl-684: a supported readback of what the schedule layer's launch gates compare against.

   [Backend_intf.static_properties] (each backend's dump of the properties of all its devices) and
   [Backend_intf.hardware_limits] (the conservative per-workgroup caps derived from them) used to
   have no caller anywhere in the repository, so reading a device's queried limits back meant adding
   a throwaway executable to the tree and deleting it again -- paid twice already, and about to be
   paid a third time verifying gh-ocannl-679's per-dimension workgroup caps on hardware.

   BOTH surfaces are printed, because they are not redundant. What a gate compares against is a
   device query on one backend and an architectural constant on another: HIP queries [max_grid_size]
   for [max_grid_yz], CUDA hardcodes 65535 from the Compute Capability tables, Metal reports [None].
   So the raw props do not tell you what the gate uses, and the derived limits do not tell you
   whether the underlying query answered -- a query returning 0 and a healthy device that simply
   rejects no kernel look identical from either side alone.

   Nothing is compiled and no routine runs: both functions are deliberately [unit ->] so they can
   answer before any driver work, and this tool keeps that property so it stays usable on a machine
   where compilation is what is broken.

   Output is one [path = value] line per fact, so it greps, diffs between machines, and pastes into
   an issue. The lines that matter for the launch gates are [limits.max_threads_per_workgroup] (the
   workgroup's thread PRODUCT), [limits.max_workgroup_dims] (its per-dimension caps -- CUDA's third
   entry is 64, not 1024) and [limits.max_grid_yz], each beside the raw attribute it was derived
   from under [static.].

   Usage: build it and run the executable directly rather than through [dune exec] -- the
   configuration search walks up from the invocation cwd, so [dune exec] finds no [ocannl_config]
   (CLAUDE.md's bin/ cwd trap). Pin the backend explicitly, since with none configured
   [Context.auto] silently walks metal -> cuda -> hip -> cc and would report a different device than
   the one being asked about:

   {v dune build bin/device_props.exe _build/default/bin/device_props.exe --ocannl_backend=metal
   OCANNL_BACKEND=cuda _build/default/bin/device_props.exe v} *)

open Base

(* Flushed per line ([Bench_out]): see gh-ocannl-829 -- a buffered table is invisible until
   the process exits, which on a slow leg reads as a hang. *)
let p fmt = Bench_out.p fmt

(* Flattens a sexp to [path = value] lines. The backends' dumps are [Sexp.message]-shaped -- lists
   of [(key value)] pairs, nested -- so keys carry through as dotted path components; anything else
   (a tuple, a list of tuples) falls back to positional [\[i\]] indices rather than being dropped,
   because a tool whose job is readback must never silently omit a fact it did not anticipate. *)
let rec emit prefix (sexp : Sexp.t) =
  let is_pair = function Sexp.List [ Sexp.Atom _; _ ] -> true | _ -> false in
  let key = function Sexp.List (Sexp.Atom k :: _) -> k | _ -> "?" in
  let pair_value = function Sexp.List [ _; v ] -> v | other -> other in
  let atom = function Sexp.Atom a -> Some a | _ -> None in
  match sexp with
  | Sexp.Atom a -> p "%s = %s\n" prefix a
  (* [None], and the empty list, which must print rather than vanish. *)
  | Sexp.List [] -> p "%s = ()\n" prefix
  (* [Some x] and every other single-element wrapper: the wrapper carries no name of its own. *)
  | Sexp.List [ x ] -> emit prefix x
  | Sexp.List [ Sexp.Atom k; v ] -> emit (prefix ^ "." ^ k) v
  | Sexp.List items when List.for_all items ~f:(fun i -> Option.is_some (atom i)) ->
      (* A tuple or array of scalars: one line, space-separated, so [(1024 1024 64)] reads as the
         one fact it is. *)
      p "%s = %s\n" prefix (String.concat ~sep:" " (List.filter_map items ~f:atom))
  | Sexp.List (Sexp.Atom k :: rest)
    when (not (List.is_empty rest))
         && List.for_all rest ~f:(function Sexp.List _ -> true | _ -> false) ->
      emit (prefix ^ "." ^ k) (Sexp.List rest)
  | Sexp.List items when List.for_all items ~f:is_pair ->
      let keys = List.map items ~f:key in
      let distinct =
        List.length (List.dedup_and_sort keys ~compare:String.compare) = List.length keys
      in
      List.iteri items ~f:(fun i item ->
          let path =
            if distinct then prefix ^ "." ^ key item
            else Printf.sprintf "%s.%s[%d]" prefix (key item) i
          in
          emit path (pair_value item))
  | Sexp.List items ->
      List.iteri items ~f:(fun i item -> emit (Printf.sprintf "%s[%d]" prefix i) item)

(* The per-device entries are read through [Backend_intf.parse_static_properties], the single reader
   of the [static_properties] contract (gh-ocannl-710), so this tool and
   [test/operations/static_properties_contract.ml] cannot drift apart about what a device entry is.
   Indexing the entries explicitly, rather than letting [emit] infer structure, keeps a one-device
   machine's paths identical to a four-device one's -- the whole point of a format meant to be
   diffed across boxes.

   A dump the contract does not cover -- an unlinked backend's [(<name>_missing (error ...))], or a
   future shape -- falls through to the generic flattening, which loses nothing. A readback tool
   inventing structure is worse than one printing a shape it does not recognize: this tool's first
   version accepted any group atom followed by children, and duly reported the [Multidev]
   scheduler's backend-level [(device_name CPU) (num_devices 16)] pairs as two devices, neither of
   which existed. *)
let emit_static (props : Sexp.t) =
  match Ir.Backend_intf.parse_static_properties props with
  | Some { Ir.Backend_intf.group; devices } ->
      p "static.group = %s\n" group;
      p "static.device_count = %d\n" (List.length devices);
      List.iteri devices ~f:(fun i fields ->
          List.iter fields ~f:(fun (key, value) ->
              emit (Printf.sprintf "static.device[%d].%s" i key) value))
  | None -> emit "static" props

let () =
  let ctx = Context.auto () in
  p "backend = %s\n" (Context.backend_name ctx);
  emit_static (Context.static_properties ctx);
  emit "limits" (Ir.Backend_intf.sexp_of_hardware_limits (Context.hardware_limits ctx))
