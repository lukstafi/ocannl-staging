(** What [ocannl_read_config] writes, and what each value it writes MEANS (gh-ocannl-1581).

    The reader resolves a configuration key the way a test run will and writes the answer to
    [ocannl_<key>.txt], which rules elsewhere read back through [%{read:…}]: a per-backend golden's
    name, an emitted-code artifact's suffix. Its decisions about the backend live here rather than
    in its executable, so that a scan reading such a pform back can say which backend a value stands
    for by asking the code that wrote it: [cu] means CUDA because {!written} says a CUDA build
    writes [cu] to [ocannl_backend_extension.txt], not because a second table in the scan agrees
    with the reader today. The inverse ({!backends_meant}) is that same function run over every
    backend. *)

open Base

(** The file [--read=<key>] writes, in the working directory of the rule running it. *)
let file_of_key key = "ocannl_" ^ key ^ ".txt"

(** The deprecated spellings the backend selector still accepts, each with the backend it names: the
    entries of [Backends.accepted_names] that are not a backend's own name.
    [test/operations/marker_backend_vocabulary] holds the two equal, so an alias added to the parse
    table alone fails there rather than resolving here to a name no backend has. Restated rather
    than read because this library links [base] alone, for the tests and scans that link no backend.
*)
let deprecated_aliases = [ ("sync_cc", "cc"); ("multicore_cc", "multidev_cc") ]

(** A backend setting as the backend's own name: lower case, a deprecated alias resolved. A stream
    or log file is named after the canonical backend (e.g.
    [log_files/micrograd_demo_logging/cc-0-0.log]), so an alias-spelled run must not leak the raw
    string into the paths dune reads. *)
let canonical_backend setting =
  let name = String.lowercase setting in
  Option.value (List.Assoc.find deprecated_aliases name ~equal:String.equal) ~default:name

(** The extension a backend's emitted code carries, without the dot, for a canonical backend name;
    [None] for a name this reader was never taught. [marker_backend_vocabulary] holds it equal, for
    every backend OCANNL has, to the [source_extension] fact the backend's artifact writer appends
    ([Ir.Backend_intf.backend_facts]); restated for the same reason as {!deprecated_aliases}. *)
let code_extension = function
  | "cc" | "multidev_cc" -> Some "c"
  | "cuda" -> Some "cu"
  | "hip" -> Some "hip"
  | "metal" -> Some "metal"
  | _ -> None

(** The keys whose value the backend setting alone decides. *)
let backend_keys = [ "backend"; "backend_extension" ]

(** What [--read=<key>] writes when the backend setting resolves to [setting], for each of
    {!backend_keys}; [None] for any other key, which the reader answers with the key's raw text. An
    unknown backend emits C, the default backend's dialect. *)
let written ~key setting =
  match key with
  | "backend" -> Some (canonical_backend setting)
  | "backend_extension" ->
      Some (Option.value (code_extension (canonical_backend setting)) ~default:"c")
  | _ -> None

(** The backends, among [backends], whose build writes [value] into the file the pform [pform]
    reads: [Some] when [pform] is a [read:] of a file this reader writes for one of {!backend_keys},
    matched by its base name, and [None] for any other pform, whose value says nothing about the
    backend. [Some []] is a value no backend's build writes -- a golden only it names is compared by
    no run at all. *)
let backends_meant ~backends ~pform ~value =
  match String.chop_prefix pform ~prefix:"read:" with
  | None -> None
  | Some path ->
      let base = match String.rsplit2 path ~on:'/' with Some (_, base) -> base | None -> path in
      List.find backend_keys ~f:(fun key -> String.equal (file_of_key key) base)
      |> Option.map ~f:(fun key ->
          List.filter backends ~f:(fun backend ->
              Option.equal String.equal (written ~key backend) (Some value)))
