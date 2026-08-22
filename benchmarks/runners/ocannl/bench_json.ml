(* JSON scalars and the result line of the OCANNL benchmark runners (gh-ocannl-676).

   Separate from [Bench_harness] — which is a module of the runner executables and drags in the
   whole library — so that a test can feed the line fabricated values: a diverged loss vector, a
   time that was never measured, a backend diagnostic full of control characters. The line is one
   ~300-character format with optional pre-formatted fragments and a nested object, and
   [orchestrate.py] reads a cell's result by [json.loads]ing it; before this module nothing
   anywhere parsed it, and a cell whose line does not parse is reported as a broken runner after
   the whole measurement has been paid for. *)

open Base

(** A JSON number, or [null] when the value is not finite.

    Every non-finite float in this file becomes [null] rather than OCaml's [nan] / [inf] / [-inf],
    none of which JSON has: a diverged training run is exactly the run whose evidence the result
    line exists to carry, so it must not be the run whose result line fails to parse. The consumer
    side of the same rule is [orchestrate.py]'s DIVERGED verdict — [null] in a loss vector means
    the cell ran and diverged, which is a parity failure naming its cause, not a missing cell. *)
let num ?(prec = 6) v = if Float.is_finite v then Printf.sprintf "%.*g" prec v else "null"

(** As {!num}, with a fixed number of decimals ([%f] rather than [%g]). *)
let fixed ?(prec = 3) v = if Float.is_finite v then Printf.sprintf "%.*f" prec v else "null"

(** A JSON array of numbers, each by {!num}. *)
let nums ?prec arr = String.concat ~sep:"," (Array.to_list (Array.map arr ~f:(num ?prec)))

(** Quote-and-control-character scrubbing rather than escaping: these strings are diagnostics
    (schedule labels, an exception's message) and the result line has to stay one parseable JSON
    line. JSON forbids every unescaped byte below U+0020, not just the whitespace ones, so the test
    is the code point — a NUL or an ESC from a backend diagnostic would otherwise invalidate the
    record and cost the whole measurement. *)
let string s =
  String.map s ~f:(function
    | '"' -> '\''
    | '\\' -> '/'
    | c when Char.to_int c < 0x20 || Char.to_int c = 0x7f -> ' '
    | c -> c)

(** One arm of the [tune] object: the crowned candidate of one placement arm, its search
    provenance, and how its best timed tensorized candidate compared (gh-ocannl-546). [best_ms] and
    [mma_best_ms] are [infinity] when the arm timed nothing at all, which {!num} renders [null].

    [state] names what the arm did about searching — the {!Autotune.outcome_name} of its outcome
    (gh-ocannl-677), one of ["searched"], ["search-died"], ["cache-replay"], ["search-disabled"],
    ["pre-search-failure"]. [searched] and [cache_hit] are that same fact projected onto the two
    booleans the wire format carried before, kept for readers that predate the field; they are NOT
    complements, and deriving the state from them is the mistake the outcome type exists to stop. *)
let tune_arm ~name ~state ~searched ~cache_hit ~best_ms ~best_label ~tensorized
    ~mma_scalar_fallbacks ~mma_seeded ~mma_timed ~mma_best_ms ~terminal_failure =
  Printf.sprintf
    {|{"arm":"%s","state":"%s","searched":%b,"cache_hit":%b,"best_ms":%s,"best_label":"%s","tensorized":%b,"mma_scalar_fallbacks":%d,"mma_seeded":%d,"mma_timed":%d,"mma_best_ms":%s,"terminal_failure":%s}|}
    (string name) (string state) searched cache_hit (num best_ms) (string best_label) tensorized
    mma_scalar_fallbacks mma_seeded mma_timed (num mma_best_ms)
    (Option.value_map terminal_failure ~default:"null" ~f:(fun detail ->
         Printf.sprintf {|"%s"|} (string detail)))

(** The [tune] object of the result line, over arms already rendered by {!tune_arm}.

    Three provenance totals, not two (gh-ocannl-677): [no_searches] counts the arms that neither
    searched nor replayed — [autotune_search=false] and every pre-search failure — so
    [orchestrate.py] reads that case instead of inferring it from [searches] and [replays] both
    being zero. *)
let tune_object ~shipped ~searches ~replays ~no_searches ~arms =
  Printf.sprintf {|{"shipped":"%s","searches":%d,"replays":%d,"no_searches":%d,"arms":[%s]}|}
    (string shipped) searches replays no_searches
    (String.concat ~sep:"," arms)

(** The result line [orchestrate.py] parses, as a string without its trailing newline.

    [tune] is the already-built [tune] object (see [Bench_harness.tune_json]) or [None] for an
    untuned cell; [tokens_per_step] is present only for workloads that have one. The percentiles
    and [queued_ms] are milliseconds. *)
let result_line ~backend ~variant ~precision ~workload ~compile_s ~searched ?tokens_per_step ?tune
    ~p10 ~p50 ~p90 ~queued_ms ~timed_steps ~losses () =
  let tokens_field =
    match tokens_per_step with Some t -> Printf.sprintf {|"tokens_per_step":%d,|} t | None -> ""
  in
  let tune_field = match tune with Some j -> Printf.sprintf {|"tune":%s,|} j | None -> "" in
  Printf.sprintf
    {|{"framework":"ocannl","backend":"%s","variant":"%s","precision":"%s","workload":"%s","compile_s":%s,"searched":%b,%s%s"step_ms":{"p10":%s,"p50":%s,"p90":%s},"queued_step_ms":%s,"timed_steps":%d,"losses":[%s]}|}
    (string backend) (string variant) (string precision) (string workload) (fixed compile_s)
    searched tokens_field tune_field (num p10) (num p50) (num p90) (num queued_ms) timed_steps
    (nums ~prec:9 losses)
