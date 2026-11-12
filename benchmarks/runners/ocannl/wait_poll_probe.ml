(* Temporary experiment for post-1-task-2; never merge this harness. Both arms time the same
   compiled routine through the production tuner. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let blocking = ref false
let poll_calls = ref 0
let sync_calls = ref 0

module Driver = struct
  include Autotune.Context_driver

  let await_within ctx ~seconds =
    if !blocking then (
      Int.incr sync_calls;
      Context.sync ctx;
      true)
    else (
      Int.incr poll_calls;
      Context.await_within ctx ~seconds)
end

module T = Autotune.Make (Driver)

let () =
  let args = Sys.get_argv () in
  let rounds = Int.of_string args.(1) in
  let box = args.(2) in
  (* No fixture is consumed. Check the installed environment and one existing recorded fixture,
     rather than demand the store's unrelated absent fixtures. *)
  let gate =
    Stdlib.Sys.command
      ("benchmarks/check-box.sh --box " ^ Stdlib.Filename.quote box
     ^ " --fixture lenet --fixture-dir /home/lukstafi/ocannl-staging/benchmarks/fixtures")
  in
  if gate <> 0 then Stdlib.exit gate;
  let n = 262144 in
  let av = Array.init n ~f:(fun i -> Float.of_int ((i * 7 % 23) - 11) /. 16.) in
  let bv = Array.init n ~f:(fun i -> Float.of_int ((i * 3 % 19) - 9) /. 16.) in
  let a = TDSL.ndarray av ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~output_dims:[ n ] () in
  List.iter [ a; b ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let%op c = a + (!.0.5 *. b) in
  let fwd = Train.forward c in
  let comp = Ir.Assignments.comp ~embedded:fwd.embedded_nodes fwd.asgns in
  let candidates =
    List.map [ 32; 128; 256 ] ~f:(fun width ->
        let ctx = Context.auto () in
        let expected_backend = if String.equal box "rog-nv-linux" then "cuda" else "hip" in
        Verdict.p "the requested backend is running"
          (String.equal (Context.backend_name ctx) expected_backend);
        let limits = Context.hardware_limits ctx in
        let ctx, routine =
          Context.compile
            ~name:("wait_poll_w" ^ Int.to_string width)
            ~lowered_transform:(fun o ->
              [
                Ir.Schedule.apply
                  (Ir.Schedule.default_gpu ~block_size:width ~workgroup_fill:width ~min_parallel:1
                     ~limits o)
                  o;
              ])
            ctx comp Ir.Indexing.Empty
        in
        let ctx = Context.run ctx routine in
        let got = Context.get_values ctx c.value in
        Verdict.p_all "compiled candidate matches the host triad" (List.range 0 n) ~f:(fun p ->
            Float.equal got.(p) (av.(p) +. (0.5 *. bv.(p))));
        Stdio.printf "CANDIDATE,%s,w%d,triad262144,f32\n%!" (Context.backend_name ctx) width;
        (width, ctx, routine))
  in
  (* Discard one full queued call per arm/candidate before recording. Each subsequent call
     independently calibrates; no prior depth or cached winner is supplied. ABBA/BAAB pairs reverse
     treatment order and candidate order on alternate rounds. *)
  let run ~record round width ctx routine arm =
    blocking := arm;
    let call = ref None in
    let result =
      T.time_routine
        ~on_call:(fun c -> call := Some c)
        ~timing:Autotune.Queued ~repeats:16 ctx routine
    in
    let call = Option.value_exn !call in
    if record then (
      Stdio.printf "READING,%d,w%d,%s,%.9f,%d,%.9f,%d,%b,%b,%.9f,%.9f\n%!" round width
        (if arm then "sync" else "poll")
        result.ms call.decision.depth result.spread result.samples result.contended result.unbatched
        call.setup_ms call.window.median_wall_ms;
      let samples =
        List.init 16 ~f:(fun _ ->
            let clock = Mtime_clock.counter () in
            for _ = 1 to 512 do
              ignore (Context.run ~timing:true ctx routine : Context.t)
            done;
            if not (Driver.await_within ctx ~seconds:60.) then failwith "fixed batch timed out";
            Mtime.Span.to_float_ns (Mtime_clock.count clock) /. 1e6 /. 512.)
        |> List.sort ~compare:Float.compare
      in
      Stdio.printf "FIXED,%d,w%d,%s,%.9f,512,%.9f,%.9f,16\n%!" round width
        (if arm then "sync" else "poll")
        (List.hd_exn samples)
        ((List.nth_exn samples 7 +. List.nth_exn samples 8) /. 2.)
        (List.last_exn samples))
  in
  List.iter candidates ~f:(fun (w, ctx, r) ->
      List.iter [ false; true ] ~f:(run ~record:false 0 w ctx r));
  for round = 1 to rounds do
    let arms =
      if round % 2 = 1 then [ false; true; true; false ] else [ true; false; false; true ]
    in
    let cs = if round % 2 = 1 then candidates else List.rev candidates in
    List.iter cs ~f:(fun (w, ctx, r) -> List.iter arms ~f:(run ~record:true round w ctx r))
  done;
  Verdict.p "both wait implementations were exercised" (!poll_calls > 0 && !sync_calls > 0);
  Stdio.printf "WAITS,poll=%d,sync=%d\n%!" !poll_calls !sync_calls;
  List.iter candidates ~f:(fun (_, ctx, _) -> Context.release ctx)
