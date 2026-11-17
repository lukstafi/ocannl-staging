(* Temporary gh-ocannl-2215 experiment, carried forward from bd5747b1c. Both wait arms time the same
   compiled routine through the production tuner. The blocking arm is bounded by the experiment's
   process watchdog. Keep the experiment off master. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let blocking = ref false

module Driver = struct
  include Autotune.Context_driver

  let await_within ctx ~seconds =
    if !blocking then (
      Context.sync ctx;
      true)
    else Context.await_within ctx ~seconds
end

module T = Autotune.Make (Driver)

type candidate = { id : string; ctx : Context.t; routine : Context.routine }

let elapsed_us c = Mtime.Span.to_float_ns (Mtime_clock.count c) /. 1e3
let arm_name blocking = if blocking then "sync" else "poll"
let triad_widths = [ 32; 128; 256 ]

let comp out =
  Train.set_materialized out.Tensor.value;
  let fwd = Train.forward out in
  Ir.Assignments.comp ~embedded:fwd.embedded_nodes fwd.asgns

let compile ~id ~transform program (out : Tensor.t) ~expected =
  let ctx, routine =
    Context.compile ~name:("wait_probe_" ^ id) ~lowered_transform:transform
      (Context.hip ~ordinal:0 ()) program Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx out.value in
  let matches =
    Array.length got = Array.length expected && Array.for_all2_exn got expected ~f:Float.equal
  in
  Verdict.p ("output parity for " ^ id) matches;
  if not matches then failwith ("wrong output for " ^ id);
  { id; ctx; routine }

let triad () =
  let n = 262144 in
  let av = Array.init n ~f:(fun i -> Float.of_int ((i * 7 % 23) - 11) /. 16.) in
  let bv = Array.init n ~f:(fun i -> Float.of_int ((i * 3 % 19) - 9) /. 16.) in
  let a = TDSL.ndarray av ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~output_dims:[ n ] () in
  List.iter [ a; b ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let%op c = a + (!.0.5 *. b) in
  let program = comp c in
  let expected = Array.map2_exn av bv ~f:(fun a b -> a +. (0.5 *. b)) in
  let root = Context.hip ~ordinal:0 () in
  let limits = Context.hardware_limits root in
  Context.release root;
  List.map triad_widths ~f:(fun width ->
      let transform o =
        [
          Ir.Schedule.apply
            (Ir.Schedule.default_gpu ~block_size:width ~workgroup_fill:width ~min_parallel:1 ~limits
               o)
            o;
        ]
      in
      compile ~id:("w" ^ Int.to_string width) ~transform program c ~expected)

let seeds ~metadata ~m ~n ~k ~keep =
  let values rows cols a b divisor =
    Array.init (rows * cols) ~f:(fun i ->
        Float.of_int ((((i / cols * a) + (i % cols * b)) % 17) - 8) /. divisor)
  in
  let av = values m k 5 3 16. and bv = values k n 7 2 32. in
  let a = TDSL.ndarray av ~output_dims:[ m; k ] () in
  let b = TDSL.ndarray bv ~output_dims:[ k; n ] () in
  List.iter [ a; b ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let%op c = a +* "ik;kj=>ij" b in
  let program = comp c in
  let expected =
    Array.init (m * n) ~f:(fun i ->
        let sum = ref 0. in
        for p = 0 to k - 1 do
          sum := !sum +. (av.((i / n * k) + p) *. bv.((p * n) + (i % n)))
        done;
        !sum)
  in
  let root = Context.hip ~ordinal:0 () in
  let capabilities = Context.codegen_capabilities root in
  let limits = Context.hardware_limits root in
  let base =
    Context.lowered_for_decisions ~name:"wait_probe_seeds" root program Ir.Indexing.Empty
  in
  let seeded =
    Autotune.search_seeding ~capabilities ~is_gpu:true ~is_cpu:false ~limits ~static_indices:[] base
  in
  Context.release root;
  let population = List.filter seeded.ss_whole ~f:Autotune.scalar_blocktile in
  let candidates =
    List.filter_mapi population ~f:(fun i p ->
        let id = Printf.sprintf "seed%d" i in
        Stdio.printf "MENU,%s,%s,%S\n%!" metadata id (Autotune.sketch_params_label p);
        if not (List.is_empty keep || List.mem keep id ~equal:String.equal) then None
        else
          let transform o =
            [
              Ir.Schedule.apply
                (Autotune.sketch_schedule ~accum_prec:capabilities.accum_prec ~p o)
                o;
            ]
          in
          Some (compile ~id ~transform program c ~expected))
  in
  let complete =
    List.length candidates >= 2 && (List.is_empty keep || List.length candidates = List.length keep)
  in
  Verdict.p "at least two seeded candidates compiled" complete;
  if not complete then failwith "seed shortlist did not match";
  candidates

let run_tuner ?prior_depth { ctx; routine; _ } arm =
  blocking := arm;
  let call = ref None in
  let result =
    T.time_routine ?prior_depth
      ~on_call:(fun c -> call := Some c)
      ~timing:Autotune.Queued ~repeats:16 ctx routine
  in
  (result, Option.value_exn !call)

let record_tuner ~metadata ?prior_depth ~round ~visit ({ id; _ } as candidate) arm =
  let result, call = run_tuner ?prior_depth candidate arm in
  let depth = call.decision.depth in
  let kind = if Option.is_none prior_depth then "adaptive" else "matched" in
  Stdio.printf "READING,%s,%s,%d,%d,%s,%s,%.9f,%d,%s,%.9f,%d,%b,%b,%.9f,%.9f\n%!" metadata kind
    round visit id (arm_name arm) result.ms depth
    (Autotune.batch_settle_name call.decision.settle)
    result.spread result.samples result.contended result.unbatched call.setup_ms
    call.window.median_wall_ms;
  List.iteri call.calibration_probes ~f:(fun ordinal probe ->
      Stdio.printf "CALIBRATION,%s,%s,%d,%d,%s,%s,%d,%d,%d,%.9f\n%!" metadata kind round visit id
        (arm_name arm) ordinal probe.depth probe.runs probe.min_ms)

let candidates_abba ~metadata ~rounds ~process candidates =
  (* A common prior comes from the two discarded warm-up calls. The recorded calls still run the
     production timing loop, and their settle reason and observed depth identify whether the common
     prior was accepted. *)
  let warmed =
    List.map candidates ~f:(fun candidate ->
        let first = process % 2 = 0 in
        let _, a = run_tuner candidate first in
        let _, b = run_tuner candidate (not first) in
        (candidate, (a.decision.depth + b.decision.depth) / 2))
  in
  for round = 1 to rounds do
    let forward = (round + process) % 2 = 0 in
    let arms = if forward then [ false; true; true; false ] else [ true; false; false; true ] in
    let ordered = if forward then warmed else List.rev warmed in
    List.iteri arms ~f:(fun visit arm ->
        let order = if visit % 2 = 0 then ordered else List.rev ordered in
        List.iter order ~f:(fun (candidate, depth) ->
            if (round + visit) % 2 = 0 then
              record_tuner ~metadata ~prior_depth:depth ~round ~visit candidate arm;
            record_tuner ~metadata ~round ~visit candidate arm;
            if (round + visit) % 2 <> 0 then
              record_tuner ~metadata ~prior_depth:depth ~round ~visit candidate arm))
  done

type wait = Sync | Tight_query | Sparse_spin | Sparse_sleep

let wait_name = function
  | Sync -> "sync"
  | Tight_query -> "tight"
  | Sparse_spin -> "sparse-spin"
  | Sparse_sleep -> "sparse-sleep"

let fixed_batch ~stream ~first ~last ~depth { ctx; routine; _ } wait =
  let current = ref ctx in
  let wall = Mtime_clock.counter () in
  Hip.Event.record first stream;
  for _ = 1 to depth do
    current := Context.run ~timing:true !current routine
  done;
  Hip.Event.record last stream;
  let wait_clock = Mtime_clock.counter () in
  let submit_us = elapsed_us wall in
  let queries = ref 0 and last_negative_us = ref None and ready_us = ref None in
  let next_query_us = ref 0. in
  (match wait with
  | Sync -> ()
  | Tight_query | Sparse_spin | Sparse_sleep ->
      let ready () =
        (* Space queries by 200 us in both sparse cells. The device event clock excludes the
           observation lag and the log carries the achieved query spacing for both cells. *)
        let started = elapsed_us wait_clock in
        Int.incr queries;
        if Hip.Stream.is_ready stream then (
          ready_us := Some (elapsed_us wait_clock);
          true)
        else (
          last_negative_us := Some started;
          next_query_us := Float.max (!next_query_us +. 200.) started;
          (match wait with
          | Tight_query -> ()
          | Sparse_spin ->
              while Float.(elapsed_us wait_clock < !next_query_us) do
                Stdlib.Domain.cpu_relax ()
              done
          | Sparse_sleep ->
              Unix.sleepf (Float.max 0. (!next_query_us -. elapsed_us wait_clock) /. 1e6)
          | Sync -> assert false);
          false)
      in
      if not (Utils.poll_within ~seconds:60. ready) then failwith "diagnostic poll timed out");
  let await_clock = Mtime_clock.counter () in
  Context.sync !current;
  let await_us = elapsed_us await_clock in
  let wall_us = elapsed_us wall in
  let gpu_us = Hip.Event.elapsed_time ~start:first ~end_:last *. 1e3 in
  let bracket_us =
    match (!last_negative_us, !ready_us) with Some low, Some high -> high -. low | _ -> Float.nan
  in
  (wall_us, gpu_us, submit_us, !queries, bracket_us, await_us)

let diagnostics ~metadata ~rounds ~process candidates =
  let stream = Hip_backend.Impl.wait_probe_stream ~ordinal:0 in
  let first = Hip.Event.create ~enable_timing:true () in
  let last = Hip.Event.create ~enable_timing:true () in
  let cases = [ Sync; Tight_query; Sparse_spin; Sparse_sleep ] in
  let widths = [ ("w32", 1290); ("w128", 1320); ("w256", 1325) ] in
  for round = 1 to rounds do
    let reverse = (round + process) % 2 <> 0 in
    let candidates = if reverse then List.rev candidates else candidates in
    List.iter candidates ~f:(fun ({ id; _ } as candidate) ->
        let depths = [ 512; List.Assoc.find_exn widths id ~equal:String.equal ] in
        let depths = if reverse then List.rev depths else depths in
        List.iter depths ~f:(fun depth ->
            for sample = 0 to 15 do
              let shift = ((sample / 2) + round + process) % 4 in
              let left, right = List.split_n cases shift in
              let order = right @ left in
              let order = if (sample + round + process) % 2 = 0 then order else List.rev order in
              List.iter order ~f:(fun wait ->
                  let wall, gpu, submit, queries, bracket, await =
                    fixed_batch ~stream ~first ~last ~depth candidate wait
                  in
                  Stdio.printf "FIXED,%s,%d,%d,%s,%s,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f\n%!" metadata
                    round sample id (wait_name wait) depth wall gpu submit queries bracket await)
            done))
  done

let screen ~metadata ~rounds candidates =
  let depth = 512 in
  let batch { ctx; routine; _ } =
    let current = ref ctx in
    let start = Mtime_clock.counter () in
    for _ = 1 to depth do
      current := Context.run ~timing:true !current routine
    done;
    Context.sync !current;
    elapsed_us start /. Float.of_int depth
  in
  List.iter candidates ~f:(fun candidate -> ignore (batch candidate : float));
  for round = 1 to rounds do
    let order = if round % 2 = 0 then candidates else List.rev candidates in
    List.iter order ~f:(fun ({ id; _ } as candidate) ->
        Stdio.printf "SCREEN,%s,%d,%s,%.6f\n%!" metadata round id (batch candidate))
  done

let () =
  let usage =
    "wait_poll_probe.exe PROCESS ROUNDS triad|diagnostic\n\
     wait_poll_probe.exe PROCESS ROUNDS seed|screen M N K [seedN ...]"
  in
  let metadata process site =
    let ctx = Context.hip ~ordinal:0 () in
    let open Ir.Backend_intf in
    let props = Option.value_exn (parse_static_properties (Context.static_properties ctx)) in
    let device = List.hd_exn props.devices in
    let field name = List.Assoc.find_exn device name ~equal:String.equal in
    let memory = if Bool.t_of_sexp (field "integrated") then "unified" else "discrete" in
    let arch = Sexp.to_string (field "gcn_arch_name") in
    Context.release ctx;
    Printf.sprintf "%d,hip,%s,%s,%s" process memory arch site
  in
  let run process rounds site =
    let process = Int.of_string process and rounds = Int.of_string rounds in
    if rounds <= 0 then failwith usage;
    let name =
      match site with
      | [ "triad" ] | [ "diagnostic" ] -> "triad262144"
      | ("seed" | "screen") :: m :: n :: k :: _ -> String.concat ~sep:"x" [ m; n; k ]
      | _ -> failwith usage
    in
    let metadata = metadata process name in
    let candidates =
      match site with
      | [ "triad" ] | [ "diagnostic" ] -> triad ()
      | ("seed" | "screen") :: m :: n :: k :: keep ->
          seeds ~metadata ~m:(Int.of_string m) ~n:(Int.of_string n) ~k:(Int.of_string k) ~keep
      | _ -> assert false
    in
    Exn.protect
      ~f:(fun () ->
        List.iter candidates ~f:(fun { id; routine; _ } ->
            let dims a = String.concat ~sep:"x" (List.map (Array.to_list a) ~f:Int.to_string) in
            let segments =
              List.map routine.segments ~f:(fun seg ->
                  let open Ir.Low_level in
                  let launch = launch_dims seg.llc in
                  let digest = Ir.Schedule_cache.(digest (canonicalize seg)) in
                  Printf.sprintf "%s:%s/%s" digest (dims launch.grid) (dims launch.block))
            in
            Stdio.printf "CANDIDATE,%s,%s,%s\n%!" metadata id (String.concat ~sep:"+" segments));
        match site with
        | [ "diagnostic" ] -> diagnostics ~metadata ~rounds ~process candidates
        | "screen" :: _ -> screen ~metadata ~rounds candidates
        | _ -> candidates_abba ~metadata ~rounds ~process candidates)
      ~finally:(fun () -> List.iter candidates ~f:(fun c -> Context.release c.ctx))
  in
  match Array.to_list (Sys.get_argv ()) with
  | _ :: process :: rounds :: site -> run process rounds site
  | _ -> failwith usage
