(* gh-ocannl-1661: a shared Stage partitions its copy across the workgroup axis its tile does not
   use, rather than restricting the copy to one representative thread of that axis.

   The staged matmul of schedule_smem_matmul ([mc = ma * mb], i and j split into Grid x Workgroup by
   [bm], k into Serial x Serial by [bk], both operands staged at k_o), at three geometries, one per
   way [Schedule.apply_stage] folds the unused axis (extent [w = bm]) into the copy's serial k loop
   (extent [e = bk]): - divide: [w] divides [e] (8 into 16): the loop runs [e / w] steps over [w * s
   + t], no guard; - drop: [e <= w] (8 against 16): the axis replaces the loop under [t < e]; -
   restrict: neither (24 against 16): one representative thread ([t == 0]) copies, as before.

   On GPU backends each kernel executes against the serial twin, its accumulator prefilled off the
   result so a dropped zero shows. Everywhere, the scheduled IR is read on the host
   ([Ll_test.captured_lowering]): which guards survive, and whether every tile store names both
   Workgroup loops -- each thread copying cells of its own. A broadcast-copy fixture also exercises
   two unused axes: slot 0 takes the minor loop for divide, drop, and a minor drop beside a
   divisible major loop (gh-ocannl-1842). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let skipped = Verdict.skipped ~backend:backend_name
let has_shared = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let geometry ~tag ~n ~bm ~bk ~expect_guard ~partitioned =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ tag ^ "_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ tag ^ "_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  (* The serial twin: [i, j] sums [ma[i, k] * mb[k, j]] over k, on the host. *)
  let want =
    Array.init (n * n) ~f:(fun c ->
        let i = c / n and j = c % n in
        List.sum
          (module Float)
          (List.range 0 n)
          ~f:(fun k -> mav.((i * n) + k) *. mbv.((k * n) + j)))
  in
  p_exists (tag ^ ": the serial twin has a nonzero cell") (Array.to_list want) ~f:(fun x ->
      Float.(x <> 0.));
  let%op mc = ma * mb in
  let schedule (opt : LL.optimized) : Sched.schedule =
    let paths = Ll_test.nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:mc.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    [
      ez;
      sp_zi;
      sp_zj;
      sp_i;
      sp_j;
      sp_k;
      stage ma.Tensor.value [ i_i; k_i ];
      stage mb.Tensor.value [ k_i; j_i ];
    ]
  in
  let comp = named ("scp_" ^ tag) (Train.forward mc) in
  let transform opt = Sched.apply (schedule opt) opt in
  let parity = tag ^ ": staged matmul matches the serial twin" in
  if has_shared then
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    let prefill = Array.init (n * n) ~f:(fun c -> Float.of_int (1000 + c)) in
    let ctx = Context.set_values ctx mc.Tensor.value prefill in
    let ctx = Context.run ctx routine in
    p_close parity (Context.get_values ctx mc.Tensor.value) want ~tol:(fun _ -> 1e-3)
  else skipped parity;
  let post = transform (Ll_test.captured_lowering ~name:("scp_capture_" ^ tag) comp) in
  let guards = ref [] in
  Ll_test.walk post.LL.llc ~on_if:(fun c -> guards := c :: !guards);
  let is_guard op = function LL.Binop (op', _, _) -> Ir.Ops.equal_binop op op' | _ -> false in
  let stores = Ll_test.tile_store_partitions ~tiles:post.LL.workgroup_shared post.LL.llc in
  (match expect_guard with
  | None -> p_empty (tag ^ ": no guard survives over the tile copies") ~over:stores !guards
  | Some (what, op) ->
      p (tag ^ ": exactly the two copies are guarded") (List.length !guards = 2);
      p_all (tag ^ ": each guard is " ^ what) !guards ~f:(is_guard op));
  p_all
    (tag
    ^
    if partitioned then ": every tile store is partitioned across both Workgroup loops"
    else ": every tile store sits under both Workgroup loops, indexed by one")
    stores
    ~f:(fun (n_wg, named_all) -> n_wg = 2 && Bool.equal named_all partitioned)

let () =
  geometry ~tag:"divide" ~n:32 ~bm:8 ~bk:16 ~expect_guard:None ~partitioned:true;
  geometry ~tag:"drop" ~n:32 ~bm:16 ~bk:8
    ~expect_guard:(Some ("the thread bound t < e", Ir.Ops.Cmplt))
    ~partitioned:true;
  geometry ~tag:"restrict" ~n:48 ~bm:16 ~bk:24
    ~expect_guard:(Some ("the representative t == 0", Ir.Ops.Cmpeq))
    ~partitioned:false

(* With two workgroup axes absent from the source, both must share the copy and slot 0 takes its
   minor loop first. With slot 0 reused as the source's minor axis, slot 1 must keep the longer,
   divisible major copy loop. The earlier matmuls have only one fresh copy loop. *)
let broadcast_copy ~tag ~rows ~cols ~reuse_minor =
  let width = if reuse_minor then 16 else 8 in
  let dims = if reuse_minor then [| rows; cols; width |] else [| rows; cols |] in
  let node = Ll_test.node_factory ~first_id:18420 ~dims () in
  let src = node (tag ^ "_src") in
  let dst = node ~dims:[| width; width; rows; cols |] (tag ^ "_dst") in
  let y = Ll_test.sym () and x = Ll_test.sym () in
  let a = Ll_test.sym () and b = Ll_test.sym () in
  let it = Ll_test.iter in
  let raw =
    Ll_test.loop_n y width
      (Ll_test.loop_n x width
         (Ll_test.loop_n a rows
            (Ll_test.loop_n b cols
               (Ll_test.set dst
                  [| it y; it x; it a; it b |]
                  (Ll_test.get src
                     (if reuse_minor then [| it a; it b; it x |] else [| it a; it b |]))))))
  in
  let serial = Ll_test.optimize ~materialized:[ src; dst ] ~name:(tag ^ "_serial") raw in
  let staged =
    Sched.apply
      [
        Sched.Retype { axis = y; ty = LL.Workgroup };
        Sched.Retype { axis = x; ty = LL.Workgroup };
        Sched.Stage
          {
            source = src;
            tile_loops = (if reuse_minor then [ a; b; x ] else [ a; b ]);
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]
      serial
  in
  let slot0 =
    LL.hardware_axes staged.LL.llc
    |> List.filter ~f:(fun h -> Poly.equal h.LL.ha_kind `Workgroup && h.LL.ha_slot = 0)
    |> List.map ~f:(fun h -> h.LL.ha_index)
  in
  let copies = ref [] in
  Ll_test.walk staged.LL.llc ~on_stmt:(function
    | LL.Set { tn; idcs; _ } when Set.mem staged.LL.workgroup_shared tn -> copies := idcs :: !copies
    | _ -> ());
  let coefficient s = function
    | Ir.Indexing.Iterator t -> if Ir.Indexing.equal_symbol s t then 1 else 0
    | Ir.Indexing.Affine { symbols; _ } ->
        List.sum
          (module Int)
          symbols
          ~f:(fun (c, t) -> if Ir.Indexing.equal_symbol s t then c else 0)
    | _ -> 0
  in
  p_all (tag ^ ": consecutive slot-0 lanes copy consecutive minor-axis cells") !copies
    ~f:(fun idcs ->
      List.exists slot0 ~f:(fun s ->
          let minor = Array.length idcs - 1 in
          coefficient s idcs.(minor) = 1
          && Array.for_alli idcs ~f:(fun i idx -> i = minor || coefficient s idx = 0)));
  let stores = Ll_test.tile_store_partitions ~tiles:staged.LL.workgroup_shared staged.LL.llc in
  p_all (tag ^ ": every tile store is partitioned across both Workgroup loops") stores
    ~f:(fun (n, all) -> n = 2 && all);
  (if reuse_minor then
     let slot1 =
       LL.hardware_axes staged.LL.llc
       |> List.filter ~f:(fun h -> Poly.equal h.LL.ha_kind `Workgroup && h.LL.ha_slot = 1)
       |> List.map ~f:(fun h -> h.LL.ha_index)
     in
     p_all (tag ^ ": slot 1 divides the major copy loop without idling lanes") !copies
       ~f:(fun idcs ->
         List.exists slot1 ~f:(fun s -> coefficient s idcs.(0) = 1 && coefficient s idcs.(1) = 0)));
  let parity = tag ^ ": broadcast staging matches the serial twin" in
  if has_shared then
    let values =
      Array.init (Array.fold dims ~init:1 ~f:( * )) ~f:(fun i -> Float.of_int (i + 1) /. 8.)
    in
    let execute name opt =
      Ll_test.execute ~name opt ~seed:[ (src, values) ] ~read:[ dst ] |> List.hd_exn
    in
    let want = execute (tag ^ "_reference") serial in
    let got = execute (tag ^ "_staged") staged in
    p_close parity got want ~tol:(fun _ -> 0.)
  else skipped parity

let () =
  broadcast_copy ~tag:"two-axis divide" ~rows:16 ~cols:16 ~reuse_minor:false;
  (* These two cases discriminate a reverse-only fix, rather than the base scheduler. *)
  broadcast_copy ~tag:"two-axis drop" ~rows:4 ~cols:4 ~reuse_minor:false;
  broadcast_copy ~tag:"minor drop before major divide" ~rows:16 ~cols:4 ~reuse_minor:false;
  broadcast_copy ~tag:"reused slot 0" ~rows:64 ~cols:8 ~reuse_minor:true

let copy_stage source tile_loops =
  Sched.Stage
    {
      source;
      tile_loops;
      shared = true;
      cooperative = None;
      hoisted = false;
      swizzle = None;
      pad_stride = None;
      pipeline_depth = 1;
      tile_prec = None;
    }

let check_copy ~tag ~prefix ~src ~dst ~serial ~(staged : LL.optimized) ~expected ~consumer_stride =
  let slot0 =
    LL.hardware_axes staged.LL.llc
    |> List.filter ~f:(fun a -> Poly.equal a.LL.ha_kind `Workgroup && a.LL.ha_slot = 0)
    |> List.map ~f:(fun a -> a.LL.ha_index)
  in
  let copies = ref [] and consumers = ref [] in
  Ll_test.walk staged.LL.llc ~on_scalar:(function
    | LL.Get (tn, idcs) when Ir.Tnode.equal tn src -> copies := idcs :: !copies
    | LL.Get (tn, idcs) when Set.mem staged.LL.workgroup_shared tn ->
        consumers := idcs :: !consumers
    | _ -> ());
  let coefficient sym = function
    | Ir.Indexing.Iterator s -> if Ir.Indexing.equal_symbol sym s then 1 else 0
    | Ir.Indexing.Affine { symbols; _ } ->
        List.sum
          (module Int)
          symbols
          ~f:(fun (c, s) -> if Ir.Indexing.equal_symbol sym s then c else 0)
    | _ -> 0
  in
  p_all (tag ^ ": consecutive slot-0 lanes read consecutive source-minor cells") !copies
    ~f:(fun idcs ->
      List.exists slot0 ~f:(fun s ->
          let minor = Array.length idcs - 1 in
          coefficient s idcs.(minor) = 1
          && Array.for_alli idcs ~f:(fun i idx -> i = minor || coefficient s idx = 0)));
  let partitions = Ll_test.tile_store_partitions ~tiles:staged.LL.workgroup_shared staged.LL.llc in
  p_all (tag ^ ": both Workgroup axes partition the copy") partitions ~f:(fun (n, all) ->
      n = 2 && all);
  p_all (tag ^ ": consumer keeps its requested slot-0 stride") !consumers ~f:(fun idcs ->
      List.exists slot0 ~f:(fun s -> coefficient s idcs.(Array.length idcs - 1) = consumer_stride));
  (* Closed fixture grammar: one block, no scopes, reductions or nested phases. Replay every
     cooperative write before every consumer, serializing only the hardware loop bindings.
     Materialize the tile to prevent optimization from erasing the copy under test. *)
  let symbol env s = List.Assoc.find_exn env s ~equal:Ir.Indexing.equal_symbol in
  let index env = function
    | Ir.Indexing.Fixed_idx _ as i -> i
    | Ir.Indexing.Iterator s -> Ir.Indexing.Iterator (symbol env s)
    | Ir.Indexing.Affine { symbols; offset } ->
        Ll_test.aff (List.map symbols ~f:(fun (c, s) -> (c, symbol env s))) offset
    | _ -> failwith "copy phase replay: unexpected fixture index"
  in
  let rec scalar env = function
    | LL.Get (tn, idcs) -> LL.Get (tn, Array.map idcs ~f:(index env))
    | LL.Embed_index i -> LL.Embed_index (index env i)
    | (LL.Constant _ | LL.Constant_bits _) as v -> v
    | LL.Unop (op, a) -> LL.Unop (op, arg env a)
    | LL.Binop (op, a, b) -> LL.Binop (op, arg env a, arg env b)
    | LL.Ternop (op, a, b, c) -> LL.Ternop (op, arg env a, arg env b, arg env c)
    | _ -> failwith "copy phase replay: unexpected fixture scalar"
  and arg env (v, p) = (scalar env v, p) in
  let rec phase ~copy env = function
    | LL.For_loop fl ->
        let fresh = Ll_test.sym () in
        LL.For_loop
          {
            fl with
            index = fresh;
            axis = LL.Serial;
            body = phase ~copy ((fl.index, fresh) :: env) fl.body;
          }
    | LL.Seq (a, b) -> LL.Seq (phase ~copy env a, phase ~copy env b)
    | LL.If { cond; body } -> LL.If { cond = arg env cond; body = phase ~copy env body }
    | LL.Set ({ tn; idcs; llsc; _ } as s) ->
        if Bool.equal (Set.mem staged.LL.workgroup_shared tn) copy then
          LL.Set { s with idcs = Array.map idcs ~f:(index env); llsc = scalar env llsc }
        else LL.Noop
    | LL.Noop | LL.Comment _ | LL.Workgroup_barrier -> LL.Noop
    | _ -> failwith "copy phase replay: unexpected fixture statement"
  in
  let tile = Set.to_list staged.LL.workgroup_shared in
  let replay =
    Ll_test.optimize ~materialized:(src :: dst :: tile) ~name:(prefix ^ "_phase_replay")
      (LL.Seq (phase ~copy:true [] staged.LL.llc, phase ~copy:false [] staged.LL.llc))
  in
  let values =
    Array.init
      (Array.fold (Lazy.force src.Ir.Tnode.dims) ~init:1 ~f:( * ))
      ~f:(fun i -> Float.of_int (i + 1) /. 8.)
  in
  let execute name opt =
    Ll_test.execute ~name opt ~seed:[ (src, values) ] ~read:[ dst ] |> List.hd_exn
  in
  let want = execute (prefix ^ "_reference") serial in
  p_close (tag ^ ": serial twin matches the indexed input") want expected ~tol:(fun _ -> 0.);
  p_close
    (tag ^ ": serial phase replay matches the serial twin")
    (execute (prefix ^ "_replay") replay)
    want
    ~tol:(fun _ -> 0.);
  in_leg ~leg:(Verdict.leg ~when_:has_shared ~on:backend_name ())
    (tag ^ ": native shared staging matches the serial twin") (fun label ->
      p_close label (execute (prefix ^ "_native") staged) want ~tol:(fun _ -> 0.))

(* The real register-tile copy: slot 0 is reused in [4*x + t], so keeping that binding gives
   adjacent lanes a stride of four. Only the copy is remapped; the consumer keeps [4*x + t]. Serial
   phase replay helper exercises those exact copy and consumer indices on cc as well as the GPU
   backends, without asking cc to implement shared memory. *)
let register_copy ~transposed =
  let tag = if transposed then "transposed register copy" else "register copy" in
  let prefix = if transposed then "trc" else "rc" in
  let dims = if transposed then [| 32; 64 |] else [| 64; 32 |] in
  let node = Ll_test.node_factory ~first_id:(if transposed then 18470 else 18460) ~dims () in
  let src = node (prefix ^ "_src") in
  let dst = node ~dims:[| 8; 8; 64; 4 |] (prefix ^ "_dst") in
  let y = Ll_test.sym () and x = Ll_test.sym () in
  let k = Ll_test.sym () and t = Ll_test.sym () in
  let it = Ll_test.iter in
  let raw =
    Ll_test.loop_n y 8
      (Ll_test.loop_n x 8
         (Ll_test.loop_n k 64
            (Ll_test.loop_n t 4
               (Ll_test.set dst
                  [| it y; it x; it k; it t |]
                  (Ll_test.get src
                     (if transposed then [| Ll_test.aff [ (4, x); (1, t) ] 0; it k |]
                      else [| it k; Ll_test.aff [ (4, x); (1, t) ] 0 |]))))))
  in
  let serial = Ll_test.optimize ~materialized:[ src; dst ] ~name:(prefix ^ "_serial") raw in
  let staged =
    Sched.apply
      [
        Sched.Retype { axis = y; ty = LL.Workgroup };
        Sched.Retype { axis = x; ty = LL.Workgroup };
        copy_stage src [ k; x; t ];
      ]
      serial
  in
  let expected =
    List.concat_map (List.range 0 8) ~f:(fun _y ->
        List.concat_map (List.range 0 8) ~f:(fun x ->
            List.concat_map (List.range 0 64) ~f:(fun k ->
                List.init 4 ~f:(fun t ->
                    let col = (4 * x) + t in
                    let src_flat = if transposed then (col * 64) + k else (k * 32) + col in
                    Float.of_int (src_flat + 1) /. 8.))))
    |> Array.of_list
  in
  check_copy ~tag ~prefix ~src ~dst ~serial ~staged ~expected ~consumer_stride:4

(* A vector whose coordinate already combines two Workgroup axes must retain both lanes; making one
   serial copy axis and restricting the other would lose participation. *)
let vector_copy () =
  let node = Ll_test.node_factory ~first_id:18480 ~dims:[| 256 |] () in
  let src = node "vc_src" and dst = node ~dims:[| 16; 16 |] "vc_dst" in
  let y = Ll_test.sym () and x = Ll_test.sym () in
  let raw =
    Ll_test.loop_n y 16
      (Ll_test.loop_n x 16
         (Ll_test.set dst
            [| Ll_test.iter y; Ll_test.iter x |]
            (Ll_test.get src [| Ll_test.aff [ (16, y); (1, x) ] 0 |])))
  in
  let serial = Ll_test.optimize ~materialized:[ src; dst ] ~name:"vc_serial" raw in
  let staged =
    Sched.apply
      [
        Sched.Retype { axis = y; ty = LL.Workgroup };
        Sched.Retype { axis = x; ty = LL.Workgroup };
        copy_stage src [ y; x ];
      ]
      serial
  in
  let expected = Array.init 256 ~f:(fun i -> Float.of_int (i + 1) /. 8.) in
  check_copy ~tag:"combined Workgroup vector" ~prefix:"vc" ~src ~dst ~serial ~staged ~expected
    ~consumer_stride:1

let () =
  register_copy ~transposed:false;
  register_copy ~transposed:true;
  vector_copy ()
