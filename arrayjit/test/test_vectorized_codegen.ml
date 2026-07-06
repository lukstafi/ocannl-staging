(* gh-ocannl-164: exact C rendering of the CPU-improvements bundle, on hand-built low-level IR
   through the actual backend codegen path ([C_syntax.compile_proc]).

   - A [Vectorized] loop renders as the backend's vectorization pragmas followed by a plain serial
     [for] under [Pure_C_config] (whose [vector_bytes = 0] disables explicit SIMD); with
     [vectorize_pragma = []] (the GPU configs' choice) it renders as the plain serial loop — the
     legal fallback.
   - With [vector_bytes > 0] (the cc backend's default), an eligible [Vectorized] body renders via
     GCC/Clang vector extensions: typedef + vector loads/arithmetic/stores in lanes-sized chunks,
     a splat for lane-uniform stores, and a serial remainder loop; ineligible bodies (e.g. a
     non-contiguous access) keep the pragma rendering.
   - Materialized kernel parameters carry the [restrict] qualifier; local stack arrays carry the
     SIMD alignment attribute.
   - A slice-alias tnode reaching the parameter list is rejected loudly: with [restrict] an
     aliased parent+view parameter pair would be a miscompile, not just a redundant pointer.
     Assignments lowering never produces one, but hand-built IR (schedule layer, tests) could. *)

open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module LL = Ir.Low_level

let make_optimized llc tns : LL.optimized =
  let traced_store = Hashtbl.create (module Tn) in
  List.iter tns ~f:(fun tn -> ignore (LL.get_node traced_store tn : LL.traced_array));
  {
    traced_store;
    optimize_ctx = Ir.Low_level.empty_optimize_ctx ();
    llc;
    merge_node = None;
    workgroup_shared = Base.Set.empty (module Tn);
  }

let make_on_device id label =
  let tn =
    Tn.create (Tn.Default Ops.single) ~id ~label:[ label ]
      ~unpadded_dims:(lazy [| 8 |])
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode tn Tn.On_device 998;
  tn

let vec_loop ~axis tn =
  let i = Idx.get_symbol () in
  LL.For_loop
    {
      index = i;
      from_ = 0;
      to_ = 7;
      trace_it = false;
      axis;
      body = LL.Set { tn; idcs = [| Idx.Iterator i |]; llsc = LL.Constant 1.0; debug = "" };
    }

let compile_with_pure_config ~name optimized =
  let module Syntax = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    type buffer_ptr = unit Ctypes.ptr

    let procs = [| optimized |]
    let full_printf_support = true
  end))
  in
  let _kparams, doc, _launch = Syntax.compile_proc ~name [] optimized in
  doc

let () =
  (* --- [Vectorized] under the C config: pragmas + serial loop; restrict on the parameter. --- *)
  let out = make_on_device 1 "out" in
  let doc = compile_with_pure_config ~name:"vec_kernel" (make_optimized (vec_loop ~axis:LL.Vectorized out) [ out ]) in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc;
  Stdio.printf "\n";

  (* --- [Vectorized] under a config with no pragmas (the GPU configs' serial fallback). --- *)
  let out2 = make_on_device 2 "out2" in
  let optimized2 = make_optimized (vec_loop ~axis:LL.Vectorized out2) [ out2 ] in
  let module Fallback_syntax = Ir.C_syntax.C_syntax (struct
    include Ir.C_syntax.Pure_C_config (struct
      type buffer_ptr = unit Ctypes.ptr

      let procs = [| optimized2 |]
      let full_printf_support = true
    end)

    let vectorize_pragma = []
  end)
  in
  let _kparams, doc2, _launch = Fallback_syntax.compile_proc ~name:"vec_fallback_kernel" [] optimized2 in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc2;
  Stdio.printf "\n";

  (* --- A local (stack-array) node picks up the SIMD alignment attribute. --- *)
  let local =
    let tn =
      Tn.create (Tn.Default Ops.single) ~id:3 ~label:[ "scratch" ]
        ~unpadded_dims:(lazy [| 8 |])
        ~padding:(lazy None)
        ()
    in
    Tn.update_memory_mode tn Tn.Local 997;
    tn
  in
  let out3 = make_on_device 4 "out3" in
  let i = Idx.get_symbol () in
  let llc3 =
    LL.For_loop
      {
        index = i;
        from_ = 0;
        to_ = 7;
        trace_it = false;
        axis = LL.Serial;
        body =
          LL.Seq
            ( LL.Set { tn = local; idcs = [| Idx.Iterator i |]; llsc = LL.Constant 2.0; debug = "" },
              LL.Set
                {
                  tn = out3;
                  idcs = [| Idx.Iterator i |];
                  llsc = LL.Get (local, [| Idx.Iterator i |]);
                  debug = "";
                } );
      }
  in
  let doc3 = compile_with_pure_config ~name:"aligned_local_kernel" (make_optimized llc3 [ local; out3 ]) in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc3;
  Stdio.printf "\n";

  (* --- Explicit SIMD emission with [vector_bytes = 32]: an eligible elementwise body renders
     as vector-extension code (8 float lanes); the lane-uniform constant store splats. --- *)
  let compile_with_vector_config ~name optimized =
    let module Syntax = Ir.C_syntax.C_syntax (struct
      include Ir.C_syntax.Pure_C_config (struct
        type buffer_ptr = unit Ctypes.ptr

        let procs = [| optimized |]
        let full_printf_support = true
      end)

      let vector_bytes = 32
    end)
    in
    let _kparams, doc, _launch = Syntax.compile_proc ~name [] optimized in
    doc
  in
  let inp = make_on_device 7 "inp" in
  let out4 = make_on_device 8 "out4" in
  let i = Idx.get_symbol () in
  let elementwise =
    LL.For_loop
      {
        index = i;
        from_ = 0;
        to_ = 7;
        trace_it = false;
        axis = LL.Vectorized;
        body =
          LL.Seq
            ( LL.Set
                {
                  tn = out4;
                  idcs = [| Idx.Iterator i |];
                  llsc =
                    LL.Binop
                      ( Ops.Add,
                        (LL.Get (inp, [| Idx.Iterator i |]), Ops.single),
                        (LL.Constant 2.0, Ops.single) );
                  debug = "";
                },
              LL.Set { tn = inp; idcs = [| Idx.Iterator i |]; llsc = LL.Constant 1.0; debug = "" }
            );
      }
  in
  let doc4 =
    compile_with_vector_config ~name:"vec_simd_kernel"
      (make_optimized elementwise [ inp; out4 ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc4;
  Stdio.printf "\n";

  (* --- Ineligible for explicit SIMD (non-contiguous: coefficient 2 on the loop index): the
     pragma rendering remains. --- *)
  let inp2 = make_on_device 9 "inp2" in
  let out5 = make_on_device 10 "out5" in
  let i = Idx.get_symbol () in
  let strided =
    LL.For_loop
      {
        index = i;
        from_ = 0;
        to_ = 3;
        trace_it = false;
        axis = LL.Vectorized;
        body =
          LL.Set
            {
              tn = out5;
              idcs = [| Idx.Iterator i |];
              llsc = LL.Get (inp2, [| Idx.Affine { symbols = [ (2, i) ]; offset = 0 } |]);
              debug = "";
            };
      }
  in
  let doc5 =
    compile_with_vector_config ~name:"vec_strided_kernel" (make_optimized strided [ inp2; out5 ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc5;
  Stdio.printf "\n";

  (* --- An alias view as a would-be kernel parameter must be rejected loudly. --- *)
  let parent = make_on_device 5 "parent" in
  let view = make_on_device 6 "view" in
  Tn.set_alias_of view ~parent
    ~batch_idx:{ Idx.static_symbol = Idx.get_symbol (); static_range = Some 1 };
  (match
     try
       ignore
         (compile_with_pure_config ~name:"alias_kernel"
            (make_optimized (vec_loop ~axis:LL.Serial view) [ view ])
           : PPrint.document);
       None
     with Invalid_argument msg -> Some msg
   with
  | Some msg ->
      Stdio.printf "alias parameter rejected: %b\n"
        (String.is_substring msg ~substring:"restrict")
  | None -> Stdio.printf "alias parameter rejected: false\n");
  Stdio.printf "%!"
