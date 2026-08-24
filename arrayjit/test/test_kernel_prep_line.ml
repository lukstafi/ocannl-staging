(* GPU-free coverage for the backend kernel-entry policy hook (gh-ocannl-735).

   HIP uses this hook for [_Pragma("clang fp reassociate(off)")]. Testing the pure C-syntax
   functor keeps the assertion runnable without hipjit or a device, and checks the shipped renderer
   rather than scanning [hip_backend.ml] as text. The HIP [reduction_forms] run separately proves
   that its real config selects this value and that hiprtc honors it numerically. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode

let optimized : LL.optimized =
  {
    traced_store = Hashtbl.create (module Tn);
    optimize_ctx = Ir.Low_level.empty_optimize_ctx ();
    llc = LL.Noop;
    merge_node = None;
    workgroup_shared = Set.empty (module Tn);
    simdgroup_fragments = Set.empty (module Tn);
    swizzled = Map.empty (module Tn);
    pipelined = Map.empty (module Tn);
    zero_fringe = Set.empty (module Tn);
    flip_candidates = [];
    spliced_rbw = Set.empty (module Tn);
  }

module Syntax = Ir.C_syntax.C_syntax (struct
  include Ir.C_syntax.Pure_C_config (struct
    type buffer_ptr = unit Ctypes.ptr

    let procs = [| optimized |]
    let full_printf_support = true
  end)

  let kernel_prep_line = "_Pragma(\"clang fp reassociate(off)\")"
end)

let () =
  let _, doc, _ = Syntax.compile_proc ~name:"kernel_prep_probe" [] optimized in
  let buf = Buffer.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf doc;
  let src = Buffer.contents buf in
  let pragma = "_Pragma(\"clang fp reassociate(off)\");" in
  Verdict.p "the kernel-entry policy is emitted as the first body statement"
    (match (String.substr_index src ~pattern:pragma, String.substr_index src ~pattern:"/* Main logic. */") with
    | Some prep, Some logic -> prep < logic
    | _ -> false)
