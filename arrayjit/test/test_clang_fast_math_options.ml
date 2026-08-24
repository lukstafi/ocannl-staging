(* GPU-free coverage for HIP's compiler-level reduction-order policy (gh-ocannl-735).

   The real HIP backend consumes this pure helper. Keeping the ordering assertion in arrayjit.ir
   makes it runnable without hipjit or a device; [reduction_forms] remains the hardware-backed
   proof that hiprtc honors the options for every scalar reduction spelling. *)

open Base

let () =
  let ordered = Ir.C_syntax.clang_fast_math_options ~reassociate:false in
  Verdict.p "the no-reassociation override follows the fast-math umbrella flag"
    (List.equal String.equal ordered [ "-ffast-math"; "-fno-associative-math" ]);
  Verdict.p "the reassociating policy needs no redundant override"
    (List.equal String.equal
       (Ir.C_syntax.clang_fast_math_options ~reassociate:true)
       [ "-ffast-math" ])
