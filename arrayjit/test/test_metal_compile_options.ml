(* GPU-free coverage for Metal's MTLCompileOptions property sequence (gh-ocannl-848).

   Metal_backend interprets this pure list into the mutable Objective-C options object. Keeping the
   complete production state here pins both debug variants without linking Metal or requiring a
   device. The arithmetic and math-function properties are separate on purpose: Safe prevents the
   measured reassociation, while Fast retains the function family the old default selected. *)

open Base

let cases =
  [
    ( false,
      [
        Ir.Compiler_options.Language_version_3_1;
        Math_mode_safe;
        Math_functions_fast;
        Enable_logging false;
      ] );
    ( true,
      [
        Ir.Compiler_options.Language_version_3_2;
        Math_mode_safe;
        Math_functions_fast;
        Enable_logging true;
      ] );
  ]

let () =
  List.iter cases ~f:(fun (routine_logging, want) ->
      let got = Ir.Compiler_options.metal ~routine_logging in
      Stdio.eprintf "debug=%b:\n  got:  %s\n  want: %s\n" routine_logging
        (Ir.Compiler_options.render_metal got)
        (Ir.Compiler_options.render_metal want));
  Verdict.p_all "every Metal variant pins language, safe arithmetic, fast functions and logging"
    cases ~f:(fun (routine_logging, want) ->
      List.equal Ir.Compiler_options.equal_metal_option
        (Ir.Compiler_options.metal ~routine_logging)
        want)
