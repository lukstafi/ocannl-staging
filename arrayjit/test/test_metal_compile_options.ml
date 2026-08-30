(* GPU-free coverage for Metal's MTLCompileOptions property sequence (gh-ocannl-848).

   Metal_backend interprets this pure list into the mutable Objective-C options object. Keeping the
   complete property sequences here pin both debug variants and both API capabilities without
   linking Metal or requiring a device. The modern arithmetic and math-function properties are
   separate on purpose: Safe prevents the measured reassociation, while Fast retains the function
   family the old default selected. macOS 14 receives the safe legacy spelling instead. *)

open Base

let cases =
  [
    ( false,
      Ir.Compiler_options.Modern_split,
      [ Ir.Compiler_options.Language_version_3_1; Math_mode_safe; Math_functions_fast ] );
    ( true,
      Modern_split,
      [
        Ir.Compiler_options.Language_version_3_2;
        Math_mode_safe;
        Math_functions_fast;
        Enable_logging;
      ] );
    (false, Legacy, [ Ir.Compiler_options.Language_version_3_1; Fast_math_enabled false ]);
    ( true,
      Legacy,
      [ Ir.Compiler_options.Language_version_3_2; Fast_math_enabled false; Enable_logging ] );
  ]

let () =
  List.iter cases ~f:(fun (routine_logging, math_api, want) ->
      let got = Ir.Compiler_options.metal ~routine_logging ~math_api in
      Stdio.eprintf "debug=%b api=%s:\n  got:  %s\n  want: %s\n" routine_logging
        (match math_api with Modern_split -> "modern" | Legacy -> "legacy")
        (Ir.Compiler_options.render_metal got)
        (Ir.Compiler_options.render_metal want));
  Verdict.p_all "every Metal API/debug variant selects its declared option sequence" cases
    ~f:(fun (routine_logging, math_api, want) ->
      List.equal Ir.Compiler_options.equal_metal_option
        (Ir.Compiler_options.metal ~routine_logging ~math_api)
        want)
