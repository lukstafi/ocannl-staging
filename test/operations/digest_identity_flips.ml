(* gh-ocannl-572, calibration: the classification is real, not just declared.

   [digest_completeness] checks the registry's shape — every key classified, every claimed cache-key
   component existing. Shape is not substance: a key could be classified code-borne and reach
   nothing. So this test flips one representative key of each class and observes the identity a
   cache would key on, which is what the classes are statements about:

   - code-borne: the canonical digest changes (and with it the whole key),
   - keyed: the digest is UNCHANGED — the flip is invisible to the lowered code, which is the hazard
     — while the cache key separates the regimes,
   - search-shaping / execution-neutral: neither changes.

   The issue's own caution applies: flips are calibration, not the mechanism. Absence of change is
   weak evidence in general, so this covers a handful of representatives rather than every key, and
   the declarative registry stays the invariant. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module SC = Ir.Schedule_cache

let p name b = Stdio.printf "%s: %b\n" name b

(* Config values resolve per lookup, so a config-file entry poked in is what later reads see —
   unless the commandline or the environment states the key, which take precedence. Each flip below
   asserts it actually took effect before drawing a conclusion from it. *)
let set_config key value = Hashtbl.set Utils.config_file_args ~key ~data:value
let unset_config key = Hashtbl.remove Utils.config_file_args key

(* A fresh computation per compile: tensors carry their memory modes and lowering results, so a
   second compile of the same graph would not re-decide anything. [a]'s eight values are small
   enough to be filled by generated code under the default [limit_constant_fill_size], which is what
   gives the code-borne flip below something to act on. *)
let identity_of ctx =
  let a =
    TDSL.ndarray [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] ~label:[ "dif_a" ] ~output_dims:[ 8 ] ()
  in
  let%op b = a *. a in
  let%op c = b + b in
  let comp = Train.forward c in
  let canon = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        canon := Some (SC.canonicalize ~static_indices:[] opt);
        opt)
      ctx comp Ir.Indexing.Empty
  in
  let canon = Option.value_exn ~here:[%here] !canon in
  ( SC.digest canon,
    SC.cache_key ~limits:(Context.hardware_limits ctx) canon ~backend:(Context.backend_name ctx) )

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let is_cc = String.is_prefix backend ~prefix:"cc" || String.is_prefix backend ~prefix:"multidev" in
  let digest0, key0 = identity_of ctx in
  p "the identity is stable across two compiles of the same program"
    (let digest1, key1 = identity_of ctx in
     String.equal digest0 digest1 && String.equal key0 key1);

  (* Code-borne: below the limit a constant tensor is filled by generated code, above it from a
     host array — a different program either way. *)
  set_config "limit_constant_fill_size" "1";
  p "the code-borne flip took effect"
    (String.equal "1" (Utils.get_global_arg ~default:"" ~arg_name:"limit_constant_fill_size"));
  let digest_v, key_v = identity_of ctx in
  unset_config "limit_constant_fill_size";
  p "a code-borne knob (limit_constant_fill_size) changes the digest"
    (not (String.equal digest0 digest_v));
  p "and therefore the cache key" (not (String.equal key0 key_v));

  (* Keyed, backend-independent: the index/pool-slot width is read when a kernel is emitted. *)
  let large_models0 = Utils.settings.large_models in
  Utils.settings.large_models <- not large_models0;
  let digest_l, key_l = identity_of ctx in
  Utils.settings.large_models <- large_models0;
  p "a keyed codegen knob (large_models) leaves the digest alone" (String.equal digest0 digest_l);
  p "while the cache key separates the regimes" (not (String.equal key0 key_l));

  (* Keyed, backend-specific: the cc backend's vector width. On other backends the knob reaches no
     codegen, and the key is expected NOT to move — the same statement, read the other way. *)
  set_config "cc_vector_bytes" "0";
  let digest_c, key_c = identity_of ctx in
  unset_config "cc_vector_bytes";
  p "a cc codegen knob (cc_vector_bytes) leaves the digest alone" (String.equal digest0 digest_c);
  p "and separates the key exactly on the backend that reads it"
    (Bool.equal is_cc (not (String.equal key0 key_c)));

  (* Search-shaping and execution-neutral: neither touches the identity. *)
  set_config "autotune_beam_width" "7";
  set_config "print_decimals_precision" "9";
  let digest_n, key_n = identity_of ctx in
  unset_config "autotune_beam_width";
  unset_config "print_decimals_precision";
  p "a search-shaping and an execution-neutral knob leave the identity untouched"
    (String.equal digest0 digest_n && String.equal key0 key_n);

  (* And the numerics component, whose omission was gh-ocannl-568. *)
  let base = Ir.Numerics.get () in
  Ir.Numerics.set_policy { base with Ir.Numerics.tf32_matmuls = not base.Ir.Numerics.tf32_matmuls };
  let digest_t, key_t = identity_of ctx in
  Ir.Numerics.set_policy base;
  p "a numerics knob (tf32_matmuls) leaves the digest alone" (String.equal digest0 digest_t);
  p "while the cache key separates the policies" (not (String.equal key0 key_t))
