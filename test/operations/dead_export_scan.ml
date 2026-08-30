(* gh-ocannl-806: a top-level helper in a module without an [.mli] is an exported value, even when
   its author meant it to be private. Two incidental hand sweeps found many such exports and still
   undercounted one family twofold. This repository scan makes the census mechanical for direct
   [*.ml] modules in [arrayjit/lib/] and [tensor/].

   The matching policy lives in [test/support/dead_export_scan.ml] and its synthetic negative
   controls. In particular, qualified paths through aliases count, and a bare identifier inside the
   lexical scope of [open M] counts conservatively. The latter can mistake a shadowing local for
   [M]'s value; that false-positive direction is deliberate for this first cut. An [include M]
   counts every value because it re-exports the interface. PPX-generated values and values brought
   into the defining module by [include] are outside this source-level census.

   Every pre-existing zero-reference value is an exact [Module.value] exemption below. A new
   implicit export fails rather than silently extending that baseline; an exemption that gains a
   reference also fails stale, so cleanup cannot leave a permanent hole. *)

open Base
open Stdio
module Read = Test_utils.Config_key_scan
module Scan = Test_utils.Dead_export_scan
module Dune = Test_utils.Dune_stanza_scan

(* Filled from the first live census. Every entry names one exact implicit export and shares the
   reason stated above: it predates the ratchet and needs an explicit interface or a usage
   decision. *)
let exempt_zero_reference_exports =
  [
    "Affine.ap_covered_chunk";
    "Affine.ap_of_form";
    "Affine.ap_subset";
    "Affine.axis_index_to_string";
    "Affine.ceil_div";
    "Affine.equal_verdict";
    "Affine.equation_of";
    "Affine.floor_div";
    "Affine.forced_pairs";
    "Affine.gcd";
    "Affine.infeasible";
    "Affine.linear_terms";
    "Affine.range_of_var";
    "Affine.terms_of";
    "Backend_impl._get_local_debug_runtime";
    "Backend_impl.next_global_device_id";
    "Backend_intf.sexp_of_device";
    "Builtins_cc.source";
    "C_syntax._all_precs_is_complete";
    "C_syntax._get_local_debug_runtime";
    "C_syntax.all_precs";
    "C_syntax.c_float_literal";
    "C_syntax.c_keywords";
    "C_syntax.c_stdlib_idents";
    "C_syntax.c_stdlib_macros";
    "C_syntax.current_kernel_name";
    "C_syntax.empty_mma_summary";
    "C_syntax.empty_peel_summary";
    "C_syntax.empty_volatility_summary";
    "C_syntax.extract_idents";
    "C_syntax.is_f32_tie";
    "C_syntax.is_tensorized_rendering";
    "C_syntax.log_declines";
    "C_syntax.mma_census_enabled";
    "C_syntax.op_syntax_idents";
    "C_syntax.summarize_peel_census";
    "C_syntax.summarize_volatility_census";
    "Compiler_options.clang_fast_math_options";
    "Cpu_topology.affinity_mask_stub";
    "Cpu_topology.effective_cpu_count_stub";
    "Cpu_topology.hypervisor_present_stub";
    "Cpu_topology.linux_classes";
    "Cpu_topology.native_core_classes_str";
    "Cpu_topology.read_sys_file";
    "Cpu_topology.set_process_affinity_stub";
    "Cpu_topology.sort_fastest_first";
    "Cpu_topology.total_logical_count";
    "Host_inits.table";
    "Interval.all_finite";
    "Interval.arith_result";
    "Interval.endpoint_exact";
    "Interval.float_exact_int_limit";
    "Interval.value_fits";
    "Lexer.__ocaml_lex_multichar_token_rec";
    "Lexer.__ocaml_lex_single_char_token_rec";
    "Lexer.__ocaml_lex_tables";
    "Lexer.buffered_token";
    "Lexer.multichar_token";
    "Lexer.single_char_token";
    "Ndarray._get_local_debug_runtime";
    "Ndarray.adjust_idx_for_padding";
    "Ndarray.big_ptr_to_string";
    "Ndarray.bigarray_start_not_managed";
    "Ndarray.c_ptr_to_string";
    "Ndarray.compare";
    "Ndarray.compute_end_idx";
    "Ndarray.count_logical_elems";
    "Ndarray.create_bigarray";
    "Ndarray.create_bigarray_of_prec";
    "Ndarray.decoded_count";
    "Ndarray.default_kind";
    "Ndarray.equal";
    "Ndarray.fill_from_float";
    "Ndarray.float_to_uint32";
    "Ndarray.float_to_uint64";
    "Ndarray.fold_bigarray";
    "Ndarray.get_voidptr_not_managed";
    "Ndarray.hash";
    "Ndarray.hash_fold_t";
    "Ndarray.hash_t";
    "Ndarray.linear_offset_of_idx";
    "Ndarray.log_debug_info";
    "Ndarray.mappable_file_region";
    "Ndarray.mapped_count";
    "Ndarray.precision_string";
    "Ndarray.precision_to_bigarray_kind";
    "Ndarray.ptr_to_string_hum";
    "Ndarray.sexp_of_bfloat16_nd";
    "Ndarray.sexp_of_bigarray";
    "Ndarray.sexp_of_byte_nd";
    "Ndarray.sexp_of_double_nd";
    "Ndarray.sexp_of_fp8_nd";
    "Ndarray.sexp_of_half_nd";
    "Ndarray.sexp_of_int32_nd";
    "Ndarray.sexp_of_int64_nd";
    "Ndarray.sexp_of_single_nd";
    "Ndarray.sexp_of_uint16_nd";
    "Ndarray.sexp_of_uint32_nd";
    "Ndarray.sexp_of_uint4x32_nd";
    "Ndarray.sexp_of_uint64_nd";
    "Ndarray.to_native";
    "Ndarray.two_pow_63";
    "Ndarray.uint32_to_float";
    "Ndarray.uint64_to_float";
    "Ndarray.used_memory";
    "Numerics.default";
    "Numerics.policy";
    "Operation._get_local_debug_runtime";
    "Operation.add";
    "Operation.centered_uniform1_param_init";
    "Operation.centered_uniform_param_init";
    "Operation.compose_op_of_spec";
    "Operation.concat";
    "Operation.concat_sum";
    "Operation.cos";
    "Operation.default_uniform_param_init";
    "Operation.deinterleave_even";
    "Operation.deinterleave_odd";
    "Operation.einmax1";
    "Operation.embed_dim";
    "Operation.embed_self_id";
    "Operation.embed_symbol";
    "Operation.eq";
    "Operation.exp";
    "Operation.exp2";
    "Operation.fma";
    "Operation.grad";
    "Operation.interleave";
    "Operation.is_prohibit_grad";
    "Operation.le";
    "Operation.log";
    "Operation.log2";
    "Operation.log_2";
    "Operation.lt";
    "Operation.matmul";
    "Operation.mul";
    "Operation.ne";
    "Operation.neg";
    "Operation.not";
    "Operation.offsets";
    "Operation.outer_sum";
    "Operation.pin_counter_spec";
    "Operation.pointdiv";
    "Operation.pointpow";
    "Operation.range_of_shape";
    "Operation.rebatch";
    "Operation.recip";
    "Operation.recip_sqrt";
    "Operation.relu";
    "Operation.reshape";
    "Operation.sat01";
    "Operation.sin";
    "Operation.slice";
    "Operation.sqrt";
    "Operation.stack";
    "Operation.stop_gradient";
    "Operation.stretch";
    "Operation.sub";
    "Operation.tanh";
    "Operation.threefry4x32";
    "Operation.threefry4x32_crypto";
    "Operation.threefry4x32_light";
    "Operation.transpose_op_of_spec";
    "Operation.tropical";
    "Operation.uint4x32_to_prec_uniform";
    "Operation.uint4x32_to_prec_uniform1";
    "Operation.uniform1";
    "Operation.uniform_at";
    "Operation.uniform_at1";
    "Operation.wrap";
    "Ops.bfloat16_to_uint4x32";
    "Ops.byte_to_uint4x32";
    "Ops.c_ptr_to_string";
    "Ops.compare_prec";
    "Ops.compare_voidptr";
    "Ops.double_to_uint4x32";
    "Ops.equal_voidptr";
    "Ops.fp8_to_uint4x32";
    "Ops.half_to_uint4x32";
    "Ops.hum_typ_of_prec";
    "Ops.int32_to_uint4x32";
    "Ops.int64_to_uint4x32";
    "Ops.interpret_ternop";
    "Ops.is_assign_op";
    "Ops.is_binop_infix";
    "Ops.pack_prec";
    "Ops.precision_to_string";
    "Ops.ptr_to_string_hum";
    "Ops.single_to_uint4x32";
    "Ops.uint16_to_uint4x32";
    "Ops.uint32_to_uint4x32";
    "Ops.uint64_to_uint4x32";
    "Ppx_cd.args_for";
    "Ppx_cd.assignment";
    "Ppx_cd.compare_slots";
    "Ppx_cd.empty_comp";
    "Ppx_cd.empty_tns";
    "Ppx_cd.equal_projections_slot";
    "Ppx_cd.guess_pun_hint";
    "Ppx_cd.handle_cases";
    "Ppx_cd.is_unknown";
    "Ppx_cd.make_vb";
    "Ppx_cd.prod_slot_tensor_dims";
    "Ppx_cd.project_p_dims";
    "Ppx_cd.project_p_slot";
    "Ppx_cd.reduce_res_vbs";
    "Ppx_cd.setup_array";
    "Ppx_cd.slot_permutation_suffix";
    "Ppx_cd.slot_to_string";
    "Ppx_cd.translate";
    "Ppx_extend_dsls.transform_dsl_binding";
    "Ppx_ocannl.rules";
    "Ppx_op.dsl_fn";
    "Ppx_op.dsl_open_o";
    "Ppx_op.dsl_path";
    "Ppx_op.is_ndarray_constant_expr";
    "Ppx_op.make_p";
    "Ppx_op.make_vb";
    "Ppx_op.make_vb_nd";
    "Ppx_op.translate_block_tensor";
    "Ppx_op.translate_tdsl";
    "Ppx_shared.axes_or_dims_arg";
    "Ppx_shared.axis_basis_of_type";
    "Ppx_shared.dim_spec_to_string";
    "Ppx_shared.flatten_str";
    "Ppx_shared.non_alphanum_regexp";
    "Ppx_shared.operators";
    "Ppx_shared.opt_pat2string";
    "Ppx_shared.string_literal";
    "Ppx_shared.string_of_constant";
    "Ppx_shared.string_of_pat";
    "Ppx_shared.translate_str";
    "PrintBox_utils._get_local_debug_runtime";
    "PrintBox_utils.concise_float";
    "PrintBox_utils.nolines";
    "PrintBox_utils.render_group";
    "PrintBox_utils.sexp_of_box";
    "Schedulers._get_local_debug_runtime";
    "Schedulers.cpu_mma_limits";
    "Task._get_local_debug_runtime";
    "Tnode._get_local_debug_runtime";
    "Tnode.bounds_scan_worthwhile";
    "Tnode.collapse_consecutive";
    "Tnode.current_namespace";
    "Tnode.fresh_uid";
    "Tnode.hash_fold_t";
    "Tnode.hash_t";
    "Tnode.header";
    "Tnode.initial_default_prec";
    "Tnode.is_alphanum_";
    "Tnode.known_non_virtual";
    "Tnode.known_not_materialized";
    "Tnode.log_accessible_headers";
    "Tnode.most_local_materialized_mode";
    "Tnode.next_uid";
    "Tnode.prec_of_dalayed";
    "Tnode.propose_bounds";
    "Tnode.scan_host_bounds";
    "Tnode.sexp_of_t_map";
    "Tnode.sexp_of_t_set";
    "Tnode.transition_memory_mode";
    "Tnode.validate_padded_numel_contract";
    "Utils._get_local_debug_runtime";
    "Utils.accessed_global_args";
    "Utils.artifacts_subdir";
    "Utils.captured_log_processors";
    "Utils.clean_filename";
    "Utils.cmdline_var_prefixes";
    "Utils.config_table_of_lines";
    "Utils.default_indent";
    "Utils.describe_config_source";
    "Utils.doc_of_sexp";
    "Utils.enable_runtime_debug";
    "Utils.ensure_artifacts_dir";
    "Utils.env_var_reserved_prefixes";
    "Utils.equal_config_level";
    "Utils.filename_concat";
    "Utils.filename_of_parts";
    "Utils.filename_parts";
    "Utils.flush_c_streams";
    "Utils.get_debug_output_channel";
    "Utils.header_sep";
    "Utils.input_line";
    "Utils.input_scan_line";
    "Utils.!@";
    "Utils.log_config_sourcing";
    "Utils.log_config_sourcing_arg";
    "Utils.log_files_dir";
    "Utils.log_trace_tree";
    "Utils.never_capture_stdout";
    "Utils.normalize_exponent";
    "Utils.original_log_level";
    "Utils.pair";
    "Utils.parallel_merge";
    "Utils.parse_profile_payload";
    "Utils.performance_profile_payload";
    "Utils.profile_ineligible_keys";
    "Utils.profile_lookup";
    "Utils.qualified_only_config_keys";
    "Utils.read_cmdline_or_env_var";
    "Utils.reproducible_profile_payload";
    "Utils.same_env_name";
    "Utils.sexp_deep_mem";
    "Utils.sexp_of_atomic_bool";
    "Utils.sexp_of_atomic_int";
    "Utils.split_with_seps";
    "Utils.str_nonempty";
  ]

let in_scan_root path =
  let directory = Stdlib.Filename.dirname path in
  String.equal directory "arrayjit/lib" || String.equal directory "tensor"

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = Dune.base_dir Stdlib.Sys.argv.(1) in
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune.repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  let source_paths = Read.sources_among (List.map arguments ~f:fst) in
  let sources =
    List.map source_paths ~f:(fun source ->
        (source, In_channel.read_all (Map.find_exn on_disk source)))
  in
  let interfaces =
    List.filter (List.map arguments ~f:fst) ~f:(String.is_suffix ~suffix:".mli")
    |> Set.of_list (module String)
  in
  let implementations =
    List.filter sources ~f:(fun (source, _) -> in_scan_root source)
    |> List.filter ~f:(fun (source, _) ->
        match String.chop_suffix source ~suffix:".ml" with
        | Some stem -> not (Set.mem interfaces (stem ^ ".mli"))
        | None -> false)
  in
  if List.is_empty implementations then (
    Verdict.fail "no .mli-less implementation modules found under arrayjit/lib or tensor";
    Stdlib.exit 1);
  let exports =
    List.concat_map implementations ~f:(fun (source, contents) ->
        match Scan.exports_of_source ~source contents with
        | exports -> exports
        | exception exn ->
            Verdict.fail
              (Printf.sprintf "%s does not parse as OCaml, so this scan cannot vouch for it: %s"
                 source (Exn.to_string exn));
            [])
  in
  let references =
    match Scan.references ~exports ~sources with
    | references -> references
    | exception exn ->
        Verdict.fail
          (Printf.sprintf "an OCaml source does not parse, so this scan cannot vouch for it: %s"
             (Exn.to_string exn));
        []
  in
  let counts = Scan.counts ~exports references in
  let exemptions = Set.of_list (module String) exempt_zero_reference_exports in
  let zero_reference =
    List.filter exports ~f:(fun export -> Hashtbl.find_exn counts (Scan.export_key export) = 0)
  in
  let offenders =
    List.filter zero_reference ~f:(fun export -> not (Set.mem exemptions (Scan.export_key export)))
  in
  let stale =
    List.filter exempt_zero_reference_exports ~f:(fun key ->
        match Hashtbl.find counts key with Some 0 -> false | Some _ | None -> true)
  in
  List.iter offenders ~f:(fun export ->
      eprintf
        "%s:%d: %s has no external reference; add an .mli that hides or deliberately publishes it, \
         remove it, or add the exact name to exempt_zero_reference_exports with a reasoned decision\n"
        export.source export.line (Scan.export_key export));
  List.iter stale ~f:(fun key ->
      eprintf
        "%s is a stale dead-export exemption: the value is now referenced or no longer exported\n"
        key);
  eprintf "Scanned %d .mli-less modules, %d source-declared exports, and %d external references.\n"
    (List.length implementations) (List.length exports) (List.length references);
  Verdict.p_empty "every zero-reference implicit export is named" ~over:exports offenders;
  Verdict.p "named dead-export exemptions are unique"
    (Set.length exemptions = List.length exempt_zero_reference_exports);
  Verdict.p_empty "every named dead-export exemption remains necessary"
    ~over:exempt_zero_reference_exports stale;
  Test_utils.Refusal_control_manifest.print "dead_export_scan.ml"
