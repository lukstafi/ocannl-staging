open Base

(** Numerics policy (gh-ocannl-478's "option 3" knob): compute-precision decisions that change
    results, so they are chosen by the user — via the global config or {!set_policy} — never by the
    optimizer. Storage precisions live on tensor nodes ({!Tnode.t.storage_prec}); this record
    governs how computations over those storages are carried out. It must be identical across
    sibling autotune candidates: candidate schedules compete on speed, never on numerics (the
    bitwise-parity discipline of the tensorized twins depends on it).

    The record is deliberately open-ended — later compute-precision questions (fast-math
    transcendentals, accumulation widths, fp8 format selection per tensor class, gh-ocannl-492) land
    here rather than growing ad-hoc booleans elsewhere. *)

type t = {
  tf32_matmuls : bool;
      (** Allow tensor-core matmuls over uniform-f32 operands to compute in tf32 on backends with a
          tf32 tile shape (CUDA sm_80+): f32's exponent range with a 10-bit mantissa, accumulation
          in f32. Off by default — opt-in like PyTorch's [allow_tf32], because enabling it silently
          changes numerics. Metal ([simdgroup_float8x8] is genuine f32) and HIP (RDNA WMMA has no
          tf32-like shape) are unaffected. *)
  narrow_compute_f32 : bool;
      (** Run the arithmetic over narrow-float storage (bf16, fp16, fp8) in f32 on backends that
          have no native narrow arithmetic — the CPU backends, where every narrow operator is an
          explicit widen/op/narrow round-trip anyway (gh-ocannl-517). Storage stays narrow: reads
          widen once at the load, the result narrows once at the store, and the intermediates of an
          assignment keep f32 mantissa instead of being rounded per operator. That is both the
          faithful reading of "16-bit storage with f32 compute" and what makes the vectorized
          renderings — which are f32/f64 shaped — reachable for narrow-storage kernels.

          A reduction accumulator is such an intermediate (gh-ocannl-639): every rendering of an
          accumulation nest — the plain serial fallback included — holds the accumulator at the
          resolved compute precision across the whole nest and narrows it once at the store, so
          the effective accumulation width is this policy's, never a property of which schedule
          happened to place the accumulator in a register. (Narrowing POINTS beyond that single
          one remain a property of a schedule's reduction structure: a k-blocked schedule stores
          storage-precision partials at its block boundaries by construction.)

          On by default: it strictly increases accuracy relative to per-operator rounding and is the
          precondition for narrow storage being a speedup rather than a pessimization on CPU. Turn
          it off to recover the pre-gh-517 semantics, where every operator rounds to the target
          node's storage precision. The GPU backends are unaffected either way — they have native
          16-bit types and arithmetic, so their compute precision is their storage precision. *)
  fp16_arithmetic : bool;
      (** Compute fp16 in fp16 on CPU targets that have native 16-bit arithmetic (ARMv8.2-FP16,
          AVX512-FP16), instead of widening to f32 (gh-ocannl-516). This is the one narrow format a
          CPU can execute natively — bf16 has no C type and no general ARM/x86 arithmetic, and
          stays emulated by design — and where the hardware has it, the lane count doubles against
          f32.

          Off by default, and the asymmetry with {!narrow_compute_f32} is deliberate: computing in
          fp16 keeps intermediates at fp16's 10-bit mantissa and 65504 range, so it trades accuracy
          for throughput, while widening to f32 trades nothing. It also only pays where the target's
          arithmetic is genuinely 16-bit ({!Ir.Backend_intf.hardware_limits.native_fp16_arithmetic});
          on a target that merely promotes to float, turning it on costs accuracy for no speed, so
          the backend ignores it there. *)
}
[@@deriving sexp, compare, equal]

let default () =
  {
    tf32_matmuls = Utils.get_global_flag ~default:false ~arg_name:"tf32_matmuls";
    narrow_compute_f32 = Utils.get_global_flag ~default:true ~arg_name:"narrow_compute_f32";
    fp16_arithmetic = Utils.get_global_flag ~default:false ~arg_name:"fp16_arithmetic";
  }

(** A stable, exhaustive rendering of a policy, for cache keys and digests (gh-ocannl-568). Derived
    from the sexp rather than spelled out field by field, so a knob added to this deliberately
    open-ended record enters every fingerprint by construction — the failure mode being guarded
    against is precisely a policy field that a cache key forgot about. *)
let fingerprint p = Sexp.to_string (sexp_of_t p)

let policy : t option ref = ref None

(** The current policy; reads the global config on first use. *)
let get () =
  match !policy with
  | Some p -> p
  | None ->
      let p = default () in
      policy := Some p;
      p

(** Programmatic override, e.g. per-experiment toggles from training scripts. Call before
    compilation; routines already compiled keep the numerics they were compiled with. *)
let set_policy p = policy := Some p

(** The compute precision the CPU backends resolve a storage precision to under the current policy:
    fp16 stays fp16 only where {!field-fp16_arithmetic} asks for it AND the target's arithmetic is
    genuinely 16-bit (gh-ocannl-516); every other narrow float computes in f32 under
    {!field-narrow_compute_f32} (gh-ocannl-517); everything else is itself. The single source of
    truth shared by [Cc_backend.compute_prec] (emission) and autotune's sketch seeding (the
    candidate pre-filter and the packed-[Stage] tile precisions, gh-ocannl-575) — the gh-ocannl-545
    lesson: when seeding and emission can disagree about a gate, candidates get timed under labels
    their rendering does not honor. *)
let cpu_compute_prec ~native_fp16_arithmetic (prec : Ops.prec) : Ops.prec =
  match prec with
  | Ops.Half_prec _ when (get ()).fp16_arithmetic && native_fp16_arithmetic -> prec
  | _ when Ops.is_narrow_float prec && (get ()).narrow_compute_f32 -> Ops.single
  | _ -> prec
