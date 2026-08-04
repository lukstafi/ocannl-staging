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

          On by default: it strictly increases accuracy relative to per-operator rounding and is the
          precondition for narrow storage being a speedup rather than a pessimization on CPU. Turn
          it off to recover the pre-gh-517 semantics, where every operator rounds to the target
          node's storage precision. The GPU backends are unaffected either way — they have native
          16-bit types and arithmetic, so their compute precision is their storage precision. *)
}
[@@deriving sexp, compare, equal]

let default () =
  {
    tf32_matmuls = Utils.get_global_flag ~default:false ~arg_name:"tf32_matmuls";
    narrow_compute_f32 = Utils.get_global_flag ~default:true ~arg_name:"narrow_compute_f32";
  }
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
