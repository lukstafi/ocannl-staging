open Base

(** Numerics policy (gh-ocannl-478's "option 3" knob): compute-precision decisions that change
    results, so they are chosen by the user — via the global config or {!set_policy} — never by the
    optimizer. Storage precisions live on tensor nodes ({!Tnode.t.storage_prec}); this record
    governs how computations over those storages are carried out. It must be identical across
    sibling autotune candidates: candidate schedules compete on speed, never on numerics (the
    bitwise-parity discipline of the tensorized twins depends on it).

    The record is deliberately open-ended — later compute-precision questions (fast-math
    transcendentals, accumulation widths, fp8 format selection per tensor class, gh-ocannl-492)
    land here rather than growing ad-hoc booleans elsewhere. *)

type t = {
  tf32_matmuls : bool;
      (** Allow tensor-core matmuls over uniform-f32 operands to compute in tf32 on backends with a
          tf32 tile shape (CUDA sm_80+): f32's exponent range with a 10-bit mantissa, accumulation
          in f32. Off by default — opt-in like PyTorch's [allow_tf32], because enabling it silently
          changes numerics. Metal ([simdgroup_float8x8] is genuine f32) and HIP (RDNA WMMA has no
          tf32-like shape) are unaffected. *)
}
[@@deriving sexp, compare, equal]

let default () =
  { tf32_matmuls = Utils.get_global_flag ~default:false ~arg_name:"tf32_matmuls" }

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
