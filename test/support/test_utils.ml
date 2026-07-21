(** Portable output helpers for OCANNL tests.

    The Windows and glibc C runtimes format floats differently: Windows prints 3-digit exponents
    ([e+018] vs. Linux's [e+18]) and rounds representable decimal ties away from zero where glibc
    rounds to even ([%.1f] of [2.25] prints [2.3] on Windows, [2.2] on Linux). Tests that print
    floats with raw [%g]/[%e]/[%f] into [.expected] files therefore fail across platforms. Use
    these printers instead — they are portable by construction. *)

open Base
open Stdio

(** [concise_float ~prec v] formats [v] with [prec] decimals, normalizing exponent digits portably.
    Re-export of [Ir.Ndarray.concise_float]. *)
let concise_float = Ir.Ndarray.concise_float

(** [hex_float v] formats [v] with OCaml's [%h] hex-float notation: bit-exact on every platform,
    sidestepping decimal-tie rounding divergence entirely. *)
let hex_float v = Printf.sprintf "%h" v

(** Prints [v] via [concise_float]. *)
let print_float ?(prec = 6) v = printf "%s" (concise_float ~prec v)

(** Prints [v] via [concise_float], followed by a newline. *)
let print_float_ln ?(prec = 6) v = printf "%s\n" (concise_float ~prec v)

(** Prints [vs] separated by [sep] (default a single space) via [concise_float]. *)
let print_floats ?(prec = 6) ?(sep = " ") vs =
  printf "%s" (String.concat ~sep (List.map vs ~f:(concise_float ~prec)))

(** Puts stdout in binary mode. Required when echoing a golden [.expected] file byte-for-byte
    (e.g. in [.missing.ml] backend stubs): text-mode stdout on Windows rewrites ["\n"] to
    ["\r\n"], corrupting the comparison. *)
let set_binary_stdout () = Out_channel.set_binary_mode stdout true
