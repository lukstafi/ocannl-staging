type t = Obj.t Weak.t

let witness x =
  let w = Weak.create 1 in
  Weak.set w 0 (Some (Obj.repr x));
  w

let settle () =
  Gc.full_major ();
  Gc.full_major ();
  Gc.compact ()

(* [Weak.check] reads the observer without taking a reference to the value. *)
let alive w = Weak.check w 0

let reclaimed ?(rounds = 3) w =
  if rounds < 1 then invalid_arg "Gc_witness.reclaimed: rounds must be at least 1";
  let rec go rounds =
    if not (alive w) then true
    else if rounds = 0 then false
    else (
      settle ();
      go (rounds - 1))
  in
  go rounds

let kept ?(rounds = 3) w =
  if rounds < 1 then invalid_arg "Gc_witness.kept: rounds must be at least 1";
  let rec go rounds =
    settle ();
    alive w && (rounds <= 1 || go (rounds - 1))
  in
  go rounds

type allocation = { words : float; drained : float }

let to_words bytes = bytes /. float_of_int (Sys.word_size / 8)

let drained_finalisers f =
  let drain_start = Gc.allocated_bytes () in
  Gc.full_major ();
  let before = Gc.allocated_bytes () in
  let result = f () in
  let after = Gc.allocated_bytes () in
  (result, { words = to_words (after -. before); drained = to_words (before -. drain_start) })

let control_words = 4096

let window_sees_allocation () =
  let (_ : float array), { words; drained = _ } =
    drained_finalisers (fun () -> Sys.opaque_identity (Array.make control_words 0.))
  in
  words >= float_of_int control_words
