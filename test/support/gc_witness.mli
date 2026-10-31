(** What the GC did, read by a test without the reading taking part in it: [Weak] witnesses on root
    records, and an allocation window whose count leaves out the finalisers of what died before it.

    A witness is taken in an [[@inline never]] function that returns it and only what the case
    keeps, so no register or stack slot of the caller holds the observed value. One collection is
    not evidence either way -- a value can survive a [Gc.full_major] through a finaliser or an
    ephemeron that the next cycle clears -- so a claim reads the witness over {!settle} rounds:
    [reclaimed] retries until the value is gone, [kept] requires it to survive every round. *)

type t
(** A weak observer of one value. *)

val witness : 'a -> t
(** [witness x] observes [x] without keeping it alive. *)

val settle : unit -> unit
(** One collection round: [Gc.full_major] twice (the second runs what the first one's finalisers
    released), then [Gc.compact]. *)

val reclaimed : ?rounds:int -> t -> bool
(** [true] once the observed value has been collected: read before any collection, then after each
    of up to [rounds] (default 3) {!settle} rounds. *)

val kept : ?rounds:int -> t -> bool
(** [true] when the observed value survives each of [rounds] (default 3, at least 1) {!settle}
    rounds. A test's "kept" control is what shows a witness can report alive at all, so that its
    "reclaimed" claims are not true of a witness that observes nothing. *)

type allocation = {
  words : float;  (** Heap words the function allocated inside the window. *)
  drained : float;
      (** Heap words the finalisers run by the leading [Gc.full_major] allocated, outside the
          window. *)
}

val drained_finalisers : (unit -> 'a) -> 'a * allocation
(** [drained_finalisers f] runs [f] in an allocation window opened after a [Gc.full_major]. The
    domain's allocation counter also counts whatever OCaml code the GC runs inside a window: a large
    allocation can close a major cycle, which calls the finalisers of everything that died earlier
    in the process -- on metal, one [release] message per dead Objective-C object, tens of thousands
    of words (gh-ocannl-1416). Running them first leaves the window holding [f] alone; what they
    allocated is reported as [drained], for the caller's stderr. *)

val window_sees_allocation : unit -> bool
(** The window's "kept" control: a window around a known 4096-word float array counts at least those
    words, so a bound over [words] is not true of a counter that reads nothing. *)
