(** The part of the [metal.runtime] library ([Runtime], the Objective-C runtime bindings under
    [metal]) that the stub needs: the one type [metal.mli] names, and the values
    [arrayjit/lib/metal_math_api_runtime.ml] calls. [metal.runtime] ships no interface file -- its
    signature is inferred from [runtime.ml], and [Runtime.Objc] includes all of [Ctypes] -- so this
    is the stub's only hand-written interface, kept to what those two files use, rather than a
    verbatim copy. It is not free to drift: wherever [metal.runtime] is installed,
    [runtime_agreement.ml] (generated in this directory from this file) checks that the installed
    [Runtime] satisfies this signature, on [runtest-runtime] and on every build. *)

module Objc : sig
  type objc_object
  type objc_selector
  type object_t = objc_object Ctypes.structure Ctypes.ptr

  val _SEL : objc_selector Ctypes.structure Ctypes.ptr Ctypes.typ
  val bool : bool Ctypes.typ
  val ( @-> ) : 'a Ctypes.typ -> 'b Ctypes.fn -> ('a -> 'b) Ctypes.fn
  val returning : 'a Ctypes.typ -> 'a Ctypes.fn

  val msg_send :
    self:object_t -> cmd:objc_selector Ctypes.structure Ctypes.ptr -> typ:'a Ctypes.fn -> 'a
end

val alloc_object : string -> Objc.object_t
val init : Objc.object_t -> Objc.object_t
val gc_autorelease : Objc.object_t -> Objc.object_t
val selector : string -> Objc.objc_selector Ctypes.structure Ctypes.ptr
