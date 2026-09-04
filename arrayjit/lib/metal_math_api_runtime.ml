(* The Metal-runtime half of compile-option selection. Keep it separate from the pure builder so
   [arrayjit.ir] remains portable, but share this query between the backend and the standalone RTC
   probe: they must agree about which property sequence the running Objective-C runtime accepts. *)

let compile_options =
  lazy (Runtime.alloc_object "MTLCompileOptions" |> Runtime.init |> Runtime.gc_autorelease)

let get () =
  (* Query an actual compile-options object. On macOS 26.6.2 the nominal [MTLCompileOptions] class
     answers false to [instancesRespondToSelector:] for these protocol properties even though the
     initialized object accepts and reports them; the old class query therefore selected the
     macOS-14 fallback on a modern runtime. *)
  let compile_options = Lazy.force compile_options in
  let selector_available selector =
    Runtime.Objc.msg_send ~self:compile_options
      ~cmd:(Runtime.selector "respondsToSelector:")
      ~typ:Runtime.Objc.(_SEL @-> returning bool)
      (Runtime.selector selector)
  in
  Ir.Compiler_options.metal_math_api ~selector_available
